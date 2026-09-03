"""Fetch and clean market data.

This module owns the messy outside world: the network, missing rows, and the
column layout that yfinance returns. It hands the rest of the project a tidy
table of daily returns. No risk maths lives here.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

WEIGHT_SUM_TOLERANCE = 1e-6
DOWNLOAD_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 2.0


class DataError(Exception):
    """Raised when the market data is absent or unusable."""


# --- Weights -----------------------------------------------------------------


def validate_weights(weights: dict[str, float], tickers: list[str]) -> None:
    """Check the weights against the tickers. Raise a clear error if they clash.

    Most failures in this project come from a config edit, not from the maths.
    These two checks catch them at the start of the run.
    """
    missing = [t for t in tickers if t not in weights]
    if missing:
        raise DataError(f"No weight given for: {', '.join(missing)}")

    extra = [t for t in weights if t not in tickers]
    if extra:
        raise DataError(f"Weight given for a stock that is not held: {', '.join(extra)}")

    total = sum(weights[t] for t in tickers)
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise DataError(f"Weights must sum to 1.0. They sum to {total:.6f}.")


def weights_array(weights: dict[str, float], tickers: list[str]) -> np.ndarray:
    """Return the weights as an array in ticker order.

    Every matrix operation later assumes this order. Building the array here,
    from the ticker list, removes the risk that a weight lands on the wrong
    stock.
    """
    validate_weights(weights, tickers)
    return np.array([weights[t] for t in tickers], dtype=float)


# --- Prices ------------------------------------------------------------------


def _download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Call yfinance once and return the adjusted close prices.

    auto_adjust=True is passed on purpose. It folds the dividend and split
    adjustment into the Close column and removes the separate Adj Close column.
    The default for this flag has changed between yfinance versions, so the
    code states it rather than relying on the default.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise DataError(
            f"Yahoo Finance returned no data for {tickers} "
            f"between {start} and {end}. Check the network and the date range."
        )

    # A multi-ticker request returns two column levels: (Price, Ticker).
    # Select the Close level, so nothing downstream sees that layout.
    prices = raw["Close"] if raw.columns.nlevels == 2 else raw[["Close"]]

    # Force the requested order. yfinance sorts the tickers alphabetically.
    prices = prices.reindex(columns=tickers)

    # yfinance names the column index "Ticker". Reading the cache back from CSV
    # does not. Clearing the name keeps a cached run and a --refresh run
    # printing the same tables.
    prices.columns.name = None
    prices.index.name = "Date"
    return prices


def _failed_tickers(prices: pd.DataFrame, tickers: list[str]) -> list[str]:
    """Return the tickers that carry no usable data.

    A failed ticker does not raise. yfinance returns a column of NaN instead,
    which would flow silently into the returns and corrupt every result.
    """
    return [t for t in tickers if t not in prices.columns or prices[t].isna().all()]


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_path=None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return daily adjusted close prices, one column per ticker.

    The result is cached to a CSV file. Yahoo Finance is a free and unofficial
    endpoint. It rate-limits and it goes down. The cache lets the project run
    with no network, and it freezes the numbers that the README reports.
    Pass refresh=True to force a new download.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        prices = prices.reindex(columns=tickers)
        failed = _failed_tickers(prices, tickers)
        if failed:
            raise DataError(
                f"The price cache at {cache_path} has no data for: "
                f"{', '.join(failed)}. Delete the file or run with --refresh."
            )
        return prices

    # yfinance keeps a local SQLite cache that can report "database is locked"
    # under load. That failure is transient, so retry before giving up.
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            prices = _download(tickers, start, end)
            failed = _failed_tickers(prices, tickers)
            if not failed:
                break
            last_error = DataError(f"Yahoo Finance returned no data for: {', '.join(failed)}")
        except DataError as error:
            last_error = error
        if attempt < DOWNLOAD_ATTEMPTS:
            time.sleep(RETRY_PAUSE_SECONDS)
    else:
        raise DataError(f"The download failed after {DOWNLOAD_ATTEMPTS} attempts. {last_error}")

    # Drop any day where a stock has no price. The three stocks share the US
    # trading calendar, so this removes very few rows.
    prices = prices.dropna(how="any")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)

    return prices


# --- Returns -----------------------------------------------------------------


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Turn prices into daily simple returns.

    A simple return is (price_today / price_yesterday) - 1. It answers the
    question "what fraction did my money change today?".

    The code uses simple returns, not log returns, on purpose. Only simple
    returns add across stocks, so the weighted sum in portfolio_returns is
    exact rather than an approximation.

    The first row has no prior day, so its return is undefined. That row is
    dropped. A table of 1260 prices gives 1259 returns.
    """
    if len(prices) < 2:
        raise DataError(f"Need at least 2 price rows to compute a return. Got {len(prices)}.")

    return prices.pct_change().dropna(how="any")


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Collapse the per-stock returns into one portfolio return per day.

    The portfolio return is the weighted sum of the stock returns:

        r_portfolio = 0.40*r_AAPL + 0.35*r_GOOGL + 0.25*r_MSFT

    This function is the backbone of the project. Historical VaR sorts its
    output. The stress scenarios call it with one invented day instead of a
    real one. Keeping the weighted sum in a single place means the scenarios
    and the VaR can never disagree about how the portfolio aggregates.
    """
    w = weights_array(weights, list(returns.columns))
    return pd.Series(
        returns.to_numpy() @ w,
        index=returns.index,
        name="portfolio_return",
    )
