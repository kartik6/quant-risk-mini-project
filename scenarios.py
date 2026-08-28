"""Stress tests and scenario analysis.

A stress test is not statistical. VaR reads history and fits a distribution,
then reports a probability. A stress test asks a direct question with no
probability at all: this happens, what do I lose?

That matters because VaR cannot warn about an event outside its data window.
The window here starts in 2021, so it holds no 2008 and no COVID crash. A
stress test lets me ask about those anyway. I supply the disaster and the
arithmetic supplies the loss.

Three scenarios:

    1. Equity shock       - a different fall for each stock.
    2. Tech drawdown      - the same fall for every stock.
    3. Correlation spike  - no price move at all. The stocks simply start
                            moving together, and risk rises on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import portfolio_returns


class ScenarioError(Exception):
    """Raised when a scenario input is unusable."""


@dataclass(frozen=True)
class ScenarioResult:
    """The outcome of one price shock.

    portfolio_return is signed, so a fall is negative. pnl is the signed dollar
    change. loss flips the sign, because a loss is quoted as a positive number.
    """

    name: str
    portfolio_return: float
    pnl: float

    @property
    def loss(self) -> float:
        return -self.pnl


# --- Price shocks: scenarios 1 and 2 -----------------------------------------


def apply_shock(
    name: str,
    shock: dict[str, float],
    weights: dict[str, float],
    portfolio_value: float,
) -> ScenarioResult:
    """Apply a one-day return shock and return the resulting portfolio P&L.

    The shock is a return per stock, so -0.08 means that stock falls 8%.

    This builds a single-row table and hands it to portfolio_returns, the same
    function the VaR methods use. A scenario is just one invented day fed
    through the normal pipeline. Reusing that function means a scenario and a
    VaR can never disagree about how the portfolio aggregates.
    """
    missing = [t for t in weights if t not in shock]
    if missing:
        raise ScenarioError(f"No shock given for: {', '.join(missing)}")

    tickers = list(weights.keys())
    one_day = pd.DataFrame([[shock[t] for t in tickers]], columns=tickers)

    portfolio_return = float(portfolio_returns(one_day, weights).iloc[0])
    return ScenarioResult(
        name=name,
        portfolio_return=portfolio_return,
        pnl=portfolio_return * portfolio_value,
    )


def uniform_shock(
    name: str,
    tickers: list[str],
    move: float,
    weights: dict[str, float],
    portfolio_value: float,
) -> ScenarioResult:
    """Apply the same return shock to every stock.

    The weights cancel here, because they sum to 1.0. A uniform fall of 10%
    costs 10% of the portfolio whatever the weighting. That makes this a free
    check on the aggregation code.
    """
    return apply_shock(name, {t: move for t in tickers}, weights, portfolio_value)


# --- Correlation spike: scenario 3 -------------------------------------------


def volatilities(cov: pd.DataFrame) -> np.ndarray:
    """Return each stock's volatility, the square root of the diagonal of Sigma."""
    return np.sqrt(np.diag(cov.to_numpy()))


def correlation_matrix(cov: pd.DataFrame) -> pd.DataFrame:
    """Pull the correlation matrix out of the covariance matrix.

    Covariance and correlation measure the same thing in different units:

        cov(A,B) = rho(A,B) * vol_A * vol_B

    So dividing each covariance by the two volatilities recovers rho.
    """
    vols = volatilities(cov)
    if np.any(vols <= 0):
        raise ScenarioError("A stock has zero or negative volatility. Covariance is invalid.")
    return pd.DataFrame(
        cov.to_numpy() / np.outer(vols, vols),
        index=cov.index,
        columns=cov.columns,
    )


def rebuild_covariance(vols: np.ndarray, corr: pd.DataFrame) -> pd.DataFrame:
    """Put a covariance matrix back together from volatilities and correlations."""
    return pd.DataFrame(
        np.outer(vols, vols) * corr.to_numpy(),
        index=corr.index,
        columns=corr.columns,
    )


def spike_correlations(cov: pd.DataFrame, target: float) -> pd.DataFrame:
    """Return a covariance matrix with every pairwise correlation set to target.

    Each stock keeps its own volatility. Only the relationships change.

    The decomposition is the point of this function. Covariance mixes
    volatility and correlation together, so to reach a correlation of exactly
    0.85 while holding volatility fixed, the two have to be separated first.
    Doing that in the open makes the code state what the scenario means.

    The result shows that risk can rise with no stock becoming more volatile.
    In a crisis, correlations run toward 1 and diversification stops working,
    which is exactly when it is needed.
    """
    if not -1.0 <= target <= 1.0:
        raise ScenarioError(f"A correlation must be between -1 and 1. Got {target}.")

    vols = volatilities(cov)

    stressed_corr = np.full(cov.shape, float(target))
    np.fill_diagonal(stressed_corr, 1.0)  # a stock is perfectly correlated with itself

    return rebuild_covariance(
        vols,
        pd.DataFrame(stressed_corr, index=cov.index, columns=cov.columns),
    )
