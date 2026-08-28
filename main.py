"""Run the whole risk engine and print one report.

    python main.py              use the cached prices if they exist
    python main.py --refresh    download fresh prices from Yahoo Finance
    python main.py --no-plots   skip the figures

This file only wires the pieces together. Every calculation lives in data.py,
risk.py, or scenarios.py, so each one can be tested on its own.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import config
import plots
import risk
import scenarios as sc
from data import DataError, compute_returns, fetch_prices, portfolio_returns, weights_array

LINE = "=" * 78


def _heading(text: str) -> None:
    print(f"\n{LINE}\n  {text}\n{LINE}")


def _money(value: float) -> str:
    return f"${value:,.0f}"


def build_results(returns: pd.DataFrame) -> dict:
    """Compute every number the report needs.

    Returns a plain dict, so the notebook can call this and render the same
    figures without repeating any logic.
    """
    tickers = list(returns.columns)
    w = weights_array(config.WEIGHTS, tickers)
    value = config.PORTFOLIO_VALUE

    port_returns = portfolio_returns(returns, config.WEIGHTS)
    cov = risk.covariance_matrix(returns)
    vol = risk.portfolio_volatility(cov, w)

    # Method 3 draws from a mean vector plus the covariance matrix, as the
    # assignment specifies. The zero-mean variant is also computed, because it
    # shares every assumption with the parametric method and so must agree
    # with it. That agreement is the correctness check.
    sample_mean = returns.mean().to_numpy()
    mc_mean = sample_mean if config.MC_USE_SAMPLE_MEAN else np.zeros(len(tickers))

    mc_returns = risk.simulate_portfolio_returns(
        cov, w, mc_mean, config.MC_SIMULATIONS, config.RANDOM_SEED
    )
    mc_zero_mean = risk.simulate_portfolio_returns(
        cov, w, np.zeros(len(tickers)), config.MC_SIMULATIONS, config.RANDOM_SEED
    )

    table = {}
    for name, series in [("Historical", port_returns), ("Monte Carlo", mc_returns)]:
        row = {}
        for c in config.CONFIDENCE_LEVELS:
            row[f"VaR {c:.0%}"] = risk.empirical_var(series, c, value)
            row[f"ES {c:.0%}"] = risk.empirical_expected_shortfall(series, c, value)
        table[name] = row

    table["Parametric"] = {}
    for c in config.CONFIDENCE_LEVELS:
        table["Parametric"][f"VaR {c:.0%}"] = risk.parametric_var(vol, c, value)
        table["Parametric"][f"ES {c:.0%}"] = risk.parametric_expected_shortfall(vol, c, value)

    order = [f"{m} {c:.0%}" for c in config.CONFIDENCE_LEVELS for m in ("VaR", "ES")]
    results = pd.DataFrame(table).T[order].loc[["Historical", "Parametric", "Monte Carlo"]]

    stressed_cov = sc.spike_correlations(cov, config.STRESSED_CORRELATION)
    stressed_vol = risk.portfolio_volatility(stressed_cov, w)

    return {
        "tickers": tickers,
        "weights": w,
        "value": value,
        "portfolio_returns": port_returns,
        "mc_returns": mc_returns,
        "mc_pnl": risk.to_pnl(mc_returns, value),
        "cov": cov,
        "volatility": vol,
        "sample_mean": sample_mean,
        "results": results,
        "mc_zero_mean_var": {
            c: risk.empirical_var(mc_zero_mean, c, value) for c in config.CONFIDENCE_LEVELS
        },
        "base_correlation": sc.correlation_matrix(cov),
        "stressed_correlation": sc.correlation_matrix(stressed_cov),
        "stressed_volatility": stressed_vol,
        "stressed_var": {
            c: risk.parametric_var(stressed_vol, c, value) for c in config.CONFIDENCE_LEVELS
        },
        "scenarios": [
            sc.apply_shock("2008-style equity shock", config.EQUITY_SHOCK,
                           config.WEIGHTS, value),
            sc.uniform_shock("Tech drawdown", config.TICKERS,
                             config.TECH_DRAWDOWN_SHOCK, config.WEIGHTS, value),
        ],
    }


def print_report(prices: pd.DataFrame, returns: pd.DataFrame, r: dict) -> None:
    value = r["value"]

    _heading("PORTFOLIO")
    for t in r["tickers"]:
        print(f"  {t:<8} {config.WEIGHTS[t]:>6.0%}   {_money(config.WEIGHTS[t] * value):>12}")
    print(f"  {'TOTAL':<8} {sum(config.WEIGHTS.values()):>6.0%}   {_money(value):>12}")
    print(f"\n  Data      {prices.index.min():%Y-%m-%d} to {prices.index.max():%Y-%m-%d}")
    print(f"  Prices    {len(prices):,} trading days")
    print(f"  Returns   {len(returns):,} days  (the first day has no prior day)")

    _heading("MARKET STATISTICS (daily)")
    stats = pd.DataFrame({
        "volatility": returns.std() * 100,
        "mean return": returns.mean() * 100,
    })
    print(stats.to_string(float_format=lambda v: f"{v:8.4f} %"))
    print(f"\n  Correlations:\n{r['base_correlation'].to_string(float_format=lambda v: f'{v:7.3f}')}")
    print(f"\n  Portfolio volatility   {r['volatility'] * 100:.4f} % per day")

    naive = float(r["weights"] @ returns.std().to_numpy())
    print(f"  Weighted-average vol   {naive * 100:.4f} % per day")
    print(f"  Diversification gain   {(naive - r['volatility']) * 100:.4f} % lower")

    _heading("PART A - VALUE AT RISK  (1 day, USD)")
    print(r["results"].to_string(float_format=lambda v: f"{v:>12,.0f}"))
    print("\n  VaR 95% means: on the worst 5% of days, the loss is more than this.")
    print("  ES is the average loss on those days. It is always the larger number.")

    _heading("CROSS-CHECK: Monte Carlo against Parametric")
    print("  Method 3 uses the sample mean vector, as the assignment specifies.")
    print("  Method 2 uses no mean term, as its formula specifies. So they differ")
    print("  slightly by design. Setting the Monte Carlo mean to zero removes that")
    print("  difference, and then the two must agree.\n")
    for c in config.CONFIDENCE_LEVELS:
        p = r["results"].loc["Parametric", f"VaR {c:.0%}"]
        z = r["mc_zero_mean_var"][c]
        gap = abs(z - p) / p
        print(f"  {c:.0%}   parametric {_money(p):>10}   "
              f"MC zero-mean {_money(z):>10}   gap {gap:.2%}")

    _heading("PART B - STRESS TESTS")
    for s in r["scenarios"]:
        print(f"  {s.name:<28} {s.portfolio_return:>8.2%}   loss {_money(s.loss):>12}")

    print(f"\n  Correlation spike (all pairs to {config.STRESSED_CORRELATION})")
    print(f"    Per-stock volatility is unchanged. Only the relationships move.")
    print(f"    Portfolio volatility  {r['volatility'] * 100:.4f} %  ->  "
          f"{r['stressed_volatility'] * 100:.4f} %")
    for c in config.CONFIDENCE_LEVELS:
        base = r["results"].loc["Parametric", f"VaR {c:.0%}"]
        stressed = r["stressed_var"][c]
        print(f"    Parametric VaR {c:.0%}   {_money(base):>10}  ->  {_money(stressed):>10}"
              f"   (+{(stressed / base - 1) * 100:.1f} %)")

    _heading("SCENARIOS AGAINST VaR")
    worst = r["results"].loc["Parametric", f"VaR {max(config.CONFIDENCE_LEVELS):.0%}"]
    for s in r["scenarios"]:
        print(f"  {s.name:<28} {_money(s.loss):>12}   "
              f"{s.loss / worst:.1f}x the 99% VaR")
    print("\n  A 99% VaR is meant to be a once-in-a-hundred-days loss.")
    print("  A modest crisis is several times worse. That is why both are needed.")


def make_plots(r: dict) -> list:
    value = r["value"]
    var95_99 = {c: r["results"].loc["Historical", f"VaR {c:.0%}"]
                for c in config.CONFIDENCE_LEVELS}
    parametric_var = {c: r["results"].loc["Parametric", f"VaR {c:.0%}"]
                      for c in config.CONFIDENCE_LEVELS}

    return [
        plots.plot_return_distribution(
            r["portfolio_returns"], var95_99, value,
            config.PLOTS_DIR / "1_return_distribution.png"),
        plots.plot_method_comparison(
            r["results"], config.PLOTS_DIR / "2_method_comparison.png"),
        plots.plot_correlation_shift(
            r["base_correlation"], r["stressed_correlation"],
            config.PLOTS_DIR / "3_correlation_spike.png"),
        plots.plot_scenarios_vs_var(
            parametric_var, {s.name: s.loss for s in r["scenarios"]},
            config.PLOTS_DIR / "4_scenarios_vs_var.png"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio VaR and stress testing.")
    parser.add_argument("--refresh", action="store_true",
                        help="download fresh prices instead of using the cache")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    args = parser.parse_args()

    try:
        prices = fetch_prices(
            config.TICKERS, config.START_DATE, config.END_DATE,
            cache_path=config.PRICE_CACHE, refresh=args.refresh,
        )
        returns = compute_returns(prices)
        results = build_results(returns)
        print_report(prices, returns, results)

        if not args.no_plots:
            _heading("PLOTS")
            for path in make_plots(results):
                print(f"  saved  {path.relative_to(config.PROJECT_ROOT)}")

    except (DataError, risk.RiskError, sc.ScenarioError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
