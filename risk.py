"""Value-at-Risk and Expected Shortfall.

Three VaR methods, and Expected Shortfall for each.

The three methods answer one question: how much can this portfolio lose in one
day, at a stated confidence? They differ only in where the loss distribution
comes from.

    Historical  - the 1254 days that really happened.
    Parametric  - a bell curve fitted to those days.
    Monte Carlo - 100000 days drawn from that same bell curve.

Historical and Monte Carlo then share the same code. Both read a percentile
off a column of portfolio returns. Only the source of the rows differs. So
empirical_var and empirical_expected_shortfall serve both methods.

Every function here is pure. It takes numbers and returns numbers. Nothing
reads config, and nothing touches the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class RiskError(Exception):
    """Raised when a risk input is unusable."""


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise RiskError(f"Confidence must be between 0 and 1. Got {confidence}.")


# --- Shared building blocks --------------------------------------------------


def covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the covariance matrix of the daily returns.

    This one object carries all the risk information. Each stock's variance
    sits on the diagonal. The co-movement of each pair sits off it.

    The parametric method, the Monte Carlo method, and the correlation spike
    scenario all read from this matrix.
    """
    if len(returns) < 2:
        raise RiskError(f"Need at least 2 return rows for a covariance. Got {len(returns)}.")
    return returns.cov()


def portfolio_volatility(cov: pd.DataFrame, weights: np.ndarray) -> float:
    """Return the portfolio's daily volatility, as sqrt(w' Sigma w).

    Written out for three stocks, w' Sigma w is:

        w1^2*var1 + w2^2*var2 + w3^2*var3          <- each stock's own risk
      + 2*w1*w2*cov(1,2)                           <- the pair terms
      + 2*w1*w3*cov(1,3)
      + 2*w2*w3*cov(2,3)

    The pair terms are where correlation enters. They are the reason the
    portfolio is calmer than the weighted average of its parts, and they are
    the only terms the correlation spike scenario changes.

    This equals the plain standard deviation of the portfolio return series.
    The matrix form is used because the scenarios need to alter Sigma first.
    """
    if cov.shape[0] != len(weights):
        raise RiskError(
            f"Covariance is {cov.shape[0]}x{cov.shape[1]} but there are "
            f"{len(weights)} weights."
        )
    variance = float(weights @ cov.to_numpy() @ weights)
    if variance < 0:
        raise RiskError(f"Portfolio variance is negative ({variance}). Covariance is invalid.")
    return float(np.sqrt(variance))


# --- Empirical methods: Historical Simulation and Monte Carlo ----------------


def empirical_var(returns: pd.Series, confidence: float, portfolio_value: float) -> float:
    """Return the VaR read straight off a column of portfolio returns.

    Sort the returns. Find the point where only (1 - confidence) of the days
    are worse. Translate that return into a dollar loss.

    This serves two of the three methods:

        Historical Simulation VaR - pass the 1254 real portfolio returns.
        Monte Carlo VaR           - pass the 100000 simulated ones.

    The methods differ in where the rows come from, not in what happens next.

    The result is a positive number, because VaR is a loss.
    """
    _check_confidence(confidence)
    if len(returns) == 0:
        raise RiskError("Cannot compute VaR from an empty return series.")

    quantile = float(np.quantile(returns.to_numpy(), 1.0 - confidence))
    return -quantile * portfolio_value


def empirical_expected_shortfall(
    returns: pd.Series, confidence: float, portfolio_value: float
) -> float:
    """Return the average loss on the days worse than VaR.

    VaR is the edge of the tail. Expected Shortfall is the mean of the tail.
    Same sorted list, different question.

    Like empirical_var, this serves both the historical and the Monte Carlo
    method. The result is a positive dollar loss, and it is always larger than
    the matching VaR.
    """
    _check_confidence(confidence)
    if len(returns) == 0:
        raise RiskError("Cannot compute Expected Shortfall from an empty return series.")

    values = returns.to_numpy()
    cutoff = np.quantile(values, 1.0 - confidence)
    tail = values[values <= cutoff]

    if len(tail) == 0:
        raise RiskError(
            f"No observations fall in the worst {(1 - confidence):.1%} of "
            f"{len(values)} returns. Use a longer history or a lower confidence."
        )

    return -float(tail.mean()) * portfolio_value


# --- Parametric method -------------------------------------------------------


def parametric_var(volatility: float, confidence: float, portfolio_value: float) -> float:
    """Return VaR from the normal assumption: z * sigma * value.

    z is the number of standard deviations that leaves only (1 - confidence)
    of a bell curve to its left. It is a constant of the normal distribution,
    like pi is a constant of a circle: 1.645 at 95%, 2.326 at 99%.

    The mean return is taken as zero. A daily mean is about 0.05%, while
    z * sigma is about 2.6%, so the mean moves the answer very little and is
    estimated poorly from noisy data. The assignment states the formula this
    way as well.
    """
    _check_confidence(confidence)
    if volatility < 0:
        raise RiskError(f"Volatility cannot be negative. Got {volatility}.")

    z = stats.norm.ppf(confidence)
    return z * volatility * portfolio_value


def parametric_expected_shortfall(
    volatility: float, confidence: float, portfolio_value: float
) -> float:
    """Return Expected Shortfall under the normal assumption.

        ES = sigma * pdf(z) / (1 - confidence) * value

    The multiplier replaces the z used for VaR. It is 2.063 at 95% against a
    VaR z of 1.645, and 2.668 at 99% against 2.326. So under a bell curve, ES
    runs about 25% above VaR at 95%, and about 15% above it at 99%.
    """
    _check_confidence(confidence)
    if volatility < 0:
        raise RiskError(f"Volatility cannot be negative. Got {volatility}.")

    z = stats.norm.ppf(confidence)
    multiplier = stats.norm.pdf(z) / (1.0 - confidence)
    return multiplier * volatility * portfolio_value


# --- Monte Carlo simulation --------------------------------------------------


def simulate_portfolio_returns(
    cov: pd.DataFrame,
    weights: np.ndarray,
    mean: np.ndarray,
    n_simulations: int,
    seed: int,
) -> pd.Series:
    """Draw n_simulations fake trading days and return the portfolio return of each.

    The draws are multivariate normal, built from a mean vector and the
    covariance matrix, as the assignment specifies.

    Drawing the three stocks together is the whole point. Independent draws
    would let the stocks cancel each other far more often than they really do.
    Measured on this data, that mistake understates the risk by 29%.

    Underneath, numpy finds a mixing table A where A @ A.T equals Sigma, then
    multiplies independent noise by A. Because one column of A feeds more than
    one stock, that shared ingredient is what creates the correlation.

    Two algorithms can find such an A: Cholesky decomposition and singular
    value decomposition. numpy's default here is SVD, which is more robust when
    the covariance matrix is close to singular. Either gives draws with the
    same covariance.

    The mean argument controls a useful comparison:

      - The sample mean vector is what the assignment asks for. A positive
        drift shifts the whole distribution up, so it lowers VaR slightly.
      - A zero vector matches the parametric formula, which carries no mean
        term. With zero mean, both methods share the same Sigma and the same
        assumption, so the two VaR figures must agree within about 1%. That
        agreement is the best correctness check in the project.

    The seed is fixed, so every run gives identical numbers.
    """
    if n_simulations < 1:
        raise RiskError(f"Need at least 1 simulation. Got {n_simulations}.")
    if cov.shape[0] != len(weights):
        raise RiskError(
            f"Covariance is {cov.shape[0]}x{cov.shape[1]} but there are "
            f"{len(weights)} weights."
        )
    if len(mean) != len(weights):
        raise RiskError(
            f"The mean vector has {len(mean)} entries but there are {len(weights)} weights."
        )

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.asarray(mean, dtype=float), cov.to_numpy(), size=n_simulations)

    return pd.Series(draws @ weights, name="simulated_portfolio_return")


def to_pnl(returns: pd.Series, portfolio_value: float) -> pd.Series:
    """Turn a series of portfolio returns into a series of dollar P&L.

    The assignment asks for the simulated portfolio P&L distribution, so this
    makes that step explicit rather than folding it into the VaR call.
    """
    return returns * portfolio_value
