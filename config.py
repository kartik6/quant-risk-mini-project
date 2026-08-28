"""Every tunable value for the risk engine.

Keep all inputs here. The other modules read from this file and hold no
constants of their own. This makes the run reproducible and lets a reader
change the portfolio without a search through the code.
"""

from pathlib import Path

# --- Portfolio ---------------------------------------------------------------

TICKERS = ["AAPL", "GOOGL", "MSFT"]

# Weights map a ticker to its share of the portfolio. A dict prevents the
# order bug that a bare list invites. The weights must sum to 1.0.
WEIGHTS = {
    "AAPL": 0.40,
    "GOOGL": 0.35,
    "MSFT": 0.25,
}

PORTFOLIO_VALUE = 1_000_000.0

# --- Data window -------------------------------------------------------------

# Fixed dates, not a rolling window. A rolling window changes every result on
# every run, so the numbers in the README would not match a later run.
# Five years gives about 1260 trading days. The 99% historical VaR reads the
# 13th worst day of that sample, so a shorter window makes the tail too shaky.
START_DATE = "2021-08-30"
END_DATE = "2026-08-29"

# --- Risk parameters ---------------------------------------------------------

CONFIDENCE_LEVELS = [0.95, 0.99]

# The assignment writes Method 2 as VaR = z * sigma * value, with no mean term.
# That sets the average daily return to zero. A daily mean is about 0.05%,
# while z*sigma is about 2.6%, so the mean changes the answer very little and
# is estimated poorly from noisy data.
PARAMETRIC_ZERO_MEAN = True

# --- Monte Carlo -------------------------------------------------------------

MC_SIMULATIONS = 100_000
RANDOM_SEED = 42

# The assignment asks Method 3 to draw from a "mean vector + covariance matrix",
# so the Monte Carlo uses the real sample mean. Note that the assignment states
# Method 2 without a mean, so the two methods differ slightly by design.
#
# The run also reports a zero-mean Monte Carlo. That variant shares every
# assumption with the parametric method, so the two must agree within about 1%.
# The agreement is the best correctness check in the project.
MC_USE_SAMPLE_MEAN = True

# --- Stress scenarios --------------------------------------------------------

# Scenario 1: a 2008-style equity shock. One day of losses, per stock.
EQUITY_SHOCK = {
    "AAPL": -0.08,
    "GOOGL": -0.07,
    "MSFT": -0.06,
}

# Scenario 2: a tech drawdown. Every stock falls the same amount.
TECH_DRAWDOWN_SHOCK = -0.10

# Scenario 3: a correlation spike. Volatilities stay. Every pair moves to this.
STRESSED_CORRELATION = 0.85

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
PRICE_CACHE = DATA_DIR / "prices.csv"
