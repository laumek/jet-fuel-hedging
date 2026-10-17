"""Loading and cleaning US Gulf Coast jet fuel prices"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_FILE = DATA_DIR / "DJFUELUSGULF.csv"

# 2005-06 contain multi-week gaps in the FRED series, so the analysis starts in 2007. 2007 also gives a full year of history to measure volatility before the first hedge programme (January 2008).
DEFAULT_START = "2007-01-01"


def load_jet_fuel(path: str | Path = DEFAULT_FILE, start: str = DEFAULT_START) -> pd.Series:
    """
    Load daily jet fuel prices ($/gallon) from a FRED CSV.

    Missing values (corresponding to US holidays and weekends) are dropped rather than
    forward-filled as the latter would understate volatility and underprice protection
    """
    raw = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    prices = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    prices = prices.loc[start:].dropna()

    if (prices <= 0).any():
        raise ValueError("Found non-positive prices; check the source file.")

    prices.index.name = "date"
    return prices.rename("price")


def count_missing(path: str | Path = DEFAULT_FILE, start: str = DEFAULT_START) -> int:
    """Number of missing / non-numeric values from `start` onwards, before cleaning."""
    raw = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    values = pd.to_numeric(raw.iloc[:, 0], errors="coerce").loc[start:]
    return int(values.isna().sum())


def find_gaps(prices: pd.Series, min_days: int = 5) -> pd.Series:
    """Gaps (in calendar days) between consecutive observations of at least `min_days`."""
    gap_days = prices.index.to_series().diff().dt.days
    return gap_days[gap_days >= min_days]


def to_monthly(prices: pd.Series, complete_only: bool = True) -> pd.Series:
    """
    Monthly average prices (fuel contracts typically settle on the monthly average).

    If `complete_only`, the final month is dropped when the data ends before
    that month's last business day, so a partial month is never settled.
    """
    monthly = prices.resample("ME").mean()
    if complete_only and len(monthly):
        last_obs = prices.index[-1]
        last_bday = last_obs + pd.offsets.BMonthEnd(0)
        if last_obs.normalize() < last_bday.normalize():
            monthly = monthly.iloc[:-1]
    return monthly.rename("monthly_avg")
