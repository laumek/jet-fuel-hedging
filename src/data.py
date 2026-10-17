import pandas as pd

def load_jet_fuel(path: str, start: str = "2005-01-01") -> pd.Series:
    """
    Load daily US Gold Coast jet fuel prices ($/gallon) from a FRED CSV.
    Returns a clean daily price Series indexed by date, starting at `start`.
    """
    pass