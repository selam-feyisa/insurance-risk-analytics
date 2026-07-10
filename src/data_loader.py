"""
data_loader.py

Utilities for loading and lightly preparing the ACIS insurance dataset.
"""

from pathlib import Path
import pandas as pd

# Columns that are known to have mixed/object dtypes on raw read and
# should be forced to a consistent dtype to avoid pandas DtypeWarning.
DTYPE_OVERRIDES = {
    "CapitalOutstanding": "object",
}

DATE_COLUMNS = ["TransactionMonth", "VehicleIntroDate"]


def load_insurance_data(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load the insurance dataset from a CSV file.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file (e.g. 'data/cleaned_data.csv').
    parse_dates : bool, default True
        If True, attempt to parse known date columns.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {path}")

    df = pd.read_csv(path, low_memory=False, dtype=DTYPE_OVERRIDES)

    if parse_dates:
        for col in DATE_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def add_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure LossRatio and Margin columns exist, computing them if missing.

    LossRatio = TotalClaims / TotalPremium
    Margin    = TotalPremium - TotalClaims

    Rows where TotalPremium == 0 get a LossRatio of NaN to avoid
    division-by-zero errors.
    """
    df = df.copy()

    if "LossRatio" not in df.columns:
        df["LossRatio"] = df["TotalClaims"] / df["TotalPremium"].replace(0, pd.NA)

    if "Margin" not in df.columns:
        df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    return df


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy summary of missing values per column: count and percentage.
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100

    report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct.round(3),
    })
    report = report[report["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )
    return report


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of numeric column names in the dataframe."""
    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of categorical/object column names in the dataframe."""
    return df.select_dtypes(include=["object", "bool"]).columns.tolist()
