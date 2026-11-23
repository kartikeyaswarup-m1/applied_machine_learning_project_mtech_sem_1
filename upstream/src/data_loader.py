"""
Data loading and cleaning for the Online Retail II dataset.

The raw file has two sheets (2009-2010 and 2010-2011). We concatenate them,
standardise column names, parse dates, engineer a few columns, and split out
clean "sales" transactions from cancellations / non-product adjustments.

A pickled cache is written on first run so downstream modules are fast.
"""
import pandas as pd
import numpy as np
from . import config


# Non-product stock codes that represent fees, postage, manual adjustments, etc.
NON_PRODUCT_CODES = {
    "POST", "D", "DOT", "M", "C2", "BANK CHARGES", "BANKCHARGES",
    "AMAZONFEE", "S", "CRUK", "PADS", "B", "ADJUST", "ADJUST2",
    "TEST001", "TEST002", "gift_0001", "GIFT",
}


def _read_raw() -> pd.DataFrame:
    print("Reading raw Excel (two sheets)...")
    xl = pd.ExcelFile(config.RAW_XLSX)
    frames = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
    df = pd.concat(frames, ignore_index=True)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Customer ID": "CustomerID",
        "Price": "UnitPrice",
    })
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype(str).str.strip()

    # A cancellation invoice starts with "C"
    df["IsCancellation"] = df["Invoice"].str.startswith("C")
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Date parts useful for EDA
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour
    return df


def load_data(force: bool = False) -> pd.DataFrame:
    """Return a cleaned 'sales' DataFrame (positive qty, valid price, known customer)."""
    if config.CACHE_FILE.exists() and not force:
        print(f"Loading cleaned data from cache: {config.CACHE_FILE.name}")
        return pd.read_pickle(config.CACHE_FILE)

    raw = _read_raw()
    df = _clean(raw)

    before = len(df)
    sales = df[
        (~df["IsCancellation"])
        & (df["Quantity"] > 0)
        & (df["UnitPrice"] > 0)
        & (df["CustomerID"].notna())
    ].copy()
    sales["CustomerID"] = sales["CustomerID"].astype(int)
    print(f"Cleaning: {before:,} raw rows -> {len(sales):,} valid sales rows "
          f"({before - len(sales):,} dropped).")

    sales.to_pickle(config.CACHE_FILE)

    # Write a small, committable sample so the repo has example data
    sample = sales.sample(n=min(2000, len(sales)), random_state=42).sort_values("InvoiceDate")
    sample.to_csv(config.SAMPLE_CSV, index=False)
    print(f"Wrote {len(sample):,}-row sample -> {config.SAMPLE_CSV.name}")
    return sales


def product_transactions(sales: pd.DataFrame) -> pd.DataFrame:
    """Sales filtered to real products only (drops fees/postage/adjustments)."""
    mask = ~sales["StockCode"].str.upper().isin({c.upper() for c in NON_PRODUCT_CODES})
    return sales[mask].copy()


def dataset_overview(sales: pd.DataFrame) -> dict:
    return {
        "rows": len(sales),
        "customers": sales["CustomerID"].nunique(),
        "invoices": sales["Invoice"].nunique(),
        "products": sales["StockCode"].nunique(),
        "countries": sales["Country"].nunique(),
        "date_min": sales["InvoiceDate"].min(),
        "date_max": sales["InvoiceDate"].max(),
        "total_revenue": sales["TotalPrice"].sum(),
    }


if __name__ == "__main__":
    s = load_data(force=True)
    from pprint import pprint
    pprint(dataset_overview(s))
