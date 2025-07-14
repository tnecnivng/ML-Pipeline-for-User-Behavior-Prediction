# src/ubp/data.py
from pathlib import Path
import pandas as pd

RAW_CSV  = Path("data/raw/retail_2010_2011.csv")
RAW_XLSX = Path("data/raw/online_retail_II.xlsx")   

# 1) LOAD RAW DATA
def load_raw() -> pd.DataFrame:
    """
    Return the raw Online-Retail II data as a DataFrame.

    Priority: CSV ➜ XLSX. Raises if nothing is found.
    """
    if RAW_CSV.exists():
        return pd.read_csv(RAW_CSV, parse_dates=["InvoiceDate"])

    if RAW_XLSX.exists():
        return pd.read_excel(
            RAW_XLSX,
            sheet_name="Year 2010-2011",         
            parse_dates=["InvoiceDate"],
        )

    raise FileNotFoundError(
        f"Neither {RAW_CSV} nor {RAW_XLSX} was found 🤷‍♂️\n"
        "Please place the raw file in one of those locations."
    )

# 2) BASIC CLEANING 
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Customer ID": "CustomerID"})

    df = df[df["Quantity"] > 0].copy()
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype("string")
    return df


# 3) FEATURE ENGINEERING
def load_dataset() -> pd.DataFrame:
    from ubp.features import build_feature_table

    raw  = load_raw()
    df   = clean_raw(raw)
    dataset = build_feature_table(df)
    return dataset

if __name__ == "__main__":
    raw = load_raw()
    print("RAW cols:", raw.columns.tolist())
    dataset = load_dataset()
    print("✅ Final dataset shape:", dataset.shape)
    print(dataset.head())
