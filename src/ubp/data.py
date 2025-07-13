# src/ubp/data.py
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------------------------------
# RAW-DATA LOCATIONS — adjust if you keep files elsewhere
# ------------------------------------------------------------------------------------------
RAW_CSV  = Path("data/raw/retail_2010_2011.csv")
RAW_XLSX = Path("data/raw/online_retail_II.xlsx")   # rename if your Excel file differs

# ------------------------------------------------------------------------------------------
# 1) LOAD RAW DATA
# ------------------------------------------------------------------------------------------
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
            sheet_name="Year 2010-2011",          # the sheet used in the notebook
            parse_dates=["InvoiceDate"],
        )

    raise FileNotFoundError(
        f"Neither {RAW_CSV} nor {RAW_XLSX} was found 🤷‍♂️\n"
        "Please place the raw file in one of those locations."
    )

# ------------------------------------------------------------------------------------------
# 2) BASIC CLEANING  (identical to **01_eda.ipynb**)
# ------------------------------------------------------------------------------------------
def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning in line with the exploration notebook.
    * drop negative quantities (refunds/cancellations),
    * drop rows lacking CustomerID,
    * cast CustomerID to string (safer for one-hotting later).
    """
    # normalise the column name first (Excel → “Customer ID”, CSV → “CustomerID”)
    df = df.rename(columns={"Customer ID": "CustomerID"})

    df = df[df["Quantity"] > 0].copy()
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int).astype("string")
    return df

# ------------------------------------------------------------------------------------------
# 3) FULL MODELLING DATA (calls your feature-engineering routine)
# ------------------------------------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    """
    Load, clean, feature-engineer and return the final 57-column table
    exactly as you created in *02_features.ipynb*.

    Relies on `ubp.features.build_feature_table`.
    """
    # ⬇️ heavy deps (xgboost, etc.) stay *inside* this function scope
    from ubp.features import build_feature_table

    raw  = load_raw()
    df   = clean_raw(raw)
    dataset = build_feature_table(df)
    return dataset

# ------------------------------------------------------------------------------------------
# 4) CONVENIENT CLI DEBUG (optional)
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    raw = load_raw()
    print("RAW cols:", raw.columns.tolist())
    dataset = load_dataset()
    print("✅ Final dataset shape:", dataset.shape)
    print(dataset.head())
