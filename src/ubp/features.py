from __future__ import annotations
import pandas as pd

def _first_purchase_metrics(df: pd.DataFrame) -> pd.DataFrame:
    first_tx = (
        df.sort_values("InvoiceDate")
          .groupby("CustomerID", as_index=False)
          .first()
    )
    first_tx["first_quantity"] = first_tx["Quantity"]
    first_tx["first_revenue"]  = first_tx["Quantity"] * first_tx["Price"]
    return first_tx[["CustomerID", "first_quantity", "first_revenue"]]


def _country_dummies(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.get_dummies(
            df.groupby("CustomerID")["Country"].first(),
            prefix="country"
        )
        .reset_index()
    )


def _diversity_first_day(df: pd.DataFrame,
                         user_agg: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.merge(user_agg[["CustomerID", "first_date"]],
                      on="CustomerID", how="left")
    mask = df_tmp["InvoiceDate"].dt.normalize() == \
           df_tmp["first_date"].dt.normalize()
    return (
        df_tmp[mask]
          .groupby("CustomerID")["StockCode"].nunique()
          .rename("diversity_first_day")
          .reset_index()
    )


def _recency_to_second_purchase(df: pd.DataFrame) -> pd.DataFrame:
    tx_ranked = (
        df.sort_values("InvoiceDate")
          .groupby("CustomerID", as_index=False)
          .nth([0, 1])                       # first & second tx
          .reset_index(drop=True)
    )
    tx_ranked["rank"] = tx_ranked.groupby("CustomerID").cumcount()
    first_tx  = tx_ranked[tx_ranked["rank"] == 0]
    second_tx = tx_ranked[tx_ranked["rank"] == 1]

    rec = (
        first_tx[["CustomerID", "InvoiceDate"]]
          .rename(columns={"InvoiceDate": "first_date"})
          .merge(
              second_tx[["CustomerID", "InvoiceDate"]]
                .rename(columns={"InvoiceDate": "second_date"}),
              on="CustomerID", how="left"
          )
    )
    rec["recency_2nd"] = (
        rec["second_date"] - rec["first_date"]
    ).dt.days.fillna(999).astype(int)
    return rec[["CustomerID", "recency_2nd"]]


def _rfm_3day(df: pd.DataFrame,
              user_agg: pd.DataFrame,
              window: int = 3) -> pd.DataFrame:
    df_tmp = df.merge(user_agg[["CustomerID", "first_date"]],
                      on="CustomerID")
    df_tmp["days_since_first"] = (
        df_tmp["InvoiceDate"] - df_tmp["first_date"]
    ).dt.days
    mask = df_tmp["days_since_first"] <= window
    return (
        df_tmp[mask]
          .assign(revenue=lambda d: d["Quantity"] * d["Price"])
          .groupby("CustomerID", as_index=False)
          .agg(freq_3d=("Invoice", "nunique"),
               monetary_3d=("revenue", "sum"))
    )


def _time_of_first_purchase(first_tx: pd.DataFrame) -> pd.DataFrame:
    t = first_tx[["CustomerID", "InvoiceDate"]].rename(
        columns={"InvoiceDate": "first_date_time"}
    )
    t["first_hour"] = t["first_date_time"].dt.hour
    t["first_dow"]  = t["first_date_time"].dt.weekday
    return t[["CustomerID", "first_hour", "first_dow"]]


def _diversity_7days(df: pd.DataFrame,
                     user_agg: pd.DataFrame) -> pd.DataFrame:
    df_tmp = df.merge(user_agg[["CustomerID", "first_date"]],
                      on="CustomerID")
    df_tmp["days_since_first"] = (
        df_tmp["InvoiceDate"] - df_tmp["first_date"]
    ).dt.days
    mask = df_tmp["days_since_first"] <= 7
    return (
        df_tmp[mask]
          .groupby("CustomerID", as_index=False)["StockCode"].nunique()
          .rename(columns={"StockCode": "diversity_7d"})
    )

def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    user_agg = (
        df.groupby("CustomerID", as_index=False)
          .agg(
              total_orders   = ("Invoice",   "nunique"),
              total_quantity = ("Quantity",  "sum"),
              first_date     = ("InvoiceDate", "min"),
              last_date      = ("InvoiceDate", "max"),
          )
    )

    first_tx           = (
        df.sort_values("InvoiceDate")
          .groupby("CustomerID", as_index=False)
          .first()
    )

    feat_first_metrics = _first_purchase_metrics(df)
    feat_countries     = _country_dummies(df)
    feat_diversity_fd  = _diversity_first_day(df, user_agg)
    feat_recency       = _recency_to_second_purchase(df)
    feat_rfm_3d        = _rfm_3day(df, user_agg)
    feat_time          = _time_of_first_purchase(first_tx)
    feat_diversity_7d  = _diversity_7days(df, user_agg)

    feature_df = (
        user_agg[["CustomerID", "total_orders", "total_quantity",
                  "first_date", "last_date"]]               # keep dates
          .merge(feat_first_metrics, on="CustomerID", how="left")
          .merge(feat_diversity_fd,  on="CustomerID", how="left")
          .merge(feat_countries,     on="CustomerID", how="left")
          .merge(feat_recency,       on="CustomerID", how="left")
          .merge(feat_rfm_3d,        on="CustomerID", how="left")
          .merge(feat_time,          on="CustomerID", how="left")
          .merge(feat_diversity_7d,  on="CustomerID", how="left")
          .fillna(0)
    )

    feature_df["did_repurchase_7d"] = df.groupby("CustomerID")["InvoiceDate"]\
                                         .transform(lambda s: (s.diff().dt.days <= 7).any())\
                                         .astype(int)

    feature_df["days_to_repurchase"] = (
        df.groupby("CustomerID")["InvoiceDate"]
          .apply(lambda s: (s.sort_values().diff().dt.days.min()))
          .fillna(999)
          .astype(int)
          .values
    )

    cols = ["CustomerID", "did_repurchase_7d",
            "first_date", "last_date", "days_to_repurchase"] + \
           [c for c in feature_df.columns
            if c not in ("CustomerID", "did_repurchase_7d",
                         "first_date", "last_date", "days_to_repurchase")]
    dataset = feature_df[cols]

    return dataset

if __name__ == "__main__":
    # example usage:
    from ubp.data import load_raw, clean_raw
    raw = load_raw()
    df_clean = clean_raw(raw)
    ds = build_feature_table(df_clean)
    print("Table shape:", ds.shape)
