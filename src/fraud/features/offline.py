"""Offline velocity features: computed only from each card's past transactions."""

from pathlib import Path

import pandas as pd
import yaml

from fraud.data import make_uid


def load_config(path: str | Path = "configs/features.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rolling_by(
    key: pd.Series, ts: pd.Series, values: pd.Series, window: str, how: str
) -> pd.Series:
    """Aggregate `values` per `key` over the window [t - window, t), aligned to input rows.

    closed="left" excludes the current transaction, so every value uses only the past.
    Rows with a missing key get NaN.
    """
    frame = pd.DataFrame({"key": key, "ts": ts, "x": values}).dropna(subset=["key"])
    frame = frame.sort_values(["key", "ts"], kind="stable")
    rolled = frame.groupby("key", sort=True).rolling(window, on="ts", closed="left")["x"]
    result = pd.Series(getattr(rolled, how)().to_numpy(), index=frame.index)
    return result.reindex(key.index)


def build_uid_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-card velocity features. `df` must be sorted by TransactionDT."""
    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("df must be sorted by TransactionDT")

    uid = make_uid(df)
    has_uid = uid.notna()
    ts = pd.to_datetime(df["TransactionDT"], unit="s")
    amt = df["TransactionAmt"].astype("float64")

    out = pd.DataFrame(index=df.index)
    for name, window in cfg["uid_windows"].items():
        cnt = _rolling_by(uid, ts, amt, window, "count")
        total = _rolling_by(uid, ts, amt, window, "sum")
        out[f"uid_cnt_{name}"] = cnt.fillna(0).where(has_uid)
        out[f"uid_sum_{name}"] = total.fillna(0).where(has_uid)

    small = (amt < cfg["small_amount"]).astype("float64")
    out["uid_small_cnt_24h"] = _rolling_by(uid, ts, small, "24h", "sum").fillna(0).where(has_uid)

    usual = _rolling_by(uid, ts, amt, cfg["median_window"], "median")
    out["amt_to_uid_median"] = amt / usual

    out["secs_since_prev"] = df["TransactionDT"].groupby(uid).diff()
    return out.astype("float32")
