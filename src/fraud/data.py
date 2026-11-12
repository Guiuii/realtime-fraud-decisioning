"""IEEE-CIS data: loading, memory reduction, (later) uid and time split."""

from pathlib import Path

import pandas as pd

N_TRANSACTIONS = 590_540  # rows in train_transaction.csv


def _floats_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def read_csv_compact(path: Path, chunksize: int = 100_000) -> pd.DataFrame:
    """Read a large CSV in chunks, casting float64 -> float32 on the fly."""
    chunks = (_floats_to_float32(c) for c in pd.read_csv(path, chunksize=chunksize))
    return pd.concat(chunks, ignore_index=True)


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast ints, floats -> float32, text -> category."""
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = s.astype("float32")
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            df[col] = s.astype("category")
    return df


def build_raw(transaction_csv: Path, identity_csv: Path) -> pd.DataFrame:
    """Left-join transactions with identity, sort by time, shrink dtypes."""
    tx = read_csv_compact(transaction_csv)
    identity = pd.read_csv(identity_csv)

    df = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    if len(df) != N_TRANSACTIONS:
        raise ValueError(f"expected {N_TRANSACTIONS} rows after join, got {len(df)}")

    df = df.sort_values("TransactionDT", kind="stable", ignore_index=True)
    return reduce_memory(df)
