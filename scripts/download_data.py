"""Download IEEE-CIS Fraud Detection from Kaggle and build data/raw.parquet.

Usage:  make data   (or: uv run python scripts/download_data.py)
Needs:  accepted competition rules + Kaggle API token in ~/.kaggle/
"""

import zipfile
from pathlib import Path

from fraud.data import build_raw

COMPETITION = "ieee-fraud-detection"
RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/raw.parquet")


def download(file_name: str) -> Path:
    """Download one competition file into data/raw/ (skip if already there)."""
    import kaggle  # imported here: authenticates on import

    csv_path = RAW_DIR / file_name
    if csv_path.exists():
        print(f"skip {file_name}: already downloaded")
        return csv_path

    kaggle.api.competition_download_file(COMPETITION, file_name, path=RAW_DIR)

    zip_path = RAW_DIR / f"{file_name}.zip"
    if zip_path.exists():  # large files come zipped
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()
    return csv_path


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tx_csv = download("train_transaction.csv")
    id_csv = download("train_identity.csv")

    print("building raw.parquet ...")
    df = build_raw(tx_csv, id_csv)
    df.to_parquet(OUT_PATH, index=False)

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    file_mb = OUT_PATH.stat().st_size / 1e6
    print(f"rows={len(df):,} cols={df.shape[1]} fraud_rate={df['isFraud'].mean():.4f}")
    print(f"in memory: {mem_mb:.0f} MB, on disk: {file_mb:.0f} MB -> {OUT_PATH}")


if __name__ == "__main__":
    main()
