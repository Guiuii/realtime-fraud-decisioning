import numpy as np
import pandas as pd

from fraud.data import SECONDS_PER_DAY, make_uid

DAY = SECONDS_PER_DAY


def _frame(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["TransactionDT", "card1", "addr1", "D1"])
    return df.astype({"addr1": "float32", "D1": "float32"})  # same dtypes as raw.parquet


def test_same_card_on_different_days_gets_same_uid():
    df = _frame([(10 * DAY, 13926, 315.0, 5.0), (12 * DAY + 3_600, 13926, 315.0, 7.0)])
    uid = make_uid(df)
    assert uid.iloc[0] == uid.iloc[1] == "13926_315_5"


def test_different_address_or_open_day_gives_different_uid():
    df = _frame(
        [
            (10 * DAY, 13926, 315.0, 5.0),
            (10 * DAY, 13926, 204.0, 5.0),  # other address
            (10 * DAY, 13926, 315.0, 6.0),
        ]
    )
    assert make_uid(df).nunique() == 3


def test_missing_component_gives_missing_uid():
    df = _frame(
        [
            (10 * DAY, 13926, np.nan, 5.0),
            (10 * DAY, 13926, 315.0, np.nan),
            (10 * DAY, 13926, 315.0, 5.0),
        ]
    )
    assert make_uid(df).isna().tolist() == [True, True, False]
