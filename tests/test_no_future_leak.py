import numpy as np
import pandas as pd
import pytest

from fraud.features.offline import build_uid_features

HOUR = 3_600
CFG = {"uid_windows": {"1h": "1h", "24h": "24h"}, "small_amount": 10.0, "median_window": "30D"}


def _txns(rows: list[tuple]) -> pd.DataFrame:
    """rows = (seconds, card1, amount). D1 is set so that uid depends on card1 only."""
    df = pd.DataFrame(rows, columns=["TransactionDT", "card1", "TransactionAmt"])
    df["addr1"] = 100.0
    df["D1"] = (df["TransactionDT"] // 86_400).astype("float32")  # open day = 0 for every row
    return df


def test_first_transaction_of_a_card_has_no_history():
    f = build_uid_features(_txns([(0, 1, 50.0)]), CFG)
    assert f.loc[0, "uid_cnt_1h"] == 0
    assert f.loc[0, "uid_sum_24h"] == 0
    assert np.isnan(f.loc[0, "secs_since_prev"])
    assert np.isnan(f.loc[0, "amt_to_uid_median"])


def test_window_excludes_current_and_includes_left_edge():
    # rows at 0 s, exactly 1 h, 1 h + 1 s
    f = build_uid_features(_txns([(0, 1, 10.0), (HOUR, 1, 20.0), (HOUR + 1, 1, 30.0)]), CFG)
    assert f["uid_cnt_1h"].tolist() == [0, 1, 1]  # [t-1h, t): 0 s is in for row 2, out for row 3
    assert f["uid_sum_1h"].tolist() == [0, 10, 20]
    assert f["secs_since_prev"].tolist()[1:] == [HOUR, 1]


def test_cards_do_not_mix():
    f = build_uid_features(_txns([(0, 1, 5.0), (10, 2, 7.0), (20, 1, 9.0)]), CFG)
    assert f["uid_cnt_1h"].tolist() == [0, 0, 1]
    assert f.loc[2, "uid_small_cnt_24h"] == 1  # 5.0 < 10.0


def test_unsorted_input_is_rejected():
    with pytest.raises(ValueError):
        build_uid_features(_txns([(10, 1, 5.0), (0, 1, 5.0)]), CFG)


@pytest.mark.parametrize("cut", [1, 137, 500, 1_999])
def test_no_future_leak(cut):
    """Features of a transaction must not change when all later rows are removed."""
    rng = np.random.default_rng(42)
    n = 2_000
    rows = list(
        zip(
            np.sort(rng.integers(0, 10 * 86_400, size=n)),
            rng.integers(1, 50, size=n),
            rng.exponential(80, size=n).round(2),
            strict=True,
        )
    )
    df = _txns(rows)
    full = build_uid_features(df, CFG)
    past_only = build_uid_features(df.iloc[:cut], CFG)
    pd.testing.assert_frame_equal(full.iloc[:cut], past_only)
