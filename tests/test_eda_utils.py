"""
Basic sanity tests for eda_utils and data_loader.
Uses a small synthetic dataframe so tests don't depend on the real
(DVC-tracked) dataset being present in the CI environment.
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eda_utils import overall_loss_ratio, loss_ratio_by_group, monthly_trend
from src.data_loader import add_risk_metrics, missing_value_report


def make_sample_df():
    return pd.DataFrame({
        "TransactionMonth": ["2014-02-01", "2014-03-01", "2014-03-01", "2014-04-01"],
        "Province": ["Gauteng", "Gauteng", "Western Cape", "Western Cape"],
        "Gender": ["Male", "Female", "Male", "Female"],
        "TotalPremium": [100.0, 200.0, 150.0, 0.0],
        "TotalClaims": [50.0, 0.0, 300.0, 0.0],
    })


def test_overall_loss_ratio():
    df = make_sample_df()
    ratio = overall_loss_ratio(df)
    expected = (50.0 + 0.0 + 300.0 + 0.0) / (100.0 + 200.0 + 150.0 + 0.0)
    assert abs(ratio - expected) < 1e-9


def test_loss_ratio_by_group():
    df = make_sample_df()
    summary = loss_ratio_by_group(df, "Province")
    assert "loss_ratio" in summary.columns
    assert set(summary.index) == {"Gauteng", "Western Cape"}


def test_monthly_trend_has_expected_columns():
    df = make_sample_df()
    monthly = monthly_trend(df)
    for col in ["claim_frequency", "loss_ratio", "policy_count"]:
        assert col in monthly.columns


def test_add_risk_metrics_computes_margin():
    df = make_sample_df()
    df = add_risk_metrics(df)
    assert "Margin" in df.columns
    assert df.loc[0, "Margin"] == 50.0  # 100 - 50


def test_missing_value_report_empty_when_no_missing():
    df = make_sample_df()
    report = missing_value_report(df)
    assert report.empty
