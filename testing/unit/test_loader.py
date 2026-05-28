import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from app.data.loader import load_data, preprocess_data
from app import config as cfg


# ------------------------------------------------------------------
# Helper: raw df that pd.read_excel would return (integers for CKD_Status)
# ------------------------------------------------------------------
@pytest.fixture(scope='module')
def raw_excel_df():
    np.random.seed(7)
    n = 60
    return pd.DataFrame({
        'Age':              np.random.randint(20, 80, n).astype(float),
        'Creatinine_Level': np.random.uniform(0.5, 10.0, n),
        'BUN':              np.random.uniform(5, 100, n),
        'Diabetes':         np.random.randint(0, 2, n).astype(float),
        'Hypertension':     np.random.randint(0, 2, n).astype(float),
        'GFR':              np.random.uniform(15, 120, n),
        'Urine_Output':     np.random.uniform(200, 2500, n),
        'CKD_Status':       [1] * 30 + [0] * 30,  # integers, as in real Excel
        'Dialysis_Needed':  np.random.randint(0, 2, n),
    })


# ========================== load_data ========================== #

def test_load_data_returns_dataframe(raw_excel_df):
    with patch('app.data.loader.pd.read_excel', return_value=raw_excel_df):
        df = load_data()
    assert isinstance(df, pd.DataFrame)


def test_load_data_ckd_status_converted_to_str(raw_excel_df):
    with patch('app.data.loader.pd.read_excel', return_value=raw_excel_df):
        df = load_data()
    assert df['CKD_Status'].dtype == object


def test_load_data_ckd_status_values_are_strings(raw_excel_df):
    with patch('app.data.loader.pd.read_excel', return_value=raw_excel_df):
        df = load_data()
    assert all(isinstance(v, str) for v in df['CKD_Status'])


def test_load_data_row_count_preserved(raw_excel_df):
    with patch('app.data.loader.pd.read_excel', return_value=raw_excel_df):
        df = load_data()
    assert len(df) == len(raw_excel_df)


def test_load_data_columns_preserved(raw_excel_df):
    with patch('app.data.loader.pd.read_excel', return_value=raw_excel_df):
        df = load_data()
    assert list(df.columns) == list(raw_excel_df.columns)


# ========================== preprocess_data ========================== #

def test_preprocess_data_returns_six_items(synthetic_df):
    result = preprocess_data(synthetic_df)
    assert len(result) == 6


def test_preprocess_data_x_scaled_shape(synthetic_df):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocess_data(synthetic_df)
    assert X_scaled.shape == (len(synthetic_df), len(cfg.FEATURES))


def test_preprocess_data_y_length(synthetic_df):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocess_data(synthetic_df)
    assert len(y) == len(synthetic_df)


def test_preprocess_data_train_test_sum(synthetic_df):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocess_data(synthetic_df)
    assert len(X_train) + len(X_test) == len(synthetic_df)


def test_preprocess_data_test_size_approx_20_pct(synthetic_df):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocess_data(synthetic_df)
    ratio = len(X_test) / (len(X_train) + len(X_test))
    assert abs(ratio - 0.2) < 0.05


def test_preprocess_data_y_unique_values(synthetic_df):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocess_data(synthetic_df)
    assert set(y.unique()) == {'0', '1'}


def test_preprocess_data_x_scaled_approx_zero_mean(synthetic_df):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocess_data(synthetic_df)
    col_means = X_scaled.mean(axis=0)
    assert all(abs(m) < 1e-10 for m in col_means)
