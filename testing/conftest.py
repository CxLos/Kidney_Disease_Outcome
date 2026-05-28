import sys
import os

# Add project root to sys.path so 'app' package is importable from all test files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest


# ------------------------------------------------------------------
# Shared synthetic DataFrame — 100 rows, balanced classes
# ------------------------------------------------------------------
@pytest.fixture(scope='session')
def synthetic_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Age':              np.random.randint(20, 80, n).astype(float),
        'Creatinine_Level': np.random.uniform(0.5, 10.0, n),
        'BUN':              np.random.uniform(5, 100, n),
        'Diabetes':         np.random.randint(0, 2, n).astype(float),
        'Hypertension':     np.random.randint(0, 2, n).astype(float),
        'GFR':              np.random.uniform(15, 120, n),
        'Urine_Output':     np.random.uniform(200, 2500, n),
        'CKD_Status':       ['1'] * 50 + ['0'] * 50,
        'Dialysis_Needed':  np.random.randint(0, 2, n),
    })


@pytest.fixture(scope='session')
def preprocessed(synthetic_df):
    from app.data.loader import preprocess_data
    # Returns: X_scaled, y, X_train, X_test, y_train, y_test
    return preprocess_data(synthetic_df)


@pytest.fixture(scope='session')
def trained_models(preprocessed):
    from app.models.train import train_models
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    return train_models(X_train, y_train)


@pytest.fixture(scope='session')
def evaluated_metrics(trained_models, preprocessed):
    from app.models.train import evaluate_models
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    return evaluate_models(trained_models, X_test, y_test)


@pytest.fixture(scope='session')
def rf_artifacts(trained_models, preprocessed):
    from app.models.train import get_rf_artifacts
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    return get_rf_artifacts(trained_models['Random Forest'], X_scaled, y, X_test, y_test)
