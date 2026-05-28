import numpy as np
import pandas as pd
import pytest

from app.models.train import train_models, evaluate_models, get_rf_artifacts


# ========================== train_models ========================== #

def test_train_models_returns_dict(preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    models = train_models(X_train, y_train)
    assert isinstance(models, dict)


def test_train_models_has_three_entries(preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    models = train_models(X_train, y_train)
    assert len(models) == 3


def test_train_models_keys(preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    models = train_models(X_train, y_train)
    assert set(models.keys()) == {'Logistic Regression', 'Decision Tree', 'Random Forest'}


def test_train_models_all_have_predict(preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    models = train_models(X_train, y_train)
    for model in models.values():
        assert hasattr(model, 'predict')


# ========================== evaluate_models ========================== #

def test_evaluate_models_returns_dataframe(trained_models, preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    df_metrics = evaluate_models(trained_models, X_test, y_test)
    assert isinstance(df_metrics, pd.DataFrame)


def test_evaluate_models_row_count(trained_models, preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    df_metrics = evaluate_models(trained_models, X_test, y_test)
    assert len(df_metrics) == 3


def test_evaluate_models_column_names(trained_models, preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    df_metrics = evaluate_models(trained_models, X_test, y_test)
    expected = {'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard'}
    assert expected.issubset(df_metrics.columns)


def test_evaluate_models_accuracy_in_range(trained_models, preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    df_metrics = evaluate_models(trained_models, X_test, y_test)
    assert df_metrics['Accuracy'].between(0, 1).all()


def test_evaluate_models_all_scores_in_range(trained_models, preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    df_metrics = evaluate_models(trained_models, X_test, y_test)
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Jaccard']:
        assert df_metrics[col].between(0, 1).all(), f"{col} out of [0, 1]"


def test_evaluate_models_model_names(trained_models, preprocessed):
    X_scaled, y, X_train, X_test, y_train, y_test = preprocessed
    df_metrics = evaluate_models(trained_models, X_test, y_test)
    assert set(df_metrics['Model']) == {'Logistic Regression', 'Decision Tree', 'Random Forest'}


# ========================== get_rf_artifacts ========================== #

def test_get_rf_artifacts_returns_dict(rf_artifacts):
    assert isinstance(rf_artifacts, dict)


def test_get_rf_artifacts_has_all_keys(rf_artifacts):
    expected = {'cm', 'y_proba', 'fpr', 'tpr', 'roc_auc', 'feature_importance_df', 'cv_scores'}
    assert expected == set(rf_artifacts.keys())


def test_get_rf_artifacts_cm_shape(rf_artifacts):
    assert rf_artifacts['cm'].shape == (2, 2)


def test_get_rf_artifacts_roc_auc_in_range(rf_artifacts):
    assert 0.0 <= rf_artifacts['roc_auc'] <= 1.0


def test_get_rf_artifacts_y_proba_in_range(rf_artifacts):
    y_proba = rf_artifacts['y_proba']
    assert ((y_proba >= 0) & (y_proba <= 1)).all()


def test_get_rf_artifacts_fpr_tpr_lengths_match(rf_artifacts):
    assert len(rf_artifacts['fpr']) == len(rf_artifacts['tpr'])


def test_get_rf_artifacts_feature_importance_df(rf_artifacts):
    fi_df = rf_artifacts['feature_importance_df']
    assert isinstance(fi_df, pd.DataFrame)
    assert 'Feature' in fi_df.columns
    assert 'Importance' in fi_df.columns


def test_get_rf_artifacts_feature_importance_sorted(rf_artifacts):
    fi_df = rf_artifacts['feature_importance_df']
    importances = fi_df['Importance'].tolist()
    assert importances == sorted(importances, reverse=True)


def test_get_rf_artifacts_cv_scores_length(rf_artifacts):
    # cross_val_score with cv=5 returns 5 scores
    assert len(rf_artifacts['cv_scores']) == 5


def test_get_rf_artifacts_cv_scores_in_range(rf_artifacts):
    assert all(0 <= s <= 1 for s in rf_artifacts['cv_scores'])
