import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from app.visuals.charts import (
    build_metrics_bar,
    build_ckd_bar,
    build_confusion_matrix,
    build_feature_importance,
    build_score_dist,
    build_roc_curve,
    build_data_table,
)

# ========================== Shared Fixtures ========================== #

@pytest.fixture(scope='module')
def sample_df():
    return pd.DataFrame({
        'Age':              [45, 60, 30, 70, 55, 40, 65],
        'Creatinine_Level': [1.2, 5.4, 0.9, 8.1, 2.3, 1.5, 6.0],
        'BUN':              [15,  45,  12,  80,  30,  20,  60],
        'Diabetes':         [0,   1,   0,   1,   0,   1,   0],
        'Hypertension':     [1,   1,   0,   1,   0,   0,   1],
        'GFR':              [90,  25,  110, 12,  60,  85,  18],
        'Urine_Output':     [1500,400, 2000,250, 900, 1200,350],
        'CKD_Status':       ['0','1', '0', '1', '0', '0', '1'],
        'Dialysis_Needed':  [0,   1,   0,   1,   0,   0,   1],
    })


@pytest.fixture(scope='module')
def sample_metrics():
    return pd.DataFrame({
        'Model':     ['Logistic Regression', 'Decision Tree', 'Random Forest'],
        'Accuracy':  [0.85, 0.88, 0.92],
        'Precision': [0.83, 0.87, 0.91],
        'Recall':    [0.80, 0.85, 0.90],
        'F1 Score':  [0.81, 0.86, 0.90],
        'Jaccard':   [0.68, 0.75, 0.82],
    })


@pytest.fixture(scope='module')
def sample_cm():
    return np.array([[50, 5], [3, 42]])


@pytest.fixture(scope='module')
def sample_fi_df():
    return pd.DataFrame({
        'Feature':    ['Creatinine_Level', 'Age', 'BUN'],
        'Importance': [0.40, 0.35, 0.25],
    })


@pytest.fixture(scope='module')
def sample_y_proba():
    return np.array([0.1, 0.25, 0.5, 0.72, 0.88, 0.33, 0.65, 0.92, 0.04, 0.61])


@pytest.fixture(scope='module')
def sample_roc():
    fpr = np.array([0.0, 0.05, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.7,  0.85, 0.95, 1.0])
    return fpr, tpr, 0.93


# ========================== build_metrics_bar ========================== #

def test_metrics_bar_returns_figure(sample_metrics):
    assert isinstance(build_metrics_bar(sample_metrics), go.Figure)


def test_metrics_bar_title(sample_metrics):
    fig = build_metrics_bar(sample_metrics)
    assert 'Model Performance Metrics' in fig.layout.title.text


def test_metrics_bar_orientation_horizontal(sample_metrics):
    fig = build_metrics_bar(sample_metrics)
    assert fig.data[0].orientation == 'h'


# ========================== build_ckd_bar ========================== #

def test_ckd_bar_returns_figure(sample_df):
    assert isinstance(build_ckd_bar(sample_df), go.Figure)


def test_ckd_bar_title(sample_df):
    fig = build_ckd_bar(sample_df)
    assert 'CKD Status Distribution' in fig.layout.title.text


def test_ckd_bar_has_traces(sample_df):
    fig = build_ckd_bar(sample_df)
    assert len(fig.data) > 0


# ========================== build_confusion_matrix ========================== #

def test_confusion_matrix_returns_figure(sample_cm):
    assert isinstance(build_confusion_matrix(sample_cm), go.Figure)


def test_confusion_matrix_title(sample_cm):
    fig = build_confusion_matrix(sample_cm)
    assert 'Confusion Matrix' in fig.layout.title.text


def test_confusion_matrix_height(sample_cm):
    fig = build_confusion_matrix(sample_cm)
    assert fig.layout.height == 600


# ========================== build_feature_importance ========================== #

def test_feature_importance_returns_figure(sample_fi_df):
    assert isinstance(build_feature_importance(sample_fi_df), go.Figure)


def test_feature_importance_title(sample_fi_df):
    fig = build_feature_importance(sample_fi_df)
    assert 'Feature Importance' in fig.layout.title.text


def test_feature_importance_no_legend(sample_fi_df):
    fig = build_feature_importance(sample_fi_df)
    assert fig.layout.showlegend is False


# ========================== build_score_dist ========================== #

def test_score_dist_returns_figure(sample_y_proba):
    assert isinstance(build_score_dist(sample_y_proba), go.Figure)


def test_score_dist_title(sample_y_proba):
    fig = build_score_dist(sample_y_proba)
    assert 'Probability Distribution' in fig.layout.title.text


def test_score_dist_has_histogram_trace(sample_y_proba):
    fig = build_score_dist(sample_y_proba)
    assert any(isinstance(t, go.Histogram) for t in fig.data)


# ========================== build_roc_curve ========================== #

def test_roc_curve_returns_figure(sample_roc):
    fpr, tpr, roc_auc = sample_roc
    assert isinstance(build_roc_curve(fpr, tpr, roc_auc), go.Figure)


def test_roc_curve_title(sample_roc):
    fpr, tpr, roc_auc = sample_roc
    fig = build_roc_curve(fpr, tpr, roc_auc)
    assert 'ROC Curve' in fig.layout.title.text


def test_roc_curve_has_two_traces(sample_roc):
    fpr, tpr, roc_auc = sample_roc
    fig = build_roc_curve(fpr, tpr, roc_auc)
    assert len(fig.data) == 2


def test_roc_curve_auc_label(sample_roc):
    fpr, tpr, roc_auc = sample_roc
    fig = build_roc_curve(fpr, tpr, roc_auc)
    assert 'AUC' in fig.data[0].name


# ========================== build_data_table ========================== #

def test_data_table_returns_figure(sample_df):
    assert isinstance(build_data_table(sample_df), go.Figure)


def test_data_table_has_table_trace(sample_df):
    fig = build_data_table(sample_df)
    assert any(isinstance(t, go.Table) for t in fig.data)


def test_data_table_header_values(sample_df):
    fig = build_data_table(sample_df)
    table_trace = next(t for t in fig.data if isinstance(t, go.Table))
    assert list(table_trace.header.values) == list(sample_df.columns)
