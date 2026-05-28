# ============================ MODEL TRAINING =============================== #

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, jaccard_score, confusion_matrix, roc_curve, auc
)
from sklearn.model_selection import cross_val_score

from app import config as cfg


def train_models(X_train, y_train):
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    return {'Logistic Regression': log_reg, 'Decision Tree': dt, 'Random Forest': rf}


def evaluate_models(models, X_test, y_test):
    metrics_summary = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        metrics_summary.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, pos_label='1'),
            'Recall': recall_score(y_test, y_pred, pos_label='1'),
            'F1 Score': f1_score(y_test, y_pred, pos_label='1'),
            'Jaccard': jaccard_score(y_test, y_pred, pos_label='1'),
        })

    return pd.DataFrame(metrics_summary)


def get_rf_artifacts(rf, X_scaled, y, X_test, y_test):
    cv_scores = cross_val_score(RandomForestClassifier(), X_scaled, y, cv=5)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred, labels=['0', '1'])
    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label='1')
    roc_auc = auc(fpr, tpr)

    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': cfg.FEATURES,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return {
        'cm': cm,
        'y_proba': y_proba,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'feature_importance_df': feature_importance_df,
        'cv_scores': cv_scores,
    }
