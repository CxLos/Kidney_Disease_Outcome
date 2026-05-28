# ============================ DATA LOADER =============================== #

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app import config as cfg


def load_data():
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file = os.path.join(script_dir, 'data', 'kidney_disease_dataset.xlsx')
    df = pd.read_excel(file)
    df['CKD_Status'] = df['CKD_Status'].astype(str)
    return df


def preprocess_data(df):
    X = df[cfg.FEATURES]
    y = df[cfg.TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_scaled, y, X_train, X_test, y_train, y_test
