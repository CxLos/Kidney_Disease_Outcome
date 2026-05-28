import pytest
from app.config import FEATURES, TARGET, COLUMNS


# ========================== FEATURES ========================== #

def test_features_is_list():
    assert isinstance(FEATURES, list)


def test_features_not_empty():
    assert len(FEATURES) > 0


def test_features_all_strings():
    assert all(isinstance(f, str) for f in FEATURES)


def test_features_contains_age():
    assert 'Age' in FEATURES


def test_features_contains_creatinine():
    assert 'Creatinine_Level' in FEATURES


# ========================== TARGET ========================== #

def test_target_is_string():
    assert isinstance(TARGET, str)


def test_target_value():
    assert TARGET == 'CKD_Status'


def test_target_in_columns():
    assert TARGET in COLUMNS


# ========================== COLUMNS ========================== #

def test_columns_is_list():
    assert isinstance(COLUMNS, list)


def test_columns_not_empty():
    assert len(COLUMNS) > 0


def test_columns_all_strings():
    assert all(isinstance(c, str) for c in COLUMNS)


def test_columns_contains_target():
    assert 'CKD_Status' in COLUMNS


def test_columns_contains_gfr():
    assert 'GFR' in COLUMNS


def test_columns_count():
    assert len(COLUMNS) == 9
