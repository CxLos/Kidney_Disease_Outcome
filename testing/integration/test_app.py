import pytest
import dash
import flask
from unittest.mock import patch


# ---------------------------------------------------------------
# app.kidney_disease module-level code (Dash app + layout) is
# executed on first import and cached by Python for the rest of
# the session.  We patch load_data so the pipeline runs on the
# fast synthetic_df instead of the real Excel file.
# ---------------------------------------------------------------

@pytest.fixture(scope='module', autouse=True)
def _patch_load_data(synthetic_df):
    """Ensure the Dash app is initialised with synthetic data."""
    import sys
    # Remove cached module so the patch is in effect on first import
    for mod in list(sys.modules.keys()):
        if mod.startswith('app.kidney_disease'):
            del sys.modules[mod]

    with patch('app.layouts.layout.load_data', return_value=synthetic_df):
        import app.kidney_disease  # noqa: F401 — triggers module-level code
    yield


# ========================== app object ========================== #

def test_app_is_dash_instance():
    from app.kidney_disease import app
    assert isinstance(app, dash.Dash)


def test_server_is_flask_instance():
    from app.kidney_disease import server
    assert isinstance(server, flask.Flask)


def test_app_has_layout():
    from app.kidney_disease import app
    assert app.layout is not None


def test_server_equals_app_server():
    from app.kidney_disease import app, server
    assert server is app.server
