import pytest
from unittest.mock import patch
from dash import html

from app.layouts.layout import create_layout


# Patch load_data so tests run fast (synthetic_df from conftest, 100 rows)
# All downstream code (preprocess → train → chart builders) runs for real.


def test_create_layout_returns_html_div(synthetic_df):
    with patch('app.layouts.layout.load_data', return_value=synthetic_df):
        layout = create_layout()
    assert isinstance(layout, html.Div)


def test_create_layout_has_children(synthetic_df):
    with patch('app.layouts.layout.load_data', return_value=synthetic_df):
        layout = create_layout()
    assert layout.children is not None
    assert len(layout.children) > 0


def test_create_layout_first_child_is_header(synthetic_df):
    with patch('app.layouts.layout.load_data', return_value=synthetic_df):
        layout = create_layout()
    header = layout.children[0]
    assert isinstance(header, html.Div)
    assert header.className == 'divv'


def test_create_layout_contains_readme_section(synthetic_df):
    with patch('app.layouts.layout.load_data', return_value=synthetic_df):
        layout = create_layout()
    # Last child should be the readme section
    last_child = layout.children[-1]
    assert isinstance(last_child, html.Div)
    assert last_child.className == 'readme-section'


def test_create_layout_contains_data_table_row(synthetic_df):
    with patch('app.layouts.layout.load_data', return_value=synthetic_df):
        layout = create_layout()
    classnames = [c.className for c in layout.children if hasattr(c, 'className')]
    assert 'row0' in classnames


def test_create_layout_contains_graph_rows(synthetic_df):
    with patch('app.layouts.layout.load_data', return_value=synthetic_df):
        layout = create_layout()
    row1_count = sum(1 for c in layout.children if hasattr(c, 'className') and c.className == 'row1')
    assert row1_count >= 3
