from airbnb_oslo.helpers.sklearn_helpers import get_column_transformer


def test_imports():
    column_transformer = get_column_transformer()
    assert column_transformer.__class__.__name__ == "ColumnTransformer"
