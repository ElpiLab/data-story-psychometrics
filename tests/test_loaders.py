from src.load_data import load_kaggle, load_tree


def test_tree_loader_returns_rows_and_log_wage():
    data = load_tree()
    assert not data.empty
    assert "log_monthly_wage" in data.columns


def test_kaggle_loader_returns_rows_and_log_income():
    data = load_kaggle()
    assert not data.empty
    assert "log_income" in data.columns
