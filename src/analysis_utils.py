"""Shared helpers for analyses and interactive regression pages."""
import statsmodels.formula.api as smf

from src.load_data import KAGGLE_TRAITS, TREE_TRAITS


def report_dropna(df, columns, label):
    before = len(df)
    result = df.dropna(subset=columns).copy()
    print(f"{label}: n before dropna={before}; n after dropna={len(result)}")
    return result


def regression_formula(outcome, traits, controls=()):
    terms = list(traits)
    terms.extend(f"C({column})" if column == "sex" else column for column in controls)
    return f"{outcome} ~ " + " + ".join(terms)


def fit_kaggle(data, controls=()):
    """Fit robust OLS with traits and the selected available controls."""
    selected = [column for column in controls if column in data.columns]
    columns = KAGGLE_TRAITS + ["log_income"] + selected
    clean = report_dropna(data, columns, "Kaggle regression")
    model = smf.ols(regression_formula("log_income", KAGGLE_TRAITS, selected), data=clean).fit(cov_type="HC3")
    return model, clean


def fit_tree(data, controls=()):
    selected = [column for column in controls if column in data.columns]
    columns = list(TREE_TRAITS) + ["log_monthly_wage"] + selected
    clean = report_dropna(data, columns, "TREE regression")
    model = smf.ols(regression_formula("log_monthly_wage", TREE_TRAITS, selected), data=clean).fit(cov_type="HC3")
    return model, clean
