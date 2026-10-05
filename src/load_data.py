"""Canonical paths, dataset loaders, and shared analysis metadata."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TREE_PATH = DATA_DIR / "TreeData" / "Tree_Data_revision.csv"
KAGGLE_PATH = DATA_DIR / "Kaggle_BigFive_USA" / "big_five_life_outcomes.csv"

TREE_TRAITS = {
    "t0big5_e_comp": "Extraversion",
    "t0big5_a_comp": "Agreeableness",
    "t0big5_c_comp": "Conscientiousness",
    "t0big5_n_comp": "Neuroticism",
    "t0big5_o_comp": "Openness",
}
KAGGLE_TRAITS = [
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
]


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset not found at {path}")
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{label} dataset is missing required column(s): {', '.join(missing)}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )


def load_tree() -> pd.DataFrame:
    """Load TREE data and ensure monthly and log monthly wages are available."""
    df = _read_csv(TREE_PATH, "TREE")
    if "monthly_wage" not in df.columns and "monthly_salary" in df.columns:
        df["monthly_wage"] = pd.to_numeric(df["monthly_salary"], errors="coerce")
    if "monthly_wage" not in df.columns:
        if "wage_hourly" in df.columns:
            warnings.warn(
                "TREE data has no monthly_wage; falling back to wage_hourly as requested.",
                RuntimeWarning,
                stacklevel=2,
            )
            df["monthly_wage"] = pd.to_numeric(df["wage_hourly"], errors="coerce")
        else:
            _require_columns(df, ["monthly_wage"], "TREE")
    if "log_monthly_wage" not in df.columns:
        _require_columns(df, ["monthly_wage"], "TREE")
        df["log_monthly_wage"] = np.log(pd.to_numeric(df["monthly_wage"], errors="coerce").clip(lower=1))
    _require_columns(df, list(TREE_TRAITS) + ["log_monthly_wage"], "TREE")
    return df


def load_kaggle() -> pd.DataFrame:
    """Load the synthetic Kaggle dataset and ensure log income is available."""
    df = _read_csv(KAGGLE_PATH, "Kaggle")
    if "log_income" not in df.columns:
        _require_columns(df, ["income_usd"], "Kaggle")
        df["log_income"] = np.log(pd.to_numeric(df["income_usd"], errors="coerce").clip(lower=1))
    _require_columns(df, KAGGLE_TRAITS + ["log_income"], "Kaggle")
    return df
