"""Print TREE wage summaries from the canonical data loader."""
import pandas as pd

from src.load_data import load_tree


def main():
    df = load_tree()
    wage_column = "monthly_wage" if "monthly_wage" in df.columns else "wage_hourly"
    wage = pd.to_numeric(df[wage_column], errors="coerce")
    print(f"TREE wage summary ({wage_column}): n before dropna={len(df)}; n after dropna={wage.notna().sum()}")
    print(wage.describe())
    print(f"Mean: {wage.mean():.2f}; median: {wage.median():.2f}")


if __name__ == "__main__":
    main()
