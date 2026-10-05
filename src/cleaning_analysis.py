"""Audit the canonical source datasets without modifying source files."""
from src.load_data import load_kaggle, load_tree


def main():
    for label, loader in (("TREE", load_tree), ("Kaggle synthetic", load_kaggle)):
        frame = loader()
        print(f"{label}: {len(frame)} rows, {len(frame.columns)} columns")
        print(frame.isna().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
