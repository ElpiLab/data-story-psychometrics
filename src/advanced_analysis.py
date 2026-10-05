"""Additional TREE regression diagnostics, using the canonical loader."""
from src.analysis_utils import fit_tree
from src.load_data import TREE_TRAITS, load_tree


def main():
    df = load_tree()
    model, _ = fit_tree(df)
    print("Robust TREE regression (HC3):")
    print(model.summary())
    print("Trait coefficients:")
    print(model.params[list(TREE_TRAITS)])


if __name__ == "__main__":
    main()
