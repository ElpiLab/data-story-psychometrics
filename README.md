# Personality & Salary: a data story

This project is a static HTML data story comparing Big Five personality traits with wages in the real Swiss TREE study and with income in a separate synthetic Kaggle dataset. The datasets remain separate throughout the analysis. TREE focuses on entry-level hourly wages; the synthetic data demonstrates how personality coefficients can change when background, education, and performance controls are added.

## Folder layout

- `app.py` and `pages/`: Streamlit prototype
- `site/`: Quarto article source and static data builder
- `docs/`: rendered GitHub Pages site
- `src/`: shared data loaders and analysis scripts
- `dataset/TreeData/`: cleaned TREE2 analysis export supplied by the project team
- `dataset/Kaggle_BigFive_USA/`: canonical synthetic Kaggle CSV and dictionary
- `dataset/_archive/`: superseded source files retained for reference
- `output/`: generated figures and tables
- `tests/`: loader sanity tests

The Kaggle dataset is synthetic; its coefficients should not be interpreted as empirical effects for a real population.

### Analysis base don 5 traits VS Hourly Wage
Research question: Are Big Five traits associated with hourly wages?

Outcome: Log hourly wage.

Predictors: Five standardized personality scores.

Main finding: Agreeableness has a small positive association; the other traits do not.

### Vergleich mit Kaggle
FOlder Kaggle BigFiveUSA

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## Run analyses

```bash
make analyze
```

This runs the analysis scripts in `src/` and writes figures and tables to `output/`.

## Preview the static HTML story

```bash
make site-data
```

Open `docs/index.html` in a browser. This is the primary presentation path and does not require Quarto or a Python server.

## Run sanity tests

```bash
make test
```

The source template remains in `site/` for future Quarto use. To render through Quarto, install the Quarto CLI and run:

```bash
quarto render site
```

To publish with GitHub Pages, push the repository and select **Settings → Pages → Deploy from a branch → `main` → `/docs`**. The static page uses precomputed regression results; it does not run Python in visitors' browsers.
