# Personality & Salary: a data story

This Streamlit project compares Big Five personality traits with wages in the real Swiss TREE study and with income in a separate synthetic Kaggle dataset. The datasets remain separate throughout the analysis. TREE focuses on entry-level wages; the synthetic data demonstrates how personality coefficients can change when background, education, and performance controls are added.

## Folder layout

- `app.py` and `pages/`: Streamlit prototype
- `site/`: Quarto article source and static data builder
- `docs/`: rendered GitHub Pages site
- `src/`: shared data loaders and analysis scripts
- `dataset/TreeData/`: canonical TREE source CSV
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

## Launch Streamlit

```bash
make app
```

## Run sanity tests

```bash
make test
```

## Explore the article site

The Quarto version is a scrollable article with interactive Plotly charts. Build its data bundle and local preview with:

```bash
python site/build_data.py
```

Open `docs/index.html` in a browser to preview it. To render through Quarto, install the Quarto CLI and run:

```bash
quarto render site
```

To publish with GitHub Pages, push the repository and select **Settings → Pages → Deploy from a branch → `main` → `/docs`**. The static page uses precomputed regression results; it does not run Python in visitors' browsers.
