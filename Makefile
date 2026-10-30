.PHONY: install analyze app test site-preview

install:
	python -m pip install -r requirements.txt

analyze:
	python -m src.tree_analysis
	python -m src.kaggle_analysis
	python -m src.comparison_analysis
	python -m src.advanced_analysis
	python -m src.cleaning_analysis
	python -m src.hourlywage_mean_median

app:
	python -m streamlit run app.py

test:
	python -m pytest

site-preview:
	@echo Open docs/index.html in a browser for the local preview.
