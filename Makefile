.PHONY: install dev extract index serve test lint clean

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt ruff mypy pytest

extract:
	python winegpt/extract.py --country Espanya

index:
	python scripts/build_index.py --country Espanya

serve:
	streamlit run winegpt/app.py

test:
	pytest tests/ -v

lint:
	ruff check winegpt/ scripts/ tests/

clean:
	rm -rf data/ __pycache__/ .pytest_cache/
