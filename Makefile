.PHONY: install fetch train api test lint clean

install:
	pip install -e ".[dev]"

fetch:
	python -m forecasting.data

train:
	python -m forecasting.train

api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=forecasting --cov-report=term-missing

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache .coverage htmlcov; \
	echo "cleaned"
