.PHONY: install lint test fmt

install:
	pip install -e .

lint:
	ruff check .
	black --check .

fmt:
	black .
	ruff check . --fix

test:
	pytest -q
