.PHONY: install dev viz test docker-build docker-run clean

install:
	pip install -e .

viz:
	pip install -e ".[viz]"

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

docker-build:
	docker build -t fairopt-causal .

docker-run:
	docker run --rm -v $(PWD)/results:/app/results fairopt-causal

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf results/*
