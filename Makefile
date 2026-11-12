.PHONY: install lint format test data train up seed eval

install:  ## окружение + git-хуки
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

data:     ## скачать IEEE-CIS и собрать data/raw.parquet
	uv run python scripts/download_data.py

train:    ## обучить, откалибровать, зарегистрировать в MLflow
	@echo "TODO: phase 1"

up:       ## docker compose: api + redis + mlflow
	@echo "TODO: docker compose up"

seed:     ## загрузить историю транзакций в Redis
	@echo "TODO: seed redis"

eval:     ## метрики модели и копилота
	@echo "TODO: eval"
