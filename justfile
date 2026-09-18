set dotenv-load := true

sync:
    uv sync

lint:
    uv run ruff check .
    uv run ruff format --check .

typecheck:
    uv run mypy src/

test:
    uv run pytest

check: lint typecheck test

notebook:
    uv run jupyter lab notebooks/

ingest-footballdata:
    uv run plforecast ingest football-data

ingest-fpl:
    uv run plforecast ingest fpl

ingest-understat:
    uv run plforecast ingest understat

ingest-clubelo:
    uv run plforecast ingest clubelo

migrate:
    uv run plforecast db-migrate

bootstrap: ingest-footballdata ingest-fpl ingest-understat ingest-clubelo curate

curate:
    uv run plforecast curate

evaluate:
    uv run plforecast evaluate

render-evaluation:
    uv run plforecast render-evaluation

tune model="dixon-coles" parameter="xi":
    uv run plforecast tune --model {{model}} --parameter {{parameter}}

forecast:
    uv run plforecast forecast

validate-artifacts:
    uv run plforecast validate-artifacts

validate:
    uv run plforecast validate

prune-raw keep="4":
    uv run plforecast prune-raw --keep {{keep}}
