default:
    @just --list

install:
    uv sync

run config="models.yaml" tasks="tasks.jsonl" *args="":
    uv run llm-eval run --config {{config}} --tasks {{tasks}} {{args}}

report run_dir:
    uv run llm-eval report {{run_dir}}

validate config="models.yaml" tasks="tasks.jsonl":
    uv run llm-eval validate --config {{config}} --tasks {{tasks}}

test *args="":
    uv run pytest {{args}}

smoke:
    uv run pytest -m smoke

lint:
    uv run ruff check .
    uv run ruff format --check .

format:
    uv run ruff check --fix .
    uv run ruff format .

typecheck:
    uv run pyright

ci: lint typecheck test
