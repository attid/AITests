# llm-eval

Бенчмарк LLM как «мозга агента» через любой OpenAI-совместимый API. На вход — `tasks.jsonl` и `models.yaml`, на выход — таблица «качество × цена × латентность».

```text
Tasks → all enabled models → score → JSONL stream → CSV + leaderboard + markdown
```

## Что считаем

7 типов задач:
- **`exact`** / **`contains`** / **`regex`** — текстовое сравнение (с word-boundary для русского, опц. case-sensitive)
- **`numeric`** — извлечение числа + tolerance (abs / rel)
- **`json_exact`** — рекурсивное сравнение JSON c учётом числовой погрешности
- **`json_schema`** — валидация через `jsonschema` Draft 2020-12, опц. проверка значений
- **`llm_judge`** — отдельная модель оценивает по rubric (snap → {0, 0.5, 1})

Универсально на каждую задачу: strip `<think>...</think>`, проверка `forbidden_contains` со word-boundary.

## Quick start

```bash
# 1. Установка
just install              # uv sync

# 2. Ключи
nano .env                 # MINIMAX_API_KEY=..., OPENAI_API_KEY=..., и т.д.

# 3. Проверка конфига
just validate             # OK: 25 tasks, 14 models

# 4. Проверка ключей и связи (1 запрос на модель)
just ping                 # ✓/✗ по каждой включённой модели

# 5. Полный прогон + отчёт
just run                  # пишет в runs/<timestamp>/results.jsonl
just report runs/<timestamp>
```

## CLI

| Команда | Что делает |
|---|---|
| `llm-eval validate` | Проверяет `models.yaml` + `tasks.jsonl`, ищет дубли task_id |
| `llm-eval ping` | Отправляет 1 минимальный запрос на каждую включённую модель |
| `llm-eval run` | Прогоняет все задачи × все включённые модели × `repeats` |
| `llm-eval report` | Из `results.jsonl` делает CSV + markdown отчёт |

Флаги `run`/`ping`:
- `--only id1,id2` — только эти модели (override `enabled`)
- `--skip id3` — исключить модель
- `--config path/to/models.yaml`, `--tasks path/to/tasks.jsonl`, `--out runs/...`, `--resume`

## `models.yaml` — основные поля

```yaml
defaults:
  concurrency: 3        # параллельных запросов на модель (если не задано на самой модели)
  temperature: 0.0
  repeats: 1            # сколько раз гонять каждую задачу

judge:
  model_id: zai-glm     # ID модели-судьи для llm_judge задач

reporting:
  filter:
    avg_score_min: 0.75
    json_ok_rate_min: 0.90
    forbidden_fail_max: 2

models:
  - id: kimi-k2                        # человеко-читаемый ID
    enabled: true
    base_url: https://api.kimi.com/coding/v1
    api_key_env: KIMI_API_KEY          # имя env var, не сам ключ
    model: kimi-k2-0905-preview        # что отправлять в API
    pricing:                           # опц. — иначе cost_source=unknown
      input_per_million: 0.60
      output_per_million: 2.50
    use_provider_cost: true            # опц. — брать usage.cost из ответа (OpenRouter)
    is_reasoning: true                 # опц. — сохранить reasoning_content в JSONL
    response_format: json_object       # опц. — нативный JSON mode
    concurrency: 3                     # опц. — переопределяет defaults
    extra_headers:                     # опц. — кастомные заголовки
      User-Agent: claude-code/1.0
```

## `tasks.jsonl` — формат

Одна задача на строку. Базовые поля общие: `id`, `type`, `prompt`, `max_tokens`, `tags`, `weight`, `forbidden_contains`, `word_boundary`, `case_sensitive`.

Примеры:
```jsonl
{"id":"math_001","type":"json_exact","prompt":"Ответь JSON {\"answer\": число}. 17*23?","expected":{"answer":391},"tags":["json_format"]}
{"id":"hallu_001","type":"contains","prompt":"...","expected_contains":["нет данных"],"forbidden_contains":["@","initech.com"],"tags":["anti_hallucination"]}
{"id":"summ_001","type":"llm_judge","prompt":"Сожми в одно предложение: '...'","rubric":"Должно: (1) упомянуть X; (2) ...","tags":["summarization"]}
```

## Что лежит в `runs/<timestamp>/`

| файл | формат | для чего |
|---|---|---|
| `results.jsonl` | строка на каждый `(model × task × repeat)` | source of truth, для resume |
| `models.yaml` | копия конфига на момент прогона | фильтр в отчёте |
| `results.csv` | flat-таблица | для Excel / pandas |
| `leaderboard.csv` | агрегат по моделям | сухие цифры |
| `report.md` | human-readable markdown | главный — в Slack / PR |

`runs/` в `.gitignore`. Каждый прогон — новый каталог с микросекундным таймстампом, ничего не перетирается.

## Resume после краша

```bash
just run --out runs/20260506_210105_123456 --resume
```
Читает уже записанные строки в `results.jsonl`, пропускает done-кортежи `(model_id, task_id, repeat)`, добивает остальное.

## Как добавить модель

1. Сгенерь API-ключ у провайдера, положи в `.env` как `FOO_API_KEY=...`
2. Добавь блок в `models.yaml` (как в примерах выше)
3. `just validate && just ping --only <new_id>` — убедиться что работает
4. `just run --only <new_id>` для тестового прогона на 1 модели

Если ключ у провайдера ограничен на User-Agent (например, Kimi For Coding), используй `extra_headers: {User-Agent: claude-code/1.0}`.

## Разработка

```bash
just test             # pytest
just lint             # ruff check + format check
just typecheck        # pyright strict для src/
just ci               # все три
just format           # ruff --fix + format
just smoke            # реальный API smoke (нужен OPENAI_API_KEY)
```

Стек: Python 3.12, uv, typer, pydantic v2, httpx, jsonschema, rich, aiofiles. Без openai SDK (httpx даёт нужный контроль над retries и provider-specific полями типа `usage.cost`).

## Out of scope (v1)

- tool-use / function calling
- multi-turn диалоги
- стриминг
- HTML / Streamlit dashboards
- автоматический diff между прогонами
- автоподтягивание цен из OpenRouter `/models`
