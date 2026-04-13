# langgraph-stage-play-lab

Minimal educational LangGraph demo for a multi-agent stage play.

## What this project shows

- A state-driven graph (not a fixed `A -> B -> writer` chain)
- Dynamic routing by `manager` using current state
- Turn-based character interaction (`character_a`, `character_b`)
- Final drafting and audit pass

## State shape

The graph state keeps:

- `premise`
- `scene_notes`
- `dialogue_history`
- `draft`
- `audit`
- `next_node`

## Graph flow (dynamic)

```text
manager --(next_node)--> director | character_a | character_b | writer | auditor | END
director   -> manager
character_a-> manager
character_b-> manager
writer     -> manager
auditor    -> manager
```

`manager` inspects state and decides the next node:

- no `scene_notes` yet -> `director`
- dialogue turns not enough -> alternate `character_a` / `character_b`
- dialogue done, no `draft` -> `writer`
- no `audit` -> `auditor`
- otherwise -> `END`

## Project structure

```text
langgraph-stage-play-lab/
├─ src/
│  └─ langgraph_novel_lab/
│     ├─ __init__.py
│     └─ main.py
├─ notebooks/
│  └─ demo.ipynb
├─ .env.example
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## Setup (venv)

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure `.env` (local OpenAI-compatible provider)

Copy and edit:

### macOS / Linux

```bash
cp .env.example .env
```

### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
```

Required values:

- `OPENAI_BASE_URL` (example: `http://localhost:1234/v1`)
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

Optional values:

- `STAGE_PREMISE`
- `TARGET_DIALOGUE_TURNS` (default: `6`)

## Run from terminal

```bash
python -m src.langgraph_novel_lab.main
```

This prints:

- manager routing logs
- scene notes
- dialogue history
- scene draft
- audit summary

## Run from notebook

Open `notebooks/demo.ipynb` and run cells in order.

The notebook already includes project-root path setup and async invocation with:

```python
result = await app.ainvoke({"premise": premise, "dialogue_history": []})
```

## Extend ideas

- add more characters
- add retry/revision loop after `auditor`
- let manager use an LLM policy instead of deterministic rules
- store structured dialogue turns (dicts instead of strings)
