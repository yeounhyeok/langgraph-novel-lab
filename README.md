# langgraph-novel-lab

Minimal LangGraph demo for a tiny multi-agent novel scene.

## What this project shows

- state-driven routing with `next_node`
- explicit turn tracking with `turns`
- alternating dialogue between `character_a` and `character_b`
- director-assigned character traits that each character must keep as top priority in dialogue
- Korean-first outputs for notes, dialogue, scene draft, and auditor feedback
- a small but complete flow: `manager -> director -> characters -> writer -> auditor`
- node-level live logs (each node prints its generated message while running)
- an auditor fallback heuristic that can force revision when a draft is too weak
- a structure that stays readable for a first LangGraph exercise

## Nodes

- `manager`: creates the high-level scene plan
- `director`: turns the plan into concrete stage direction
- `character_a`: adds Character A's next line
- `character_b`: adds Character B's next line
- `writer`: turns notes + dialogue into a short scene
- `auditor`: reviews the draft

## State

The graph state keeps:

- `premise`
- `manager_notes`
- `director_notes`
- `character_a_trait`
- `character_b_trait`
- `dialogue_history`
- `draft`
- `audit`
- `audit_status`
- `audit_target`
- `revision_count`
- `max_revisions`
- `next_node`
- `turns`

## Routing

Routing is dynamic, but intentionally simple:

```text
manager -> director -> character_a <-> character_b -> writer -> auditor
                                           ^                     |
                                           |---------------------|
```

The important part is that the graph does **not** hardcode a one-shot `character_a -> character_b -> writer` pipeline.
Instead, each node updates state, and routing follows `next_node`.

`turns` and `dialogue_history` decide whether the graph should:

- keep the dialogue going
- switch speakers
- stop the exchange and move to `writer`

After `auditor`, the graph can now:

- end the run when the draft is strong enough
- send the flow back to `writer` for a revision pass
- send the flow back to `manager` when the scene plan itself needs stronger tension or clearer stakes
- auto-correct a weak `pass` into `revise` when simple quality checks fail (short draft, too much dialogue copy, weak scene/tension terms)

The auditor output is intentionally tagged in Korean so beginners can inspect it and see why the graph routed backward.

The director now also emits:

- `인물A_핵심특성`
- `인물B_핵심특성`

and both character nodes are prompted to preserve those traits as their first-priority rule.

## Project structure

```text
langgraph-novel-lab/
├─ notebooks/
│  └─ demo.ipynb
├─ src/
│  └─ langgraph_novel_lab/
│     ├─ __init__.py
│     └─ main.py
├─ .env.example
├─ .gitignore
├─ README.md
└─ requirements.txt
```

## Setup

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

## `.env`

Example:

```env
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=qwen2.5-7b-instruct
NOVEL_PREMISE=Two rival archivists must cooperate to decode a living library before it erases their memories.
TARGET_DIALOGUE_TURNS=6
```

Ollama example:

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b-instruct
NOVEL_PREMISE=Two rival archivists must cooperate to decode a living library before it erases their memories.
TARGET_DIALOGUE_TURNS=6
```

For Ollama, this project also auto-falls back to `OPENAI_API_KEY=ollama` when the key is missing and `OPENAI_BASE_URL` points to `localhost:11434`.

## Run

### Python script

```bash
python -m src.langgraph_novel_lab.main
```

The script prints the notes, dialogue history, draft, audit result, and whether a revision loop happened in Korean.
During execution, each node also prints live logs like:

- `[manager] notes`
- `[director] notes`
- `[character_a] line`
- `[character_b] line`
- `[writer] draft`
- `[auditor] audit`

### Jupyter notebook

A beginner-friendly notebook is included at `notebooks/demo.ipynb`.
It walks through imports, state setup, graph creation, execution, and inspecting outputs with the K-pop demon hunters premise used in this repo.

If you want to open it interactively:

```bash
pip install jupyter ipykernel nbformat nbclient
jupyter notebook notebooks/demo.ipynb
```

## Why this is a good first learning structure

- only one file of logic
- only a few state fields
- routing rules are visible in plain Python
- enough moving parts to show why LangGraph is useful
- small enough to modify without getting lost
