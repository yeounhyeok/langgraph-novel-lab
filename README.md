# langgraph-novel-lab

A minimal LangGraph starter repo for experimenting with a multi-agent novel-writing workflow.

## What this is

This project shows a tiny educational pipeline where several agents collaborate on one short fiction scene:

- `manager` plans the story
- `director` shapes the scene
- `character_a` and `character_b` add viewpoint-specific notes
- `writer` drafts the scene
- `auditor` reviews the result

It is intentionally simple so you can learn the flow, modify prompts, and swap in your own local OpenAI-compatible model provider.

## Architecture overview

```text
manager -> director
             |- character_a ->
             |- character_b -> writer -> auditor -> END
```

State is passed through a LangGraph `StateGraph`. Each node calls the same chat-completions API with a different role prompt.

## Project structure

```text
langgraph-novel-lab/
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

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Windows (PowerShell)

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Configure `.env`

Copy the example file and edit it:

### macOS / Linux

```bash
cp .env.example .env
```

### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
```

Set these values for your local OpenAI-compatible provider:

- `OPENAI_BASE_URL` — API base URL, for example `http://localhost:1234/v1`
- `OPENAI_API_KEY` — any key your local provider expects
- `OPENAI_MODEL` — model name exposed by that provider
- `NOVEL_PREMISE` — optional custom premise for the demo

Examples of compatible local providers include LM Studio, Ollama bridges that expose an OpenAI-style endpoint, and vLLM-based local servers.

## Run the sample

```bash
python -m src.langgraph_novel_lab.main
```

On Windows:

```powershell
py -m src.langgraph_novel_lab.main
```

You should see the generated planning notes, character notes, draft scene, and audit output in the terminal.

## Publish to GitHub

This repo is initialized locally and ready to publish. If `gh` is authenticated, you can create a public repo with:

```bash
gh repo create langgraph-novel-lab --public --source=. --remote=origin --push
```
