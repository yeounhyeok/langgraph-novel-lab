from __future__ import annotations

import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI


class NovelState(TypedDict, total=False):
    premise: str
    manager_notes: str
    director_notes: str
    character_a_notes: str
    character_b_notes: str
    draft: str
    audit: str


SYSTEM_STYLE = (
    "You are part of a tiny educational multi-agent novel-writing demo. "
    "Be concise, concrete, and collaborative."
)


def build_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def call_model(client: OpenAI, role: str, task: str, premise: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": f"{SYSTEM_STYLE} Your current role is: {role}."},
            {
                "role": "user",
                "content": (
                    f"Premise:\n{premise}\n\n"
                    f"Task:\n{task}\n\n"
                    "Return plain text only."
                ),
            },
        ],
    )
    return response.choices[0].message.content.strip()


def manager(state: NovelState) -> NovelState:
    client = build_client()
    return {
        "manager_notes": call_model(
            client,
            role="manager",
            premise=state["premise"],
            task="Create a short story plan with genre, tone, stakes, and a 3-beat outline.",
        )
    }


def director(state: NovelState) -> NovelState:
    client = build_client()
    task = (
        "Using the manager notes below, produce scene direction for one short scene.\n\n"
        f"Manager notes:\n{state['manager_notes']}"
    )
    return {"director_notes": call_model(client, "director", task, state["premise"])}


def character_a(state: NovelState) -> NovelState:
    client = build_client()
    task = (
        "Write Character A's motivation, fear, and one memorable line for the scene.\n\n"
        f"Director notes:\n{state['director_notes']}"
    )
    return {"character_a_notes": call_model(client, "character_a", task, state["premise"])}


def character_b(state: NovelState) -> NovelState:
    client = build_client()
    task = (
        "Write Character B's motivation, conflict, and one memorable line for the scene.\n\n"
        f"Director notes:\n{state['director_notes']}"
    )
    return {"character_b_notes": call_model(client, "character_b", task, state["premise"])}


def writer(state: NovelState) -> NovelState:
    client = build_client()
    task = (
        "Draft one short scene (300-500 words) using all notes below.\n\n"
        f"Manager notes:\n{state['manager_notes']}\n\n"
        f"Director notes:\n{state['director_notes']}\n\n"
        f"Character A:\n{state['character_a_notes']}\n\n"
        f"Character B:\n{state['character_b_notes']}"
    )
    return {"draft": call_model(client, "writer", task, state["premise"])}


def auditor(state: NovelState) -> NovelState:
    client = build_client()
    task = (
        "Review the draft. Give 3 bullets: what works, what is weak, and the next revision idea.\n\n"
        f"Draft:\n{state['draft']}"
    )
    return {"audit": call_model(client, "auditor", task, state["premise"])}


def build_graph():
    graph = StateGraph(NovelState)
    graph.add_node("manager", manager)
    graph.add_node("director", director)
    graph.add_node("character_a", character_a)
    graph.add_node("character_b", character_b)
    graph.add_node("writer", writer)
    graph.add_node("auditor", auditor)

    graph.set_entry_point("manager")
    graph.add_edge("manager", "director")
    graph.add_edge("director", "character_a")
    graph.add_edge("director", "character_b")
    graph.add_edge("character_a", "writer")
    graph.add_edge("character_b", "writer")
    graph.add_edge("writer", "auditor")
    graph.add_edge("auditor", END)

    return graph.compile()


def main() -> None:
    load_dotenv()

    premise = os.getenv(
        "NOVEL_PREMISE",
        "Two rival archivists must cooperate to decode a living library before it erases their memories.",
    )

    app = build_graph()
    result = app.invoke({"premise": premise})

    print("=== Premise ===")
    print(result["premise"])
    print("\n=== Manager ===")
    print(result["manager_notes"])
    print("\n=== Director ===")
    print(result["director_notes"])
    print("\n=== Character A ===")
    print(result["character_a_notes"])
    print("\n=== Character B ===")
    print(result["character_b_notes"])
    print("\n=== Draft ===")
    print(result["draft"])
    print("\n=== Audit ===")
    print(result["audit"])


if __name__ == "__main__":
    main()
