from __future__ import annotations

import asyncio
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI


NodeName = Literal[
    "manager",
    "director",
    "character_a",
    "character_b",
    "writer",
    "auditor",
    "end",
]


class StageState(TypedDict, total=False):
    premise: str
    manager_notes: str
    director_notes: str
    dialogue_history: list[str]
    draft: str
    audit: str
    next_node: NodeName
    turns: int


SYSTEM_STYLE = (
    "You are part of a tiny educational LangGraph multi-agent demo. "
    "Be concise, concrete, collaborative, and easy to follow."
)


def build_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return AsyncOpenAI(**kwargs)


async def call_model(client: AsyncOpenAI, role: str, task: str, premise: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = await client.chat.completions.create(
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
    return (response.choices[0].message.content or "").strip()


def target_turns() -> int:
    raw = os.getenv("TARGET_DIALOGUE_TURNS", "4").strip()
    try:
        return max(2, int(raw))
    except ValueError:
        return 4


def choose_next_node(state: StageState) -> NodeName:
    turns = state.get("turns", 0)
    history = state.get("dialogue_history", [])

    if not state.get("manager_notes"):
        return "manager"
    if not state.get("director_notes"):
        return "director"
    if turns < target_turns():
        if not history or history[-1].startswith("Character B:"):
            return "character_a"
        return "character_b"
    if not state.get("draft"):
        return "writer"
    if not state.get("audit"):
        return "auditor"
    return "end"


async def manager(state: StageState) -> StageState:
    client = build_client()
    history_text = "\n".join(state.get("dialogue_history", [])) or "(No dialogue yet)"
    task = (
        "Create a short plan for one scene. Give 3 bullets for tone, stakes, and desired character friction. "
        "This plan should guide the rest of the graph.\n\n"
        f"Dialogue so far:\n{history_text}"
    )
    notes = await call_model(client, "manager", task, state["premise"])
    next_node = "director"
    print(f"[manager] next_node={next_node}")
    return {"manager_notes": notes, "next_node": next_node}


async def director(state: StageState) -> StageState:
    client = build_client()
    task = (
        "Write compact stage direction for the scene. Give 3 bullets: setting, mood, and immediate pressure.\n\n"
        f"Manager notes:\n{state['manager_notes']}"
    )
    notes = await call_model(client, "director", task, state["premise"])
    next_node = choose_next_node({**state, "director_notes": notes})
    print(f"[director] next_node={next_node}")
    return {"director_notes": notes, "next_node": next_node}


async def character_a(state: StageState) -> StageState:
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) or "(No dialogue yet)"
    task = (
        "Respond as Character A with one short spoken line. React to the stage direction and prior dialogue. "
        "Do not add speaker labels.\n\n"
        f"Director notes:\n{state['director_notes']}\n\n"
        f"Dialogue so far:\n{history_text}"
    )
    line = " ".join((await call_model(client, "character_a", task, state["premise"])).split())
    new_history = [*history, f"Character A: {line}"]
    next_node = choose_next_node({**state, "dialogue_history": new_history, "turns": state.get("turns", 0) + 1})
    print(f"[character_a] turns={state.get('turns', 0) + 1} next_node={next_node}")
    return {"dialogue_history": new_history, "turns": state.get("turns", 0) + 1, "next_node": next_node}


async def character_b(state: StageState) -> StageState:
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) or "(No dialogue yet)"
    task = (
        "Respond as Character B with one short spoken line. Answer Character A and raise tension a little. "
        "Do not add speaker labels.\n\n"
        f"Director notes:\n{state['director_notes']}\n\n"
        f"Dialogue so far:\n{history_text}"
    )
    line = " ".join((await call_model(client, "character_b", task, state["premise"])).split())
    new_history = [*history, f"Character B: {line}"]
    next_node = choose_next_node({**state, "dialogue_history": new_history, "turns": state.get("turns", 0) + 1})
    print(f"[character_b] turns={state.get('turns', 0) + 1} next_node={next_node}")
    return {"dialogue_history": new_history, "turns": state.get("turns", 0) + 1, "next_node": next_node}


async def writer(state: StageState) -> StageState:
    client = build_client()
    history_text = "\n".join(state.get("dialogue_history", []))
    task = (
        "Turn the notes and dialogue into one short polished scene of about 250-400 words. "
        "Preserve the character tension from the dialogue.\n\n"
        f"Manager notes:\n{state['manager_notes']}\n\n"
        f"Director notes:\n{state['director_notes']}\n\n"
        f"Dialogue history:\n{history_text}"
    )
    draft = await call_model(client, "writer", task, state["premise"])
    next_node = "auditor"
    print(f"[writer] next_node={next_node}")
    return {"draft": draft, "next_node": next_node}


async def auditor(state: StageState) -> StageState:
    client = build_client()
    task = (
        "Review the draft in 3 bullets: what works, what feels weak, and one concrete next improvement.\n\n"
        f"Draft:\n{state['draft']}"
    )
    audit = await call_model(client, "auditor", task, state["premise"])
    next_node = "end"
    print(f"[auditor] next_node={next_node}")
    return {"audit": audit, "next_node": next_node}


def route(state: StageState) -> NodeName:
    return state.get("next_node", "end")


def build_graph():
    graph = StateGraph(StageState)
    graph.add_node("manager", manager)
    graph.add_node("director", director)
    graph.add_node("character_a", character_a)
    graph.add_node("character_b", character_b)
    graph.add_node("writer", writer)
    graph.add_node("auditor", auditor)

    graph.set_entry_point("manager")
    graph.add_conditional_edges(
        "manager",
        route,
        {
            "director": "director",
            "character_a": "character_a",
            "character_b": "character_b",
            "writer": "writer",
            "auditor": "auditor",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "director",
        route,
        {
            "character_a": "character_a",
            "character_b": "character_b",
            "writer": "writer",
            "auditor": "auditor",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "character_a",
        route,
        {
            "character_a": "character_a",
            "character_b": "character_b",
            "writer": "writer",
            "auditor": "auditor",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "character_b",
        route,
        {
            "character_a": "character_a",
            "character_b": "character_b",
            "writer": "writer",
            "auditor": "auditor",
            "end": END,
        },
    )
    graph.add_conditional_edges("writer", route, {"auditor": "auditor", "end": END})
    graph.add_conditional_edges("auditor", route, {"end": END})

    return graph.compile()


async def run_demo(premise: str) -> StageState:
    app = build_graph()
    initial_state: StageState = {
        "premise": premise,
        "dialogue_history": [],
        "turns": 0,
        "next_node": "manager",
    }
    return await app.ainvoke(initial_state)


def print_result(state: StageState) -> None:
    print("=== Premise ===")
    print(state["premise"])
    print("\n=== Manager Notes ===")
    print(state.get("manager_notes", ""))
    print("\n=== Director Notes ===")
    print(state.get("director_notes", ""))
    print("\n=== Dialogue History ===")
    for line in state.get("dialogue_history", []):
        print(line)
    print("\n=== Draft ===")
    print(state.get("draft", ""))
    print("\n=== Audit ===")
    print(state.get("audit", ""))


async def main() -> None:
    load_dotenv()
    premise = os.getenv(
        "NOVEL_PREMISE",
        "Two rival archivists must cooperate to decode a living library before it erases their memories.",
    )
    result = await run_demo(premise)
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
