from __future__ import annotations

import asyncio
import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI


NodeName = Literal["director", "character_a", "character_b", "writer", "auditor", "end"]


class StageState(TypedDict, total=False):
    premise: str
    scene_notes: str
    dialogue_history: list[str]
    draft: str
    audit: str
    next_node: NodeName


SYSTEM_STYLE = (
    "You are part of a tiny educational multi-agent stage-play demo. "
    "Be concise, concrete, and collaborative."
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
    return response.choices[0].message.content.strip()


def _target_turns() -> int:
    raw = os.getenv("TARGET_DIALOGUE_TURNS", "6").strip()
    try:
        return max(2, int(raw))
    except ValueError:
        return 6


def _choose_next_node(state: StageState) -> NodeName:
    dialogue_history = state.get("dialogue_history", [])
    if not state.get("scene_notes"):
        return "director"

    if len(dialogue_history) < _target_turns():
        if not dialogue_history:
            return "character_a"
        last_line = dialogue_history[-1]
        if last_line.startswith("Character A:"):
            return "character_b"
        return "character_a"

    if not state.get("draft"):
        return "writer"
    if not state.get("audit"):
        return "auditor"
    return "end"


async def manager(state: StageState) -> StageState:
    print("[Manager] Routing started...")
    next_node = _choose_next_node(state)
    print(f"[Manager] Routing complete! next_node={next_node}")
    return {"next_node": next_node}


async def director(state: StageState) -> StageState:
    print("[Director] Scene setup started...")
    client = build_client()
    task = (
        "Set stage direction for a short play scene.\n"
        "Return 3 concise bullets: place, mood, immediate conflict.\n\n"
        f"Premise:\n{state['premise']}"
    )
    scene_notes = await call_model(client, "director", task, state["premise"])
    print("[Director] Scene setup complete!")
    return {"scene_notes": scene_notes}


async def character_a(state: StageState) -> StageState:
    print("[Character A] Turn started...")
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) if history else "(No dialogue yet)"
    task = (
        "Speak as Character A in one short line.\n"
        "React to the scene notes and prior dialogue.\n"
        "Return only the spoken line (no speaker label).\n\n"
        f"Scene notes:\n{state['scene_notes']}\n\n"
        f"Dialogue so far:\n{history_text}"
    )
    line = await call_model(client, "character_a", task, state["premise"])
    utterance = f"Character A: {' '.join(line.split())}"
    print("[Character A] Turn complete!")
    return {"dialogue_history": [*history, utterance]}


async def character_b(state: StageState) -> StageState:
    print("[Character B] Turn started...")
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) if history else "(No dialogue yet)"
    task = (
        "Speak as Character B in one short line.\n"
        "React to Character A and raise tension slightly.\n"
        "Return only the spoken line (no speaker label).\n\n"
        f"Scene notes:\n{state['scene_notes']}\n\n"
        f"Dialogue so far:\n{history_text}"
    )
    line = await call_model(client, "character_b", task, state["premise"])
    utterance = f"Character B: {' '.join(line.split())}"
    print("[Character B] Turn complete!")
    return {"dialogue_history": [*history, utterance]}


async def writer(state: StageState) -> StageState:
    print("[Writer] Drafting started...")
    client = build_client()
    history_text = "\n".join(state.get("dialogue_history", []))
    task = (
        "Turn the notes and dialogue into a short stage-play scene (200-350 words).\n"
        "Keep speaker labels and stage cues compact.\n\n"
        f"Scene notes:\n{state['scene_notes']}\n\n"
        f"Dialogue history:\n{history_text}"
    )
    notes = await call_model(client, "writer", task, state["premise"])
    print("[Writer] Drafting complete!")
    return {"draft": notes}


async def auditor(state: StageState) -> StageState:
    print("[Auditor] Review started...")
    client = build_client()
    task = (
        "Audit the draft in 3 bullets:\n"
        "- continuity\n"
        "- voice consistency\n"
        "- tension quality\n"
        "End with one concrete revision suggestion.\n\n"
        f"Draft:\n{state['draft']}"
    )
    notes = await call_model(client, "auditor", task, state["premise"])
    print("[Auditor] Review complete!")
    return {"audit": notes}


def route_from_manager(state: StageState) -> NodeName:
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
        route_from_manager,
        {
            "director": "director",
            "character_a": "character_a",
            "character_b": "character_b",
            "writer": "writer",
            "auditor": "auditor",
            "end": END,
        },
    )
    graph.add_edge("director", "manager")
    graph.add_edge("character_a", "manager")
    graph.add_edge("character_b", "manager")
    graph.add_edge("writer", "manager")
    graph.add_edge("auditor", "manager")

    return graph.compile()


async def run_demo(premise: str) -> StageState:
    app = build_graph()
    initial_state: StageState = {"premise": premise, "dialogue_history": []}
    return await app.ainvoke(initial_state)


def print_result(state: StageState) -> None:
    print("=== Premise ===")
    print(state["premise"])
    print("\n=== Scene Notes ===")
    print(state.get("scene_notes", ""))
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
        "STAGE_PREMISE",
        "Two estranged magicians reunite backstage before a final performance that may expose their shared secret.",
    )
    result = await run_demo(premise)
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
