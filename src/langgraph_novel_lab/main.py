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
    "당신은 아주 작은 교육용 LangGraph 멀티 에이전트 데모의 일부입니다. "
    "설명과 산출물은 모두 한국어로 작성하세요. 초보자도 읽기 쉽게, 구체적이고 협력적으로 쓰세요. "
    "현재 주어진 전제는 유지하고, 설정을 멋대로 바꾸지 마세요."
)

SPEAKER_A = "인물 A"
SPEAKER_B = "인물 B"


def sanitize_line(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    for prefix in (
        f"{SPEAKER_A}:",
        f"{SPEAKER_B}:",
        "Character A:",
        "Character B:",
        "A:",
        "B:",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned.strip('"“”')


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
        temperature=0.8,
        messages=[
            {"role": "system", "content": f"{SYSTEM_STYLE} 현재 역할은 {role}입니다."},
            {
                "role": "user",
                "content": (
                    f"전제:\n{premise}\n\n"
                    f"작업:\n{task}\n\n"
                    "반드시 한국어 일반 텍스트만 출력하세요."
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def target_turns() -> int:
    raw = os.getenv("TARGET_DIALOGUE_TURNS", "6").strip()
    try:
        return max(4, int(raw))
    except ValueError:
        return 6


def choose_next_node(state: StageState) -> NodeName:
    turns = state.get("turns", 0)
    history = state.get("dialogue_history", [])

    if not state.get("manager_notes"):
        return "manager"
    if not state.get("director_notes"):
        return "director"
    if turns < target_turns():
        if not history or history[-1].startswith(f"{SPEAKER_B}:"):
            return "character_a"
        return "character_b"
    if not state.get("draft"):
        return "writer"
    if not state.get("audit"):
        return "auditor"
    return "end"


async def manager(state: StageState) -> StageState:
    client = build_client()
    history_text = "\n".join(state.get("dialogue_history", [])) or "(아직 대화 없음)"
    task = (
        "한 장면용 운영 메모를 작성하세요. 아래 형식의 한국어 불릿 3개만 작성하세요.\n"
        "- 장면 톤\n"
        "- 이번 장면의 위험/목표\n"
        "- 두 리더 사이에서 드러나야 할 감정적 마찰\n\n"
        "너무 압축하지 말고, 뒤의 대화가 길고 살아 움직이도록 구체적인 긴장 요소를 넣으세요.\n\n"
        f"현재까지의 대화:\n{history_text}"
    )
    notes = await call_model(client, "manager", task, state["premise"])
    next_node = "director"
    print(f"[manager] next_node={next_node}")
    return {"manager_notes": notes, "next_node": next_node}


async def director(state: StageState) -> StageState:
    client = build_client()
    task = (
        "장면 지시문을 한국어 불릿 3개로 작성하세요. 아래 순서를 지키세요.\n"
        "- 배경/장소\n"
        "- 분위기\n"
        "- 당장 닥친 압박\n\n"
        "초보자도 읽기 쉽도록 짧고 선명하게 쓰되, K-팝 데몬 헌터 설정은 유지하세요.\n\n"
        f"매니저 메모:\n{state['manager_notes']}"
    )
    notes = await call_model(client, "director", task, state["premise"])
    next_node = choose_next_node({**state, "director_notes": notes})
    print(f"[director] next_node={next_node}")
    return {"director_notes": notes, "next_node": next_node}


async def character_a(state: StageState) -> StageState:
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) or "(아직 대화 없음)"
    task = (
        f"{SPEAKER_A}로 말하세요. 한국어 대사만 출력하세요. 화자 라벨은 붙이지 마세요.\n"
        "한 줄이지만 내용은 충분히 실리게 쓰세요. 1~2문장으로 감정, 판단, 상황 반응이 모두 드러나야 합니다.\n"
        "상대 리더와의 갈등, 팀의 공연 책임, 악령 사냥의 긴박함을 함께 반영하세요.\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"현재까지의 대화:\n{history_text}"
    )
    line = sanitize_line(await call_model(client, "character_a", task, state["premise"]))
    new_history = [*history, f"{SPEAKER_A}: {line}"]
    next_node = choose_next_node({**state, "dialogue_history": new_history, "turns": state.get("turns", 0) + 1})
    print(f"[character_a] turns={state.get('turns', 0) + 1} next_node={next_node}")
    return {"dialogue_history": new_history, "turns": state.get("turns", 0) + 1, "next_node": next_node}


async def character_b(state: StageState) -> StageState:
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) or "(아직 대화 없음)"
    task = (
        f"{SPEAKER_B}로 말하세요. 한국어 대사만 출력하세요. 화자 라벨은 붙이지 마세요.\n"
        "직전 발화에 분명히 응답하면서 긴장을 더 끌어올리세요. 한 줄이지만 1~2문장으로 충분한 내용을 담으세요.\n"
        "반박, 걱정, 책임감, 그리고 팀을 향한 시선을 함께 드러내세요.\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"현재까지의 대화:\n{history_text}"
    )
    line = sanitize_line(await call_model(client, "character_b", task, state["premise"]))
    new_history = [*history, f"{SPEAKER_B}: {line}"]
    next_node = choose_next_node({**state, "dialogue_history": new_history, "turns": state.get("turns", 0) + 1})
    print(f"[character_b] turns={state.get('turns', 0) + 1} next_node={next_node}")
    return {"dialogue_history": new_history, "turns": state.get("turns", 0) + 1, "next_node": next_node}


async def writer(state: StageState) -> StageState:
    client = build_client()
    history_text = "\n".join(state.get("dialogue_history", []))
    task = (
        "위 메모와 대화를 바탕으로 한국어 장면 초안을 작성하세요.\n"
        "분량은 대략 500~800자로 하며, 대사를 충분히 살리고 서술도 자연스럽게 이어 주세요.\n"
        "K-팝 데몬 헌터라는 현재 전제는 유지하고, 두 리더 사이의 긴장과 팀 전체의 부담이 동시에 느껴지게 쓰세요.\n\n"
        f"매니저 메모:\n{state['manager_notes']}\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"대화 기록:\n{history_text}"
    )
    draft = await call_model(client, "writer", task, state["premise"])
    next_node = "auditor"
    print(f"[writer] next_node={next_node}")
    return {"draft": draft, "next_node": next_node}


async def auditor(state: StageState) -> StageState:
    client = build_client()
    task = (
        "초안을 한국어로 검토하세요. 아래 형식의 불릿 3개로만 답하세요.\n"
        "- 잘 작동하는 점\n"
        "- 다소 약한 점\n"
        "- 다음 개선 1가지\n\n"
        "평가는 초보자도 이해하기 쉽게 쓰고, 모호한 칭찬 대신 장면의 밀도와 대사 흐름을 기준으로 구체적으로 말하세요.\n\n"
        f"초안:\n{state['draft']}"
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
    print("=== 전제 ===")
    print(state["premise"])
    print("\n=== 매니저 메모 ===")
    print(state.get("manager_notes", ""))
    print("\n=== 디렉터 메모 ===")
    print(state.get("director_notes", ""))
    print("\n=== 대화 기록 ===")
    for line in state.get("dialogue_history", []):
        print(line)
    print("\n=== 장면 초안 ===")
    print(state.get("draft", ""))
    print("\n=== 감수 의견 ===")
    print(state.get("audit", ""))


async def main() -> None:
    load_dotenv()
    premise = os.getenv(
        "NOVEL_PREMISE",
        "낮에는 화려한 K-팝 아이돌 그룹 '루미너스'로 활동하지만, 밤에는 음악의 주파수로 도시를 잠식하는 악령들을 사냥하는 데몬 헌터들의 사투와 두 리더의 갈등.",
    )
    result = await run_demo(premise)
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
