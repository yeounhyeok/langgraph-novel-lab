from __future__ import annotations

import asyncio
import os
import re
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
    audit_status: Literal["pass", "revise"]
    audit_target: Literal["manager", "writer", "end"]
    revision_count: int
    max_revisions: int
    next_node: NodeName
    turns: int


SYSTEM_STYLE = (
    "당신은 아주 작은 교육용 LangGraph 멀티 에이전트 데모의 일부입니다. "
    "설명과 산출물은 모두 한국어로 작성하세요. 초보자도 읽기 쉽게, 구체적이고 협력적으로 쓰세요. "
    "현재 주어진 전제는 유지하고, 설정을 멋대로 바꾸지 마세요."
)

SPEAKER_A = "인물 A"
SPEAKER_B = "인물 B"
AUDIT_STATUS_LABELS = {"pass": "통과", "revise": "수정 필요"}
AUDIT_TARGET_LABELS = {"manager": "manager", "writer": "writer", "end": "end"}


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


def parse_tagged_value(text: str, label: str) -> str:
    pattern = rf"^{re.escape(label)}\s*:\s*(.+)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def strip_speaker_prefix(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    return re.sub(r"^[^:：]+:\s*", "", cleaned)


def parse_audit_result(audit: str) -> tuple[str, str]:
    status_raw = parse_tagged_value(audit, "판정").lower()
    target_raw = parse_tagged_value(audit, "되돌아갈_노드").lower()

    status = "pass" if "통과" in status_raw or status_raw == "pass" else "revise"

    if target_raw in {"manager", "writer", "end"}:
        target = target_raw
    elif "manager" in target_raw or "매니저" in target_raw:
        target = "manager"
    elif "writer" in target_raw or "작가" in target_raw:
        target = "writer"
    else:
        target = "end" if status == "pass" else "writer"

    if status == "pass":
        target = "end"

    return status, target


def extract_audit_sections(audit: str) -> dict[str, str]:
    return {
        "강점": parse_tagged_value(audit, "강점") or "(없음)",
        "약한_지점": parse_tagged_value(audit, "약한_지점") or "(없음)",
        "보강_포인트": parse_tagged_value(audit, "보강_포인트") or "(없음)",
        "수정_지시": parse_tagged_value(audit, "수정_지시") or "(없음)",
    }


def should_force_revision(draft: str, dialogue_history: list[str]) -> str | None:
    text = " ".join(draft.split()).strip()
    if len(text) < 350:
        return "초안 분량이 너무 짧습니다."

    normalized_draft = " ".join(text.lower().split())
    copied_lines = 0
    for line in dialogue_history:
        normalized_line = strip_speaker_prefix(line).lower()
        normalized_line = " ".join(normalized_line.split())
        if len(normalized_line) >= 12 and normalized_line in normalized_draft:
            copied_lines += 1

    if dialogue_history and copied_lines >= max(3, len(dialogue_history) // 2):
        return "대화 기록을 너무 많이 그대로 옮겼습니다."

    scene_terms = ("리더", "악령", "공연", "무대", "팀")
    tension_terms = ("긴장", "압박", "충돌", "갈등")

    scene_hits = sum(term in text for term in scene_terms)
    if scene_hits < 3:
        return "장면의 핵심 요소가 충분히 드러나지 않았습니다."

    if not any(term in text for term in tension_terms):
        return "갈등이나 압박이 충분히 선명하지 않습니다."

    return None


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
    revision_note = ""
    if state.get("revision_count", 0) > 0 and state.get("audit"):
        revision_note = (
            "\n\n이전 감수에서 수정 요청이 왔습니다. 아래 감수 의견을 반영해 메모를 다시 잡아 주세요.\n"
            f"감수 의견:\n{state['audit']}"
        )
    task = (
        "한 장면용 운영 메모를 작성하세요. 아래 형식의 한국어 불릿 3개만 작성하세요.\n"
        "- 장면 톤\n"
        "- 이번 장면의 위험/목표\n"
        "- 두 리더 사이에서 드러나야 할 감정적 마찰\n\n"
        "너무 압축하지 말고, 뒤의 대화가 길고 살아 움직이도록 구체적인 긴장 요소를 넣으세요.\n"
        "수정 재진입이라면 감수에서 약하다고 지적한 부분을 더 선명하게 보강하세요.\n\n"
        f"현재까지의 대화:\n{history_text}"
        f"{revision_note}"
    )
    notes = await call_model(client, "manager", task, state["premise"])
    next_node = "director"
    print(f"[manager] revision_count={state.get('revision_count', 0)} next_node={next_node}")
    return {"manager_notes": notes, "next_node": next_node}


async def director(state: StageState) -> StageState:
    client = build_client()
    task = (
        "장면 지시문을 한국어 불릿 3개로 작성하세요. 아래 순서를 지키세요.\n"
        "- 배경/장소\n"
        "- 분위기\n"
        "- 당장 닥친 압박\n\n"
        "초보자도 읽기 쉽도록 짧고 선명하게 쓰되, K-팝 데몬 헌터 설정은 유지하세요.\n"
        "수정 라운드라면 감수에서 지적한 약점을 메울 수 있는 무대 압박을 더 또렷하게 잡으세요.\n\n"
        f"매니저 메모:\n{state['manager_notes']}\n\n"
        f"감수 의견(있다면 참고):\n{state.get('audit', '(없음)')}"
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
        "상대 리더와의 갈등, 팀의 공연 책임, 악령 사냥의 긴박함을 함께 반영하세요.\n"
        "수정 라운드라면 감수에서 약하다고 한 부분을 보강하는 방향으로 새 대사를 쌓으세요.\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"감수 의견(있다면 참고):\n{state.get('audit', '(없음)')}\n\n"
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
        "반박, 걱정, 책임감, 그리고 팀을 향한 시선을 함께 드러내세요.\n"
        "수정 라운드라면 이전보다 더 직접적으로 충돌하거나 보완해 장면의 결을 바꾸세요.\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"감수 의견(있다면 참고):\n{state.get('audit', '(없음)')}\n\n"
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
    revision_note = ""
    if state.get("revision_count", 0) > 0 and state.get("audit"):
        sections = extract_audit_sections(state["audit"])
        revision_note = (
            "\n\n재작성 모드입니다. 아래 감수 결과를 반드시 반영해 초안을 새로 써야 합니다.\n"
            f"- 강점: {sections['강점']}\n"
            f"- 약한_지점: {sections['약한_지점']}\n"
            f"- 보강_포인트: {sections['보강_포인트']}\n"
            f"- 수정_지시: {sections['수정_지시']}\n\n"
            "기존 문장을 조금 다듬는 수준이 아니라, 장면의 압력과 응답을 다시 설계하세요."
        )
    task = (
        "위 메모와 대화를 바탕으로 한국어 장면 초안을 작성하세요.\n"
        "분량은 대략 500~900자로 하며, 대사를 충분히 살리고 서술도 자연스럽게 이어 주세요.\n"
        "K-팝 데몬 헌터라는 현재 전제는 유지하고, 두 리더 사이의 긴장과 팀 전체의 부담이 동시에 느껴지게 쓰세요.\n"
        "특히 반복적인 말주고받기만 늘어놓지 말고, 감정 변화나 상황 압박이 한 단계 더 움직이는 장면으로 만드세요.\n"
        "현재 대화 기록을 그대로 복붙하지 말고, 장면 서술과 대사의 비율을 새로 설계하세요.\n"
        "가능하면 행동, 시선, 공간 압박 같은 구체적 묘사를 함께 넣어서 문단형 장면으로 완성하세요.\n\n"
        f"매니저 메모:\n{state['manager_notes']}\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"대화 기록:\n{history_text}"
        f"{revision_note}"
    )
    draft = await call_model(client, "writer", task, state["premise"])
    next_node = "auditor"
    print(f"[writer] revision_count={state.get('revision_count', 0)} next_node={next_node}")
    return {"draft": draft, "next_node": next_node}


async def auditor(state: StageState) -> StageState:
    client = build_client()
    max_revisions = state.get("max_revisions", 1)
    revision_count = state.get("revision_count", 0)
    task = (
        "초안을 한국어로 감수하세요. 반드시 아래 형식을 그대로 지키세요.\n"
        "판정: 통과 또는 수정 필요\n"
        "되돌아갈_노드: manager 또는 writer 또는 end\n"
        "강점: 한 줄\n"
        "약한_지점: 한 줄\n"
        "보강_포인트: 한 줄\n"
        "수정_지시: 한 줄\n\n"
        "규칙:\n"
        "- 강점은 실제로 잘 된 한 가지를 구체적으로 말하세요.\n"
        "- 약한_지점은 무엇이 부족한지 분명히 지적하세요.\n"
        "- 보강_포인트는 무엇을 더 세게 밀어야 하는지 적으세요.\n"
        "- 수정_지시는 어느 노드가 다시 일해야 하는지 드러나게 쓰세요.\n"
        "- 전제가 유지되고 긴장과 변화가 충분하면 통과를 주세요. 그렇지 않으면 수정 필요를 주세요.\n"
        f"- 이미 {revision_count}번 수정했습니다. 최대 수정 횟수는 {max_revisions}번입니다. 마지막 허용 수정 이후라면 통과 쪽으로 마무리하지 말고, 판정은 자유롭게 하되 되돌아갈_노드는 end로 지정하세요.\n\n"
        f"초안:\n{state['draft']}"
    )
    audit = await call_model(client, "auditor", task, state["premise"])
    audit_status, audit_target = parse_audit_result(audit)
    heuristic_reason = should_force_revision(state["draft"], state.get("dialogue_history", []))

    if audit_status == "pass" and heuristic_reason and revision_count < max_revisions:
        audit_status = "revise"
        audit_target = "writer"
        audit = f"{audit}\n자동_보정: {heuristic_reason}"

    if audit_status == "revise" and revision_count < max_revisions:
        next_node = audit_target if audit_target in {"manager", "writer"} else "writer"
        new_revision_count = revision_count + 1
    else:
        next_node = "end"
        new_revision_count = revision_count
        audit_target = "end"

    print(
        "[auditor] "
        f"status={audit_status} target={audit_target} revision_count={new_revision_count} next_node={next_node}"
    )
    return {
        "audit": audit,
        "audit_status": audit_status,
        "audit_target": audit_target,
        "revision_count": new_revision_count,
        "next_node": next_node,
    }


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
            "manager": "manager",
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
            "manager": "manager",
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
            "manager": "manager",
            "character_a": "character_a",
            "character_b": "character_b",
            "writer": "writer",
            "auditor": "auditor",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "writer",
        route,
        {"manager": "manager", "writer": "writer", "auditor": "auditor", "end": END},
    )
    graph.add_conditional_edges(
        "auditor",
        route,
        {"manager": "manager", "writer": "writer", "end": END},
    )

    return graph.compile()


async def run_demo(premise: str) -> StageState:
    app = build_graph()
    initial_state: StageState = {
        "premise": premise,
        "dialogue_history": [],
        "turns": 0,
        "revision_count": 0,
        "max_revisions": 1,
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
    print("\n=== 감수 요약 ===")
    print(f"판정: {AUDIT_STATUS_LABELS.get(state.get('audit_status', 'pass'), state.get('audit_status', 'pass'))}")
    print(f"되돌아간 노드: {state.get('audit_target', 'end')}")
    print(f"수정 라운드 수: {state.get('revision_count', 0)}")


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
