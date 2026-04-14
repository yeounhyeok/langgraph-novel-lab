from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

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
    character_a_trait: str
    character_b_trait: str
    dialogue_history: list[str]
    draft: str
    audit: str
    audit_status: Literal["pass", "revise"]
    audit_target: Literal["manager", "writer", "end"]
    revision_count: int
    max_revisions: int
    next_node: NodeName
    turns: int
    run_output_dir: str


SYSTEM_STYLE = (
    "당신은 아주 작은 교육용 LangGraph 멀티 에이전트 데모의 일부입니다. "
    "설명과 산출물은 모두 한국어로 작성하세요. 초보자도 읽기 쉽게, 구체적이고 협력적으로 쓰세요. "
    "현재 주어진 전제는 유지하고, 설정을 멋대로 바꾸지 마세요."
)

SPEAKER_A = "인물 A"
SPEAKER_B = "인물 B"
AUDIT_STATUS_LABELS = {"pass": "통과", "revise": "수정 필요"}
AUDIT_TARGET_LABELS = {"manager": "manager", "writer": "writer", "end": "end"}
RUN_CALL_LOGS: list[dict[str, Any]] = []


def detect_provider(base_url: str) -> str:
    normalized = base_url.strip().lower()
    if not normalized:
        return "openai"
    if "api.openai.com" in normalized:
        return "openai"
    if "ollama" in normalized or "11434" in normalized:
        return "ollama"
    parsed = urlparse(normalized)
    return parsed.netloc or "custom"


def create_run_dir() -> Path:
    root = Path(os.getenv("NOVEL_RUNS_DIR", "logs/runs").strip() or "logs/runs")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(state: StageState, total_elapsed_ms: int) -> Path:
    run_dir = create_run_dir()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    provider = detect_provider(base_url)

    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "total_elapsed_ms": total_elapsed_ms,
        "total_elapsed_sec": round(total_elapsed_ms / 1000, 3),
        "llm_call_count": len(RUN_CALL_LOGS),
        "llm_calls": RUN_CALL_LOGS,
        "final_status": {
            "turns": state.get("turns", 0),
            "revision_count": state.get("revision_count", 0),
            "audit_status": state.get("audit_status", ""),
            "audit_target": state.get("audit_target", ""),
        },
    }

    outputs_md = (
        f"# Run Output\n\n"
        f"- Saved At: {summary['saved_at']}\n"
        f"- Provider: {provider}\n"
        f"- Model: {model}\n"
        f"- Base URL: {base_url or '(default)'}\n"
        f"- Total Time: {total_elapsed_ms} ms\n"
        f"- LLM Calls: {len(RUN_CALL_LOGS)}\n\n"
        "## Premise\n"
        f"{state.get('premise', '')}\n\n"
        "## Manager Notes\n"
        f"{state.get('manager_notes', '')}\n\n"
        "## Director Notes\n"
        f"{state.get('director_notes', '')}\n\n"
        "## Character Traits\n"
        f"- {SPEAKER_A}: {state.get('character_a_trait', '')}\n"
        f"- {SPEAKER_B}: {state.get('character_b_trait', '')}\n\n"
        "## Dialogue History\n"
        + ("\n".join(state.get("dialogue_history", [])) or "(없음)")
        + "\n\n## Draft\n"
        + f"{state.get('draft', '')}\n\n"
        + "## Audit\n"
        + f"{state.get('audit', '')}\n"
    )

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "outputs.md").write_text(outputs_md, encoding="utf-8")
    return run_dir


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
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    is_ollama = "localhost:11434" in base_url or "127.0.0.1:11434" in base_url
    if not api_key:
        if is_ollama:
            # Ollama's OpenAI-compatible endpoint does not require a real API key.
            api_key = "ollama"
        else:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return AsyncOpenAI(**kwargs)


async def call_model(client: AsyncOpenAI, role: str, task: str, premise: str) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    provider = detect_provider(base_url)
    started = perf_counter()
    response = await client.chat.completions.create(
        model=model,
        temperature=1,
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
    elapsed_ms = int((perf_counter() - started) * 1000)
    content = (response.choices[0].message.content or "").strip()
    RUN_CALL_LOGS.append(
        {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "provider": provider,
            "model": model,
            "elapsed_ms": elapsed_ms,
            "output_chars": len(content),
            "finish_reason": response.choices[0].finish_reason,
        }
    )
    print(
        f"[llm] role={role} provider={provider} model={model} "
        f"elapsed_ms={elapsed_ms} output_chars={len(content)}"
    )
    return content


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


def extract_character_traits(director_notes: str) -> tuple[str, str]:
    trait_a = parse_tagged_value(director_notes, "인물A_핵심특성") or "직진형 결단가"
    trait_b = parse_tagged_value(director_notes, "인물B_핵심특성") or "신중한 전략가"
    return trait_a, trait_b


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


def print_node_message(node: str, label: str, text: str) -> None:
    print(f"[{node}] {label}")
    print(text)


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
        "너무 압축하지 말고, 뒤의 대화가 생생하게 굉장히 구체적인 긴장 요소(갈등, 전투(중, 혹은 돌입), 등장신 등)를 넣으세요.\n"
        "수정 재진입이라면 감수에서 약하다고 지적한 부분을 더 선명하게 보강하세요.\n\n"
        f"현재까지의 대화:\n{history_text}"
        f"{revision_note}"
    )
    notes = await call_model(client, "manager", task, state["premise"])
    next_node = "director"
    print(f"[manager] revision_count={state.get('revision_count', 0)} next_node={next_node}")
    print_node_message("manager", "notes", notes)
    return {"manager_notes": notes, "next_node": next_node}


async def director(state: StageState) -> StageState:
    client = build_client()
    task = (
        "장면 지시문을 한국어 불릿 3개로 작성하세요. 아래 순서를 지키세요.\n"
        "- 배경/장소\n"
        "- 분위기\n"
        "- 당장 닥친 압박\n\n"
        "그리고 반드시 마지막에 아래 2줄을 추가하세요.\n"
        "인물A_핵심특성: (한 줄)\n"
        "인물B_핵심특성: (한 줄)\n\n"
        "초보자도 읽기 쉽도록 짧고 선명하게 쓰세요.\n"
        "수정 라운드라면 감수에서 지적한 약점을 메울 수 있는 무대 압박을 더 또렷하게 잡으세요.\n\n"
        f"매니저 메모:\n{state['manager_notes']}\n\n"
        f"감수 의견(있다면 참고):\n{state.get('audit', '(없음)')}"
    )
    notes = await call_model(client, "director", task, state["premise"])
    trait_a, trait_b = extract_character_traits(notes)
    next_node = choose_next_node({**state, "director_notes": notes})
    print(f"[director] next_node={next_node}")
    print_node_message("director", "notes", notes)
    return {
        "director_notes": notes,
        "character_a_trait": trait_a,
        "character_b_trait": trait_b,
        "next_node": next_node,
    }


async def character_a(state: StageState) -> StageState:
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) or "(아직 대화 없음)"
    trait_a = state.get("character_a_trait", "직진형 결단가")
    task = (
        f"{SPEAKER_A}로 말하세요. 한국어 대사만 출력하세요. 화자 라벨은 붙이지 마세요.\n"
        f"최우선 규칙: 아래 핵심특성을 절대 유지하세요 -> {trait_a}\n"
        "말투/판단/반응에서 이 특성이 가장 먼저 드러나야 합니다.\n"
        "실제 상황이라 간주하고 직전 발화에 분명히 응답하세요. 1~2문장으로 충분한 내용을 담으세요.\n"
        "수정 라운드라면 이전보다 더 직접적으로 충돌하거나 보완해 장면의 결을 바꾸세요.\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"감수 의견(있다면 참고):\n{state.get('audit', '(없음)')}\n\n"
        f"현재까지의 대화:\n{history_text}"
    )
    line = sanitize_line(await call_model(client, "character_a", task, state["premise"]))
    new_history = [*history, f"{SPEAKER_A}: {line}"]
    next_node = choose_next_node({**state, "dialogue_history": new_history, "turns": state.get("turns", 0) + 1})
    print(f"[character_a] turns={state.get('turns', 0) + 1} next_node={next_node}")
    print_node_message("character_a", "line", line)
    return {"dialogue_history": new_history, "turns": state.get("turns", 0) + 1, "next_node": next_node}


async def character_b(state: StageState) -> StageState:
    client = build_client()
    history = state.get("dialogue_history", [])
    history_text = "\n".join(history) or "(아직 대화 없음)"
    trait_b = state.get("character_b_trait", "신중한 전략가")
    task = (
        f"{SPEAKER_B}로 말하세요. 한국어 대사만 출력하세요. 화자 라벨은 붙이지 마세요.\n"
        f"최우선 규칙: 아래 핵심특성을 절대 유지하세요 -> {trait_b}\n"
        "말투/판단/반응에서 이 특성이 가장 먼저 드러나야 합니다.\n"
        "실제 상황이라 간주하고 직전 발화에 분명히 응답하세요. 1~2문장으로 충분한 내용을 담으세요.\n"
        "수정 라운드라면 이전보다 더 직접적으로 충돌하거나 보완해 장면의 결을 바꾸세요.\n\n"
        f"연출 메모:\n{state['director_notes']}\n\n"
        f"감수 의견(있다면 참고):\n{state.get('audit', '(없음)')}\n\n"
        f"현재까지의 대화:\n{history_text}"
    )
    line = sanitize_line(await call_model(client, "character_b", task, state["premise"]))
    new_history = [*history, f"{SPEAKER_B}: {line}"]
    next_node = choose_next_node({**state, "dialogue_history": new_history, "turns": state.get("turns", 0) + 1})
    print(f"[character_b] turns={state.get('turns', 0) + 1} next_node={next_node}")
    print_node_message("character_b", "line", line)
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
    print_node_message("writer", "draft", draft)
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
    print_node_message("auditor", "audit", audit)
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
    RUN_CALL_LOGS.clear()
    app = build_graph()
    initial_state: StageState = {
        "premise": premise,
        "dialogue_history": [],
        "turns": 0,
        "revision_count": 0,
        "max_revisions": 5,
        "next_node": "manager",
    }
    started = perf_counter()
    result = await app.ainvoke(initial_state)
    total_elapsed_ms = int((perf_counter() - started) * 1000)

    save_enabled = os.getenv("NOVEL_SAVE_RUN", "1").strip().lower() in {"1", "true", "yes", "on"}
    if save_enabled:
        run_dir = save_run_artifacts(result, total_elapsed_ms)
        result["run_output_dir"] = str(run_dir)
        print(f"[run] 저장 완료: {run_dir}")

    return result


def print_result(state: StageState) -> None:
    print("=== 전제 ===")
    print(state["premise"])
    print("\n=== 매니저 메모 ===")
    print(state.get("manager_notes", ""))
    print("\n=== 디렉터 메모 ===")
    print(state.get("director_notes", ""))
    print("\n=== 캐릭터 핵심특성 ===")
    print(f"{SPEAKER_A}: {state.get('character_a_trait', '(없음)')}")
    print(f"{SPEAKER_B}: {state.get('character_b_trait', '(없음)')}")
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
        "화려한 케이팝 걸그룹 '헌트릭스(HUNTR/X)' 멤버들이 밤에는 세상을 위협하는 악령을 물리치는 데몬 헌터로 변신하는 판타지 액션물입니다. 3인조 걸그룹 루미, 미라, 조이가 노래와 퍼포먼스, 한국 전통 무기를 활용해 악귀를 퇴치하는 이중생활을 그립니다. 핵심 설정 및 시놉시스"
        "이중생활: 낮에는 세계적인 슈퍼스타 케이팝 걸그룹, 밤에는 악마를 사냥하여 인간 세상을 지키는 헌터.",
        "전투 방식: 헌트릭스는 악령들을 물리치고 이들을 봉인하는 '황금 혼문(魂門)'을 만들기 위해 노래와 춤을 무기로 사용.",
        "대립 구조: 팬들의 영혼을 빼앗으려는 라이벌 악령 보이 그룹 '사자 보이즈(Saja Boys)'와 대결.",
        "한국적 요세: 멤버들은 각각 사인검, 월도, 신카이 등 한국 전통 무기를 개성 있게 사용.",
    )
    result = await run_demo(premise)
    print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
