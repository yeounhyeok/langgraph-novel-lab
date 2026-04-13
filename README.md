# langgraph-novel-lab

작은 멀티 에이전트 소설 장면 생성을 위한 LangGraph 미니 데모입니다.

## 이 프로젝트에서 볼 수 있는 것

- `next_node` 기반의 상태 중심 라우팅
- `turns`를 통한 명시적 턴 관리
- `character_a`와 `character_b`의 교차 대화 진행
- `director`가 부여한 캐릭터 핵심특성을 각 캐릭터가 최우선으로 유지하는 대화
- 메모/대사/초안/감수 결과의 한국어 중심 출력
- `manager -> director -> characters -> writer -> auditor`의 완결된 흐름
- 실행 중 각 노드가 생성한 내용을 즉시 보여주는 실시간 로그
- 초안 품질이 약할 때 감수 단계에서 재수정을 강제할 수 있는 보정 로직
- 첫 학습용으로 읽기 쉬운 단일 파일 구조

## 노드 설명

- `manager`: 장면 운영 메모 작성
- `director`: 연출 메모 작성 + 캐릭터 핵심특성 지정
- `character_a`: 인물 A의 다음 대사 생성
- `character_b`: 인물 B의 다음 대사 생성
- `writer`: 메모와 대화를 바탕으로 장면 초안 작성
- `auditor`: 초안 감수 및 라우팅 결정

## 상태(State) 필드

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

## 라우팅 개요

라우팅은 동적이지만 구조는 단순합니다.

```text
manager -> director -> character_a <-> character_b -> writer -> auditor
                                           ^                     |
                                           |---------------------|
```

핵심은 `character_a -> character_b -> writer`를 고정 파이프라인으로 박아두지 않는다는 점입니다.  
각 노드는 상태 일부를 갱신하고, 다음 이동은 `next_node`를 따라 결정됩니다.

`turns`와 `dialogue_history`를 바탕으로 그래프는 다음을 판단합니다.

- 대화를 계속할지
- 화자를 전환할지
- `writer`로 넘어갈지

또한 `auditor` 이후에는 다음이 가능합니다.

- 초안이 충분히 좋으면 종료
- `writer`로 되돌려 수정 라운드 진행
- 장면 기획 자체 보강이 필요하면 `manager`로 되돌림
- 단순 품질 체크 실패 시(`짧은 초안`, `대화 복붙 과다`, `긴장/장면 키워드 약함`) `pass`를 `revise`로 자동 보정

감수 결과는 한국어 태그 형식으로 출력되어, 왜 되돌아갔는지 초보자도 확인하기 쉽습니다.

## 캐릭터 특성 부여

`director` 노드는 연출 메모와 함께 아래 2개를 반드시 산출합니다.

- `인물A_핵심특성`
- `인물B_핵심특성`

이 값은 상태에 저장되고, `character_a`/`character_b` 프롬프트에서 최우선 규칙으로 사용됩니다.

## 프로젝트 구조

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

## 설치

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

## `.env` 설정

기본 템플릿(`.env.example`과 동일):

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini

# Ollama example (OpenAI-compatible endpoint)
#OPENAI_BASE_URL=http://localhost:11434/v1
#OPENAI_API_KEY=ollama
#OPENAI_MODEL=qwen2.5:7b-instruct

# Optional prompt override
#NOVEL_PREMISE=Two rival archivists must cooperate to decode a living library before it erases their memories.
#TARGET_DIALOGUE_TURNS=4
```

Ollama로 바로 실행하려면 아래처럼 바꿔서 사용하면 됩니다.

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b-instruct
```

추가로, `OPENAI_BASE_URL`이 `localhost:11434`를 가리킬 때 API 키가 비어 있으면 내부적으로 `OPENAI_API_KEY=ollama`를 자동 사용합니다.

## 실행

### Python 스크립트 실행

```bash
python -m src.langgraph_novel_lab.main
```

실행 시 메모, 대화 기록, 초안, 감수 결과, 수정 라운드 여부를 출력합니다.
또한 각 노드 실행 중 다음과 같은 실시간 로그가 함께 출력됩니다.

- `[manager] notes`
- `[director] notes`
- `[character_a] line`
- `[character_b] line`
- `[writer] draft`
- `[auditor] audit`

### Jupyter 노트북 실행

초보자용 노트북은 `notebooks/demo.ipynb`에 포함되어 있습니다.
import/state/graph 실행/결과 확인 흐름을 단계적으로 볼 수 있습니다.

```bash
pip install jupyter ipykernel nbformat nbclient
jupyter notebook notebooks/demo.ipynb
```

## 왜 입문용으로 좋은가

- 로직이 사실상 한 파일에 모여 있음
- 상태 필드 수가 적어 따라가기 쉬움
- 라우팅 규칙이 순수 Python으로 명확히 보임
- LangGraph 장점을 느끼기엔 충분한 복잡도
- 직접 수정/실험하기 쉬운 크기
