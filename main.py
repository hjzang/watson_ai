
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# --- 최신 Langchain Core Import 경로 사용 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# ---

# --- 설정 ---
CHROMA_PERSIST_DIR = "chroma_db"       # 'embed_data.py'로 만든 DB 폴더
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"              # 최신 AI 모델
# ---

# 1. .env 파일에서 OPENAI_API_KEY 불러오기
if not load_dotenv():
    print("오류: .env 파일을 찾을 수 없습니다.")
    sys.exit(1)
if not os.getenv("OPENAI_API_KEY"):
    print("오류: OPENAI_API_KEY가 .env 파일에 없습니다.")
    sys.exit(1)

# 2. FastAPI 앱 생성
# (uvicorn이 찾을 변수 이름이 'app'이어야 합니다)
app = FastAPI()

# 3. AI '왓슨'의 시스템 프롬프트 (최종 완성 버전 - '연관 단서' 활용)
system_prompt = """
너는 임시정부 소속의 냉철한 수사 조수 AI '왓슨'이다.
너의 말투는 항상 냉철하고, 분석적이며, '탐정님'이라는 호칭을 사용한다.
너의 유일한 임무는 플레이어가 수집한 '단서(Context)'를 바탕으로 '황옥 경부 사건'을 분석하고 추리 방향을 제시하는 것이다.

[규칙 1: 정보 통제 (가장 중요)]
너는 **오직 '현재 확보된 단서(Context)' 안의 내용**에 대해서만 말할 수 있다.
1. Context의 '[상세 내용]'은 '사실'로, '[AI 분석 힌트]'는 너의 '전문적인 견해'로 사용하라.
2. **[연결 및 추론]**: 플레이어의 질문에 답할 때, Context의 '[연관 단서]'나 '[반론/의문]' 내용이 있다면 이를 **적극적으로 활용**하여, 단서 간의 연결점이나 모순점을 지적하는 **지능적인 분석가**처럼 말하라.
3. **[태그 숨김]** 너의 최종 답변에는 절대로 '[단서명]', '[상세 내용]', '[AI 분석 힌트]', '[연관 단서]' 같은 대괄호 태그를 포함하지 마라.
4. **[스포일러 방지]** Context에 없는 단서(예: '황옥의 고백')에 대해 물어도, "존재할 수 있다"거나 "아직 못 찾았다"고 힌트를 주지 마라. 대신 "현재 확보된 단서 중에는 그와 관련된 정보가 없습니다."라고만 답변하라.
5. **[오류 반박 (최우선)]** 만약 플레이어의 질문이 Context의 '사실'과 명백히 다를 경우 (예: B4가 '위조 여권'인데 '편지'라고 묻는 경우), 헷갈리지 말고 "탐정님, 정보가 잘못된 것 같습니다. 확보된 B4 증거는 '위조 여권'입니다."라고 명확히 사실을 바로잡아라.
6. **[추측 금지]** 플레이어의 질문이 Context의 '사실'로 뒷받침될 수 없는 '추측'이라면 (예: 'B4 위조 여권'을 보고 '독립 자금'이냐고 묻는 경우), "그 주장은 현재 확보된 단서만으로는 확인하기 어렵습니다."라고 명확히 선을 그어라. **절대 플레이어의 추측에 동조하거나 "가능합니다"라고 말하지 마라.**

[규칙 2: 간결성 (매우 중요)]
1. **[사실 질문]**: 플레이어의 질문이 단순 확인(예: "이게 뭐야?", "A1이 뭐야?")일 경우, Context의 '[상세 내용]'을 바탕으로 **'사실(Fact)'만 1-2 문장으로 짧게** 답하라. (예: "그것은 황옥의 특무대 출장 증명서입니다.")
2. **[분석 질문]**: 플레이어의 질문이 구체적인 분석/의미(예: "이게 왜 중요해?", "분석해줘", "A1이 밀정 증거야?", "A1과 B2가 모순되지 않아?")일 경우, Context의 '[AI 분석 힌트]'와 '[연관 단서]'를 바탕으로 **2-3 문장 이내로 간결하게** 분석/답변하라.

[규칙 3: 정답 금지]
절대 '밀정이다', '아니다'와 같은 직접적인 정답을 제시하지 마라.
항상 '...라는 주장을 뒷받침합니다', '...라는 의혹이 있습니다'와 같이
플레이어가 스스로 생각하도록 '방향성'만 제시하라.

[규K칙 4: 일상 대화]
만약 플레이어의 질문이 Context(단서)와 **완전히 관련 없고**,
'황옥', '증거', '사건' 등과도 **전혀 무관한** 질문일 경우 (예: "오늘 날씨 어때?", "너 밥 먹었어?", "농담해줘"),
아주 짧게(1-2 문장) 캐릭터를 유지하며 답하고, **그때만** "이제 다시 수사로 돌아가죠."라며 대화를 사건 중심으로 되돌려라.
"""
# 4. RAG 체인 컴포넌트 준비 (앱 실행 시 한 번만 로드)
try:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # DB 폴더 존재 확인
    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"오류: ChromaDB 폴더 '{CHROMA_PERSIST_DIR}'를 찾을 수 없습니다.")
        print("'embed_data.py'를 먼저 성공적으로 실행해야 합니다.")
        sys.exit(1)

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"), # 대화 기록 공간
        ("human", "현재까지 확보한 단서(Context)는 다음과 같습니다.\n---\n{context}\n---\n\n탐정(플레이어)의 질문: {question}")
    ])
    output_parser = StrOutputParser()

except Exception as e:
    print(f"오류: AI 컴포넌트 초기화 실패. API 키 또는 DB 폴더({CHROMA_PERSIST_DIR})를 확인하세요. ({e})")
    sys.exit(1)

# 5. 백엔드 요청 형식 정의
class ChatRequest(BaseModel):
    question: str
    acquired_clue_list: List[str] = []
    chat_history: List[str] = []

# --- [ 채점 시스템 모델 ] ---
class ScoreRequest(BaseModel):
    conclusion: str
    selected_evidence_ids: List[str]
    total_collected_ids: List[str]
    reasoning_text: str = ""

class ScoreResponse(BaseModel):
    total_score: int
    grade: str
    feedback: str
# --- [ /채점 시스템 모델 ] ---


# 6. (헬퍼 함수) 검색된 단서(Docs) 포맷팅
def format_docs(docs):
    if not docs:
        return "(현재 질문과 관련된 확보 단서 없음)"
    return "\n\n".join(doc.page_content for doc in docs)

# 7. (헬퍼 함수) 채팅 히스토리 포맷팅
def format_chat_history(history_list: List[str]) -> List:
    messages = []
    for i, message in enumerate(history_list):
        if i % 2 == 0: # 짝수 인덱스는 Human (플레이어)
            messages.append(HumanMessage(content=message))
        else: # 홀수 인덱스는 AI (왓슨)
            messages.append(AIMessage(content=message))
    return messages

# 8. AI의 '입과 귀' (API 엔드포인트)
@app.post("/api/ai/ask")
def ask_watson(request: ChatRequest):
    print(f"\n--- 새 요청 수신 ---")
    print(f"질문: {request.question}")
    print(f"인벤토리: {request.acquired_clue_list}")
    print(f"대화기록: {len(request.chat_history)}개")

    try:
        # 9. 인벤토리 목록으로 DB '필터' 생성
        if not request.acquired_clue_list:
            metadata_filter = {"clue_id": "NONE"} # 빈 인벤토리는 검색 불가
        else:
            metadata_filter = {"clue_id": {"$in": request.acquired_clue_list}}
        print(f"DB 검색 필터: {metadata_filter}")

        # 10. 필터 적용된 '검색기(Retriever)' 생성
        retriever = vectorstore.as_retriever(
            search_kwargs={'filter': metadata_filter}
        )

        # 11. RAG 체인 구성 (LCEL)
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x['question']) | retriever | format_docs, # 질문으로 검색 후 포맷팅
                "question": RunnableLambda(lambda x: x['question']), # 질문 그대로 전달
                "chat_history": RunnableLambda(lambda x: format_chat_history(x['chat_history'])) # 히스토리 포맷팅
            }
            | prompt
            | llm
            | output_parser
        )

        # 12. 체인 실행을 위한 입력 데이터 준비
        chain_input = {
            "question": request.question,
            "chat_history": request.chat_history
        }

        # 13. 체인 실행 (RAG + LLM 호출)
        print("AI 답변 생성 시작...")
        answer = rag_chain.invoke(chain_input)
        print(f"AI 답변 생성 완료: {answer}")

        return {"answer": answer}

    except Exception as e:
        print(f"!!! 오류 발생: AI 답변 생성 중 문제 발생: {e}")
        raise HTTPException(status_code=500, detail=f"AI 서버 내부 오류 발생")

# 3-2. 채점 전용 시스템 프롬프트 (최종-단순화 버전: 합리성 제거)
scoring_prompt_template = """
너는 '도와줘! 왓슨!' 게임의 AI 채점관이다.
너의 유일한 임무는 '플레이어 답안지'를 '정답지'와 '채점 기준'에 따라 **단계별로 계산**하고, 그 결과를 '최종 반환 JSON 형식'으로 **정확하게** 반환하는 것이다.
절대 이 형식 외에 다른 말은 하지 마라.

[정답지 (Ground Truth)]
- A (밀정 증거): [A1, A2, A3, A4, A5]
- B (비-밀정 증거): [B1, B2, B3, B4, B5]
- 핵심 증거 (총 10개): [A1, A2, A3, A4, A5, B1, B2, B3, B4, B5]

[플레이어 답안지]
- 결론: {conclusion}
- 선택한 근거: {selected_evidence}
- 수집한 전체 증거: {total_collected}
- 작성 이유: {reasoning}

[채점 프로세스 (순서대로 엄격히 수행)]
너는 아래 2가지 항목의 점수를 **개별적으로 계산**하고, 마지막에 **모두 합산**하라.
('작성 이유'는 점수에 반영하지 말고 무시한다.)

1. [점수 1: 증거 수집도 (최대 40점)]
   '수집한 전체 증거' 목록에 '핵심 증거 (총 10개)'가 몇 개 포함되어 있는지 정확히 세어라.
   **점수 1 = (찾은 개수) * 4**
   (예: 10개 찾았으면 점수 1 = 40)

2. [점수 2: 논리적 추론 (최대 60점)]
   '결론'과 '선택한 근거'가 '정답지'와 일치하는지 확인하라.
   - (A) '결론'이 "miljeong"이고, '선택한 근거'가 **모두** A 증거로만 이루어져 있다면: **점수 2 = 60**
   - (B) '결론'이 "anti_miljeong"이고, '선택한 근거'가 **모두** B 증거로만 이루어져 있다면: **점수 2 = 60**
   - (C) 위 2가지 경우가 아니라면 (예: 'miljeong' 결론에 B 증거를 근거로 들었다면): **점수 2 = 0**

[최종 점수 계산]
- **총점 = (점수 1) + (점수 2)**
- 등급: 'S' (95-100), 'A' (85-94), 'B' (70-84), 'C' (60-69), 'D' (60 미만)

[최종 반환 JSON 형식 (계산 시작)]
'플레이어 답안지'를 '채점 기준'에 따라 엄격하게 계산하라.
총점 = (점수 1) + (점수 2).
등급: 'S' (95-100), 'A' (85-94), 'B' (70-84), 'C' (60-69), 'D' (60 미만).
피드백 문자열을 생성하라.
**반드시 아래 JSON 형식만 반환하라.**

{{
  "total_score": (계산된 총점, 숫자),
  "grade": "(계산된 등급, 문자열)",
  "feedback": "증거 수집(점수1/40), 논리성(점수2/60)"
}}
"""
# 14-2. (신규) 채점 API 엔드포인트 (초-단순 Python 직접 계산)
@app.post("/api/ai/score", response_model=ScoreResponse)
async def score_report(request: ScoreRequest):
    print(f"\n--- 채점 요청 수신 (Python 직접 계산 / 단순 버전) ---")
    print(f"결론: {request.conclusion}")
    print(f"선택 증거: {request.selected_evidence_ids}")

    try:
        # [정답지 (Ground Truth)]
        A_EVIDENCE = {"A1", "A2", "A3", "A4", "A5"}
        B_EVIDENCE = {"B1", "B2", "B3", "B4", "B5"}
        CORE_EVIDENCE = A_EVIDENCE.union(B_EVIDENCE) # 10개

        # --- [Python이 '객관식' 점수 계산] ---
        score1_collected = 0
        score2_logic = 0

        # [점수 1: 증거 수집도 (최대 40점)]
        collected_count = len(set(request.total_collected_ids).intersection(CORE_EVIDENCE))
        score1_collected = collected_count * 4 # 1개당 4점

        # [점수 2: 논리적 추론 (최대 60점)]
        selected_set = set(request.selected_evidence_ids)
        if request.conclusion == "miljeong":
            # '밀정' 결론일 때, 선택한 증거가 모두 A증거의 부분집합인가?
            if selected_set.issubset(A_EVIDENCE):
                score2_logic = 60
        elif request.conclusion == "anti_miljeong":
            # '비밀정' 결론일 때, 선택한 증거가 모두 B증거의 부분집합인가?
            if selected_set.issubset(B_EVIDENCE):
                score2_logic = 60
        # (그 외 C의 경우, score2_logic는 0점으로 유지)

        # [최종 점수 계산]
        total_score = score1_collected + score2_logic
        
        # 등급 계산
        grade = "D"
        if total_score >= 95:
            grade = "S"
        elif total_score >= 85:
            grade = "A"
        elif total_score >= 70:
            grade = "B"
        elif total_score >= 60:
            grade = "C"
        
        feedback = f"증거 수집({score1_collected}/40), 논리성({score2_logic}/60)"

        print(f"채점 완료: 총점 {total_score}, 등급 {grade}")
        
        # Pydantic 모델로 변환하여 백엔드에 반환
        return ScoreResponse(
            total_score=total_score,
            grade=grade,
            feedback=feedback
        )

    except Exception as e:
        print(f"!!! 오류 발생: Python 채점 중 문제 발생: {e}")
        raise HTTPException(status_code=500, detail=f"Python 서버 채점 오류 발생")
# 14. (테스트용) 루트 경로
@app.get("/")
def read_root():
    return {"message": "AI 조수 '왓슨' (FastAPI) 서버가 작동 중입니다. API 문서는 /docs 에서 확인하세요."}

# 15. 서버 실행 설정 (uvicorn으로 실행할 때 사용됨)
if __name__ == "__main__":
    import uvicorn
    print("서버를 직접 실행합니다. (테스트용)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
