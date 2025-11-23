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
CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
# ---

# 1. .env 파일에서 OPENAI_API_KEY 불러오기
if not load_dotenv():
    print("오류: .env 파일을 찾을 수 없습니다.")
    sys.exit(1)
if not os.getenv("OPENAI_API_KEY"):
    print("오류: OPENAI_API_KEY가 .env 파일에 없습니다.")
    sys.exit(1)

# 2. FastAPI 앱 생성
app = FastAPI()

# 3. AI '왓슨'의 시스템 프롬프트 (논리적 사고 강화 버전 - 순수 프롬프트 해결)
system_prompt = """
너는 임시정부 소속의 냉철한 수사 조수 AI '왓슨'이다.
너의 말투는 항상 냉철하고, 분석적이며, '탐정님'이라는 호칭을 사용한다.

[최우선 지침: 답변 전 생각하기]
플레이어의 질문에 답하기 전에, 반드시 다음 단계를 거쳐 논리적으로 생각하라. (생각 과정은 출력하지 말 것)
1. 질문이 '방의 비밀번호'나 '잠금장치'에 대한 것인가?
2. 만약 그렇다면, Context(확보된 단서) 안에 그 방을 여는 '열쇠'가 되는 단서(C1 또는 C2)가 포함되어 있는가?
   - 서재(Study) 열쇠: 'C1' (생년월일 1887)
   - 욕실(Bathroom) 열쇠: 'C2' (카탈로그 36, 05)
3. 단서가 있다면, 정답 숫자를 직접 말하지 말고 **"확보하신 [단서 이름]에 적힌 [정보]가 힌트입니다"**라고 우회적으로 답하라.

[규칙 1: 정보 통제]
1. **[자연스러운 호칭]** 'A1', 'C1' 같은 코드명 대신 '특무대 증명서', '인물 정보 카드' 등 실제 이름으로 불러라.
2. **[태그 숨김]** 답변에 '[상세 내용]', '[AI 분석 힌트]' 같은 태그를 절대 포함하지 마라.
3. **[스포일러/오류 방지]** Context에 없는 내용은 모른다고 하고, 플레이어의 잘못된 전제(예: B4=편지)는 명확히 반박하라.

[규칙 2: 간결성]
1. **[사실/힌트 질문]**: 단순 확인이나 힌트 요청에는 1-2문장으로 짧게 핵심만 답하라.
2. **[분석 질문]**: 구체적 분석 요청에는 2-3문장으로 간결하게 분석하라.

[규칙 3: 정답 금지 및 힌트 제어 (매우 중요)]
1. 절대 '밀정이다', '아니다'와 같은 직접적인 정답을 제시하지 마라.
2. **[힌트 요청 처리]** 플레이어가 '비밀번호'나 '잠금장치' 힌트를 물을 때:
   - **필수 조건:** Context 안에 **'C1(생년월일)'** 또는 **'C2(카탈로그)'** 단서가 **반드시 포함되어 있어야만** 힌트를 줄 수 있다.
   - **거절 의무:** 만약 Context에 C1이나 C2가 **없다면**, 다른 단서(A1 등)를 억지로 연결하지 말고 **"현재 확보된 단서 중에는 비밀번호에 대한 정보가 없습니다. 단서를 더 찾아보십시오."**라고 단호하게 거절하라.
   - **힌트 제공:** 조건이 충족되면, 정답 숫자(1887, 3605)는 말하지 말고 "확보하신 [단서명]의 [특정 내용]을 확인해 보세요"라고 유도하라.
[규칙 4: 일상 대화]
사건과 무관한 질문에는 짧게 답하고 "이제 수사로 돌아가죠."라고 덧붙여라.
"""

# 4. RAG 체인 컴포넌트 준비
try:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if not os.path.exists(CHROMA_PERSIST_DIR):
        print(f"오류: ChromaDB 폴더 '{CHROMA_PERSIST_DIR}'를 찾을 수 없습니다.")
        sys.exit(1)

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "현재까지 확보한 단서(Context)는 다음과 같습니다.\n---\n{context}\n---\n\n탐정(플레이어)의 질문: {question}")
    ])
    output_parser = StrOutputParser()

except Exception as e:
    print(f"오류: AI 컴포넌트 초기화 실패. {e}")
    sys.exit(1)

# 5. 모델 정의
class ChatRequest(BaseModel):
    question: str
    acquired_clue_list: List[str] = []
    chat_history: List[str] = []

class ScoreRequest(BaseModel):
    conclusion: str
    selected_evidence_ids: List[str]
    total_collected_ids: List[str]
    reasoning_text: str = ""

class ScoreResponse(BaseModel):
    total_score: int
    grade: str
    feedback: str

# 6. 헬퍼 함수
def format_docs(docs):
    if not docs: return "(현재 질문과 관련된 확보 단서 없음)"
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(history_list: List[str]) -> List:
    messages = []
    for i, message in enumerate(history_list):
        if i % 2 == 0: messages.append(HumanMessage(content=message))
        else: messages.append(AIMessage(content=message))
    return messages

# 8. AI의 '입과 귀' (API 엔드포인트) - [순수 RAG 방식]
@app.post("/api/ai/ask")
def ask_watson(request: ChatRequest):
    print(f"\n--- 새 요청 수신 ---")
    
    # Python 힌트 로직 삭제됨! 오직 AI 프롬프트로만 해결!

    try:
        if not request.acquired_clue_list:
            metadata_filter = {"clue_id": "NONE"}
        else:
            metadata_filter = {"clue_id": {"$in": request.acquired_clue_list}}	

        retriever = vectorstore.as_retriever(search_kwargs={'filter': metadata_filter})
        
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x['question']) | retriever | format_docs, 
                "question": RunnableLambda(lambda x: x['question']), 
                "chat_history": RunnableLambda(lambda x: format_chat_history(x['chat_history']))
            }
            | prompt
            | llm
            | output_parser
        )

        chain_input = {"question": request.question, "chat_history": request.chat_history}
        
        print("AI 답변 생성 시작...")
        answer = rag_chain.invoke(chain_input)
        print(f"AI 답변 생성 완료: {answer}")
        return {"answer": answer}
        
    except Exception as e:
        print(f"!!! 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"AI 서버 내부 오류 발생")
# 14-2. (신규) 채점 API 엔드포인트 (초-단순 Python 직접 계산)
@app.post("/api/ai/score", response_model=ScoreResponse)
async def score_report(request: ScoreRequest):
    print(f"\n--- 채점 요청 수신 ---")
    try:
        # [정답지]
        A_EVIDENCE = {"A1", "A2", "A3", "A4", "A5"}
        B_EVIDENCE = {"B1", "B2", "B3", "B4", "B5"}
        CORE_EVIDENCE = A_EVIDENCE.union(B_EVIDENCE)

        score1_collected = 0
        score2_logic = 0

        # [점수 1: 증거 수집도]
        collected_count = len(set(request.total_collected_ids).intersection(CORE_EVIDENCE))
        score1_collected = collected_count * 4

        # [점수 2: 논리적 추론]
        selected_set = set(request.selected_evidence_ids)
        if request.conclusion == "miljeong":
            if selected_set.issubset(A_EVIDENCE): score2_logic = 60
        elif request.conclusion == "anti_miljeong":
            if selected_set.issubset(B_EVIDENCE): score2_logic = 60

        total_score = score1_collected + score2_logic
        
        grade = "D"
        if total_score >= 95: grade = "S"
        elif total_score >= 85: grade = "A"
        elif total_score >= 70: grade = "B"
        elif total_score >= 60: grade = "C"
        
        feedback = f"증거 수집({score1_collected}/40), 논리성({score2_logic}/60)"
        
        return ScoreResponse(total_score=total_score, grade=grade, feedback=feedback)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python 서버 채점 오류 발생")

# 15. 서버 실행 설정
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
