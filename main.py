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

# .env 파일 있으면 로드 (로컬 개발용)
load_dotenv()

# 환경변수에서 가져오기 (Docker에서는 docker-compose가 전달)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

print("✅ OPENAI_API_KEY 로드 성공")

# 2. FastAPI 앱 생성
app = FastAPI()

# 3. AI '왓슨'의 시스템 프롬프트 (최종 HINT 우선권 부여)
system_prompt = """
너는 임시정부 소속의 냉철한 수사 조수 AI '왓슨'이다.
너의 말투는 항상 냉철하고, 분석적이며, '탐정님'이라는 호칭을 사용한다.



[규칙 1: 정보 통제]
1. **[자연스러운 호칭]** 'A1', 'C1' 같은 코드명 대신 실제 단서 이름으로 불러라.
2. **[태그 숨김]** 답변에 '[상세 내용]', '[AI 분석 힌트]' 같은 태그를 절대 포함하지 마라.
3. **[스포일러/오류 방지]** Context에 없는 내용은 모른다고 하고, 플레이어의 잘못된 전제(예: B4=편지)는 명확히 반박하라.

[규칙 2: 간결성 (최종 HINT Logic)]
1. [사실 질문]: 플레이어의 질문이 단순 확인(예: "이게 뭐야?")일 경우, Context의 '[상세 내용]'을 바탕으로 **'사실(Fact)'만 1-2 문장으로 짧게** 답하라.

2. [분석/힌트 질문]: 플레이어의 질문이 구체적인 분석/의미(예: "이게 왜 중요해?", "모순 아냐?")이거나, **'비밀번호'나 '잠금장치'와 관련된 질문**일 경우, Context의 '[AI 분석 힌트]'와 '[연관 단서]'를 바탕으로 **2-3 문장 이내로 간결하게** 분석/답변하라.

3. **[HINT 강제]**: 만약 [분석/힌트 질문]이 들어왔고 Context에 C1 또는 C2 단서가 있다면, **정답 숫자(1887, 3605)는 절대로 말하지 말고**, Context에 있는 '생년월일'이나 '모델 번호' 같은 **핵심 정보**를 이용하여 힌트만 유도하라.

[규칙 3: 정답 금지]
절대 '밀정이다', '아니다'와 같은 사건의 최종 결론(정답)을 직접 말하지 마라.
항상 '...라는 주장을 뒷받침합니다', '...라는 의혹이 있습니다'와 같이
플레이어가 스스로 생각하도록 '방향성'만 제시하라.
"""

# 4. RAG 체인 컴포넌트 준비
try:
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2)
    embeddings = OpenAIEmbeddings(model	=EMBEDDING_MODEL)

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
    print(f"🔍 RAW REQUEST 객체: {request}")  # 추가
    print(f"🔍 Request dict: {request.dict()}")  # 추가
    print(f"질문: {request.question}")
    print(f"받은 단서 목록: {request.acquired_clue_list}")
    print(f"채팅 히스토리: {request.chat_history}")
    
    
    try:
        if not request.acquired_clue_list:
            metadata_filter = {"clue_id": "NONE"}
        else:
            metadata_filter = {"clue_id": {"$in": request.acquired_clue_list}}
        
        print(f"메타데이터 필터: {metadata_filter}")
        
        retriever = vectorstore.as_retriever(search_kwargs={'filter': metadata_filter})
        
        # 🔧 수정: invoke() 메서드 사용
        test_docs = retriever.invoke(request.question)
        print(f"검색된 문서 수: {len(test_docs)}")
        for i, doc in enumerate(test_docs):
            print(f"문서 {i+1} - clue_id: {doc.metadata.get('clue_id', 'N/A')}")
            print(f"내용 미리보기: {doc.page_content[:100]}...")
        
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI 서버 내부 오류 발생")
# 3-2. 채점 전용 시스템 프롬프트 
scoring_prompt_template = """
너는 '도와줘! 왓슨!' 게임의 서술형 답안 채점관이다.
너의 임무는 플레이어의 '작성 이유'를 읽고, 아래 **두 가지 체크리스트**를 통과했는지 확인하여 점수를 매기는 것이다.

[상황 설명]
플레이어는 '{conclusion}'라는 결론을 내렸지만, 그와 상반되는 증거(반대 증거)를 수집한 상태다.
플레이어가 이 모순을 어떻게 설명하는지 평가해야 한다.

[채점 체크리스트 (각 10점, 총 20점)]
1. **[반대 증거 인지] (10점)**: 
   - 텍스트에서 '반대 증거'(예: 결론이 밀정이면 '김원봉 메모' 등)의 존재를 언급하거나 인지하고 있는가?
   - (예: "비록 김원봉의 메모가 있지만...", "B2 증거가 있긴 해도...")
   - 언급했다면 10점, 아니면 0점.

2. **[모순 해명] (10점)**:
   - 그 반대 증거가 왜 자신의 결론을 방해하지 않는지 '가설'을 세워 설명했는가?
   - (예: "...그건 위조된 것이다", "...신뢰를 얻기 위한 연기였다", "...이중간첩 활동의 일부다")
   - 설명했다면 10점, 아니면 0점.

[플레이어 답안]
- 결론: {conclusion}
- 선택한 증거: {selected_evidence}
- 작성 이유: "{reasoning}"

[출력 형식]
두 점수를 합산하여 오직 숫자 **20**, **10**, **0** 중 하나만 출력하라. (다른 말 금지)
"""
# 14-2. (신규) 채점 API 엔드포인트 (하이브리드: Python 객관식 + AI 주관식)
@app.post("/api/ai/score", response_model=ScoreResponse)
async def score_report(request: ScoreRequest):
    print(f"\n--- 채점 요청 수신 ---")
    try:
        # [정답지]
        A_EVIDENCE = {"A1", "A2", "A3", "A4", "A5"}
        B_EVIDENCE = {"B1", "B2", "B3", "B4", "B5"}
        CORE_EVIDENCE = A_EVIDENCE.union(B_EVIDENCE)

        # --- 1. Python: 객관식 채점 (80점 만점) ---
        # (1) 증거 수집도 (40점)
        collected_count = len(set(request.total_collected_ids).intersection(CORE_EVIDENCE))
        score1 = collected_count * 4

        # (2) 논리적 정합성 (40점) - 결론과 증거 ID가 매칭되는가?
        selected_set = set(request.selected_evidence_ids)
        score2 = 0
        if request.conclusion == "miljeong" and selected_set.issubset(A_EVIDENCE):
            score2 = 40
        elif request.conclusion == "anti_miljeong" and selected_set.issubset(B_EVIDENCE):
            score2 = 40

        # --- 2. AI: 주관식 서술형 채점 (20점 만점) ---
        score3_ai = 0
        if request.reasoning_text.strip(): # 작성한 이유가 있을 때만 AI 호출
            print("AI 서술형 채점 시작...")
            formatted_prompt = scoring_prompt_template.format(
                conclusion=request.conclusion,
                selected_evidence=f"{request.selected_evidence_ids}",
                reasoning=request.reasoning_text
            )
            response = await llm.ainvoke([HumanMessage(content=formatted_prompt)])
            
            # AI 답변에서 숫자만 추출 (예: "점수는 20점입니다" -> 20)
            import re
            numbers = re.findall(r'\d+', response.content)
            if numbers:
                score3_ai = int(numbers[0])
                # 안전장치: 0, 10, 20점 이외의 점수가 나오면 가장 가까운 값으로 조정하거나 그대로 둠
                if score3_ai > 20: score3_ai = 20
            
            print(f"AI 서술형 점수: {score3_ai}")

        # --- 3. 최종 합산 ---
        total_score = score1 + score2 + score3_ai
        
        grade = "D"
        if total_score >= 95: grade = "S"
        elif total_score >= 85: grade = "A"
        elif total_score >= 70: grade = "B"
        elif total_score >= 60: grade = "C"
        
        feedback = f"증거수집({score1}/40), 논리성({score2}/40), 서술평가({score3_ai}/20)"
        
        return ScoreResponse(total_score=total_score, grade=grade, feedback=feedback)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Scoring Error")
# 15. 서버 실행 설정
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

# 기존 코드 끝에 추가
@app.get("/health")
def health_check():
    return {"status": "ok"}