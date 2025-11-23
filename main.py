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

# 3. AI '왓슨'의 시스템 프롬프트 
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
