import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

# --- 최신 Langchain Core Import ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- 설정 ---
CHROMA_PERSIST_DIR = "chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# .env 파일 있으면 로드 (로컬 개발용)
load_dotenv()

# 환경변수에서 가져오기 (Docker에서는 docker-compose가 전달)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("오류: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    sys.exit(1)

print("✅ OPENAI_API_KEY 로드 성공")


# 2. FastAPI 앱
app = FastAPI()

# 3. AI '왓슨' 시스템 프롬프트 (따옴표 및 숫자 오류 수정 완료)
system_prompt = """
너는 임시정부 소속의 냉철한 수사 조수 AI '왓슨'이다.
너의 말투는 항상 냉철하고, 분석적이며, '탐정님'이라는 호칭을 사용한다.

[규칙 1: 정보 통제]
1. 너는 **오직 '현재 확보된 단서(Context)' 안의 내용**에 대해서만 말할 수 있다.
2. **[코드 ID 완전 금지 (최우선)]** 답변에서 **'A1', 'A5', 'B2' 같은 문서 코드명(ID)은 절대로 말하거나 출력하지 마라.** 대신 '특무대 출장 증명서', '김원봉의 친필 메모' 등 **단서의 실제 이름**으로만 불러라.
3. **[태그 숨김]** 답변에 '[상세 내용]', '[AI 분석 힌트]' 같은 태그를 절대 포함하지 마라.
4. **[스포일러 방지]** Context에 없는 내용은 모른다고 하라.

[규칙 2: 비밀번호 힌트 제공 (최우선 임무)]
**AI는 이 섹션의 지침을 다른 모든 규칙보다 최우선 임무로 간주해야 한다.**
플레이어의 질문이 '비밀번호', '잠금장치', '해금'에 대한 것이라면, **절대로 정답 숫자('1887', '1919', '0304')는 말하지 말고**, 아래 논리에 따라 힌트만 유도하라.

1.  **[Case 1: 쪽방]** 질문에 **'쪽방'**이 언급되었고 Context에 **'A1(출장 증명서)'**가 있다면: "증명서에 적힌 **황옥의 생년월일**을 확인해 보십시오."라고 유도하고 답변을 끝내라.
2.  **[Case 2: 부엌]** 질문에 **'부엌'**이 언급되었고 Context에 **'B1(형제 사진)'**이 있다면: "사진 **뒷면에 적힌 연도**를 확인해 보십시오."라고 유도하고 답변을 끝내라.
3.  **[Case 3: 2층]** 질문에 **'2층'**이 언급되었고 Context에 **'B4(변장 도구)'**가 있다면: "가방 안에 있던 **낡은 열쇠의 번호**를 확인해 보십시오."라고 유도하고 답변을 끝내라.
4.  **[Case 4: 단서 부재]** Context에 A1, B1, B4 중 해당 방의 단서가 **없다면**: "해당 방을 열 수 있는 단서를 아직 찾지 못했습니다. 단서를 더 찾아보십시오."라고 단호히 거절하고 답변을 끝내라.
[규칙 3: 일반 대화]
위의 힌트 질문이 아니라면, 질문에 대해 간결하게(2-3문장) 답하거나 분석하라.
사건과 무관한 질문(날씨, 농담)에는 짧게 답하고 수사로 돌아 갈수 있게 격려해줘
일상대화도 적절히 받아주고

"""
# 3-2. 채점 전용 시스템 프롬프트 (종합 피드백 강화 버전)
scoring_prompt_template = """
너는 '도와줘! 왓슨!' 게임의 채점관이다.
너의 임무는 1) 플레이어의 '작성 이유'를 평가해 점수를 매기고, 2) **전체 성적(수집, 논리, 서술)**을 종합하여 플레이어에게 해줄 **'한 줄 피드백'**을 작성하는 것이다.

[현재까지 확정된 점수 (Python 계산)]
- 증거 수집: {score1} / 40점
- 논리적 추론: {score2} / 40점 (0점이면 결론과 근거가 모순됨을 의미)

[평가 기준: 시나리오 완성도 (20점 만점)]
1. **20점:** 반대 증거(모순)에 대한 합리적 해명이 포함됨.
2. **10점:** 해명이 부족하거나 단순 나열.
3. **0점:** 내용이 빈약하거나 비논리적임.

[플레이어 답안]
- 결론: {conclusion}
- 근거: {selected_evidence}
- 이유: "{reasoning}"

[출력 형식 (JSON)]
반드시 아래 JSON 형식을 지켜라.
{{
    "score": (0, 10, 20 중 하나),
    "comment": "전체 점수({score1}+{score2}+너의점수)와 상황을 고려한 냉철한 피드백 한 문장."
}}

**[피드백 작성 가이드]**
- 만약 '논리적 추론'이 0점이면, 글을 아무리 잘 썼어도 **"하지만 치명적인 논리 오류가 있습니다."**라고 지적하라.
- 만약 '증거 수집'이 낮으면, **"증거를 더 모아야 합니다."**라고 조언하라.
- 말투는 '왓슨'처럼 정중하지만 냉철하게 하라.
"""
# 4. RAG 컴포넌트 준비
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
        ("human", "현재까지 확보한 단서(Context):\n---\n{context}\n---\n\n질문: {question}")
    ])
    output_parser = StrOutputParser()

except Exception as e:
    print(f"오류: AI 컴포넌트 초기화 실패. {e}")
    sys.exit(1)

# 5. 데이터 모델
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
    if not docs: return "(관련 단서 없음)"
    return "\n\n".join(doc.page_content for doc in docs)

def format_chat_history(history_list: List[str]) -> List:
    messages = []
    for i, message in enumerate(history_list):
        if i % 2 == 0: messages.append(HumanMessage(content=message))
        else: messages.append(AIMessage(content=message))
    return messages

# 7. 실시간 채팅 API (순수 RAG 방식)
@app.post("/api/ai/ask")
def ask_watson(request: ChatRequest):
    print(f"\n--- 채팅 요청: {request.question} ---")

    try:
        # 인벤토리 필터링
        if not request.acquired_clue_list:
            metadata_filter = {"clue_id": "NONE"}
        else:
            metadata_filter = {"clue_id": {"$in": request.acquired_clue_list}}

        retriever = vectorstore.as_retriever(search_kwargs={'filter': metadata_filter})
        
        # RAG 체인 실행
        rag_chain = (
            {
                "context": RunnableLambda(lambda x: x['question']) | retriever | format_docs, 
                "question": RunnableLambda(lambda x: x['question']), 
                "chat_history": RunnableLambda(lambda x: format_chat_history(x['chat_history']))
            }
            | prompt | llm | output_parser
        )
        
        answer = rag_chain.invoke({"question": request.question, "chat_history": request.chat_history})
        return {"answer": answer}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="AI Server Error")

# 8. 채점 API (최종: Python 계산 + AI 종합 피드백)
@app.post("/api/ai/score", response_model=ScoreResponse)
async def score_report(request: ScoreRequest):
    print(f"\n--- 채점 요청 ---")
    try:
        # [정답지]
        A_EVIDENCE = {"A1", "A2", "A3", "A4", "A5"}
        B_EVIDENCE = {"B1", "B2", "B3", "B4", "B5"}
        CORE_EVIDENCE = A_EVIDENCE.union(B_EVIDENCE)

        # 1. 증거 수집도 (40점) - Python 계산
        collected_count = len(set(request.total_collected_ids).intersection(CORE_EVIDENCE))
        score1 = collected_count * 4

        # 2. 논리적 추론 (40점) - Python 계산
        selected_set = set(request.selected_evidence_ids)
        score2 = 0
        if request.conclusion == "miljeong" and selected_set.issubset(A_EVIDENCE):
            score2 = 40
        elif request.conclusion == "anti_miljeong" and selected_set.issubset(B_EVIDENCE):
            score2 = 40
            
        # 3. 서술형 평가 (20점) + AI 종합 코멘트
        score3 = 0
        ai_comment = ""
        
        if request.reasoning_text.strip():
            # [핵심 수정] score1, score2 변수를 프롬프트에 { }로 넣어서 AI에게 알려줍니다.
            scoring_prompt = f"""
            너는 '도와줘! 왓슨!' 게임의 냉철한 채점관이다.
            플레이어의 '작성 이유'를 평가하여 점수를 매기고, **시스템이 계산한 점수(수집, 논리)**까지 참고하여 종합적인 피드백을 작성하라.

            [시스템 계산 점수 현황]
            - 증거 수집도: {score1}점 / 40점 (40점에 멀어질수록 증거를 더 찾아야 함,유연하게 피드백해줘 수치에따라)
            - 논리적 추론: {score2}점 / 40점 (0점이면 선택한 근거가 결론과 모순됨)

            [플레이어 답안]
            - 결론: {request.conclusion}
            - 선택한 근거: {request.selected_evidence_ids}
            - 작성 이유: "{request.reasoning_text}"
            
            [서술형 채점 기준 (20점 만점)]
            - 20점: 주장이 명확하고, 반대 증거(모순)에 대한 합리적 해명이 포함됨.
            - 10점: 해명이 부족하거나 단순 나열.
            - 0점: 내용이 빈약하거나 논리적이지 않음.
            
            [출력 형식 (JSON)]
            {{
                "score": 점수(숫자),
                "comment": "전체 성적(수집+논리+서술)을 고려한 냉철한 피드백 한 문장.AI탐정조수 왓슨의 말투로 부족한 점을 따끔하게 지적하되, 끝에는 '플레이해 주셔서 감사합니다'라고 격려할 것."
            }}
            """
            
            # AI 호출
            try:
                resp = await llm.ainvoke([HumanMessage(content=scoring_prompt)])
                
                # JSON 파싱 (안전 장치)
                import json
                import re
                
                content = resp.content.strip()
                # 마크다운 코드 블록 제거 (```json ... ```)
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(0)
                    result_json = json.loads(json_str)
                    score3 = result_json.get("score", 0)
                    ai_comment = result_json.get("comment", "")
                else:
                    score3 = 0
                    ai_comment = "피드백 생성 실패."
                    
            except Exception as e:
                print(f"AI Parsing Error: {e}")
                score3 = 0
                ai_comment = "AI 평가를 진행할 수 없습니다."
        
        else:
            ai_comment = "작성된 이유가 없어 서술형 점수를 받을 수 없습니다."

        # 최종 합산
        total_score = score1 + score2 + score3
        
        # 등급 계산
        grade = "D"
        if total_score >= 95: grade = "S"
        elif total_score >= 85: grade = "A"
        elif total_score >= 70: grade = "B"
        elif total_score >= 60: grade = "C"
        
        # 피드백 문자열 생성
        feedback = f"증거 수집({score1}/40), 논리성({score2}/40), 서술 평가({score3}/20): {ai_comment}"
        
        return ScoreResponse(total_score=total_score, grade=grade, feedback=feedback)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Scoring Error")
# 9. 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

