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
# 3. AI '왓슨' 시스템 프롬프트 (Context 내용 기반 판단)
system_prompt = """
너는 임시정부 소속의 냉철한 수사 조수 AI '왓슨'이다.
너의 말투는 항상 냉철하고, 분석적이며, '탐정님'이라는 호칭을 사용한다.
같은걸 물어봐도 아예똑같이 답하진말고
150-200자이내로 답변해줘
[인벤토리 단서 목록 (번호 매핑)]
1번 단서: A1 (특무대 출장 증명서)
2번 단서: A2 (경무국 비밀 급여 영수증)
3번 단서: A3 (황옥의 수첩 페이지)
4번 단서: A4 (조회성 통화 기록)
5번 단서: A5 (경무국 기밀 보고서 초안)
6번 단서: B1 (형제 황직연의 사진)
7번 단서: B2 (김원봉의 친필 메모)
8번 단서: B3 (폭탄 설계 스케치)
9번 단서: B4 (위조 여권 및 변장 도구)
10번 단서: B5 (황옥의 고백 완전본)
- 사용자가 '1번 단서'나 '첫 번째 증거'라고 물어보면 위 목록을 참고하여 답변하라.

[규칙 1: 정보 통제]
1. 너는 **오직 '제공된 Context(확보된 단서 내용)'**에 대해서만 말할 수 있다.
2. **[코드 ID 금지]** 'A1', 'B4' 같은 코드명으로 절대 말하지 않고 대신 '특무대 증명서', '변장 도구' 등 실제 이름으로 불러라.
3. **[태그 숨김]** 답변에 '[상세 내용]', '[AI 분석 힌트]' 같은 태그를 절대 포함하지 마라.
4. **[스포일러 방지]** Context에 없는 내용은 모른다고 답하라.
[규칙 2:방 잠금장치 해금 비밀번호]
비밀번호라고 물어봐도 해당하는 방에 대해서만 알려주고 다른 방은 언급하지마
숫자를 절대 언급하지마

[규칙 3: 일반 대화]
위의 힌트 질문이 아니라면, 질문에 대해 간결하게(2-3문장) 답하거나 분석하라.
사건과 무관한 질문(날씨, 농담)에는 짧게 답하고 수사에 집중 할 수 있도록 격려해줘.
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
    print("\n================= /api/ai/score 요청 수신 =================")
    print(f"[결론 conclusion] {request.conclusion}")
    print(f"[선택 증거 selected_evidence_ids] {request.selected_evidence_ids}")
    print(f"[전체 수집 단서 total_collected_ids] {request.total_collected_ids}")
    print(f"[작성 이유 reasoning_text] {request.reasoning_text}")
    
    try:
        # [정답지]
        A_EVIDENCE = {"A1", "A2", "A3", "A4", "A5"}
        B_EVIDENCE = {"B1", "B2", "B3", "B4", "B5"}
        CORE_EVIDENCE = A_EVIDENCE.union(B_EVIDENCE)

        # 1. 증거 수집도 (40점) - Python 계산
        collected_count = len(set(request.total_collected_ids).intersection(CORE_EVIDENCE))
        score1 = collected_count * 4
        
        print(f"[계산: 증거 수집도 score1] {score1}점 (수집 개수: {collected_count})")

        # 2. 논리적 추론 (40점) - Python 계산(사용x)
        selected_set = set(request.selected_evidence_ids)
        score2 = 0
        if request.conclusion == "miljeong" and selected_set.issubset(A_EVIDENCE):
            score2 = 40
        elif request.conclusion == "anti_miljeong" and selected_set.issubset(B_EVIDENCE):
            score2 = 40
            
        print(f"[계산: 논리적 추론 score2] {score2}점 (선택 증거={selected_set})")

            
        # 3. 서술형 평가 (20점) + AI 종합 코멘트
        score3 = 0
        ai_comment = ""
        
        if request.reasoning_text.strip():
            # [핵심 수정] score1, score2 변수를 프롬프트에 { }로 넣어서 AI에게 알려줍니다.
            scoring_prompt = f"""
            너는 '도와줘! 왓슨!' 게임의 유연하고 너그러운 채점관이다.
            플레이어의 글을 읽고 **0점에서 60점 사이의 점수**를 부여하라.

            [시스템 계산 점수 현황]
            - 증거 수집도: {score1}점 / 40점

            [플레이어 답안]
            - 결론: {request.conclusion}
            - 작성 이유: "{request.reasoning_text}"
            
            [채점 가이드라인 (참고용)]
            이 기준을 바탕으로 **1점 단위로 자유롭게** 점수를 매기시오.
            
            - **최상위권 (50 ~ 60점):** 주장이 명확하고 핵심 증거(A1, A2 등)를 잘 언급함. 문장이 짧아도 핵심을 찔렀다면 60점 만점 부여.
            - **중위권 (30 ~ 49점):** 결론은 맞지만 근거가 조금 빈약하거나, "그냥 범인 같음" 정도로 뭉뚱그려 설명함.
            - **하위권 (0 ~ 29점):** 논리가 아예 없거나, 사건과 무관한 이야기를 함.

            [피드백 지침]
            - 점수에 맞춰 왓슨의 말투로 자연스러운 피드백을 작성할 것.
            - 50점 이상이면 극찬하고, 그 미만이면 부드럽게 조언할 것.

            [출력 형식 (JSON)]
            {{
                "score": 점수(0~60 사이의 정수 예: 55, 48),
                "comment": "피드백 한 문장,플레이에 대한 감사"
            }}
            """
            
            print("------ AI 채점 프롬프트 ------")
            print(scoring_prompt)
            print("------------------------------")
            
            # AI 호출
            try:
                resp = await llm.ainvoke([HumanMessage(content=scoring_prompt)])
                content = resp.content.strip()
                print(f"[AI 응답 원문]\n{content}")
                
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
                    
                    print(f"[AI 서술형 점수 score3] {score3}")
                    print(f"[AI 피드백] {ai_comment}")
                else:
                    score3 = 0
                    print("[ERROR] JSON을 찾지 못함")
                    ai_comment = "피드백 생성 실패."
                    
            except Exception as e:
                print(f"AI Parsing Error: {e}")
                score3 = 0
                ai_comment = "AI 평가를 진행할 수 없습니다."
        
        else:
            print("[ERROR] JSON을 찾지 못함")
            ai_comment = "작성된 이유가 없어 서술형 점수를 받을 수 없습니다."

        # 최종 합산
        total_score = score1 + score3
        print(f"[최종 점수 total_score] {total_score}")
              
        # 등급 계산
        grade = "D"
        if total_score >= 95: grade = "S"
        elif total_score >= 85: grade = "A"
        elif total_score >= 70: grade = "B"
        elif total_score >= 60: grade = "C"
        
        print(f"[등급 grade] {grade}")
        print("================= /api/ai/score 종료 =================\n")
        # 피드백 문자열 생성
        feedback = f"증거 수집({score1}/40), 서술 평가({score3}/60): {ai_comment}"
        
        return ScoreResponse(total_score=total_score, grade=grade, feedback=feedback)

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Scoring Error")
# 9. 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

