import os
import sys
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- 설정 ---
KNOWLEDGE_BASE_DIR = "knowledge_base"  # 단서 .txt 파일이 있는 폴더
CHROMA_PERSIST_DIR = "chroma_db"       # DB를 저장할 폴더
EMBEDDING_MODEL = "text-embedding-3-small" # OpenAI 임베딩 모델
# ---

def main():
    # 1. .env 파일에서 OPENAI_API_KEY 불러오기
    if not load_dotenv():
        print("오류: .env 파일을 찾을 수 없습니다. API 키를 설정했는지 확인하세요.")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("오류: OPENAI_API_KEY가 .env 파일에 없습니다.")
        sys.exit(1)

    print(f"AI '왓슨'의 지식 베이스 임베딩을 시작합니다...")
    print(f"대상 폴더: '{KNOWLEDGE_BASE_DIR}'")

    # 2. 임베딩 모델 준비 (텍스트를 벡터로 변환할 AI)
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        print(f"오류: 임베딩 모델 로드 실패. API 키가 유효한지 확인하세요. ({e})")
        sys.exit(1)

    # 3. ChromaDB 준비 (지속형 저장소 사용)
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )

    # 4. knowledge_base 폴더에서 .txt 파일 읽기
    docs_to_add = []

    # 이미 DB에 저장된 파일 목록 가져오기 (중복 방지)
    try:
        existing_files = vectorstore.get(include=[])['ids']
    except Exception:
        existing_files = []

    print(f"이미 처리된 파일: {len(existing_files)}개")

    # 5. 새 파일 또는 변경된 파일 찾기
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if not filename.endswith(".txt"):
            continue # .txt 파일이 아니면 건너뛰기

        if filename in existing_files:
            # 이미 DB에 있는 파일이면 건너뛰기
            continue

        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)

        try:
            # 6. 텍스트 내용 읽기
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 7. 메타데이터(꼬리표) 생성 (핵심!)
            # 파일 이름 (예: "A1_특무대_증명서.txt")에서 "A1"만 추출
            clue_id = filename.split('_')[0]

            metadata = {"clue_id": clue_id}

            # 8. 나중에 한 번에 추가하기 위해 리스트에 저장
            docs_to_add.append({
                "text": content,
                "metadata": metadata,
                "id": filename  # 각 문서를 파일 이름으로 식별 (중복 방지)
            })
            print(f"  > '{filename}' 파일 추가 준비 완료 (clue_id: {clue_id})")

        except Exception as e:
            print(f"  > 오류: '{filename}' 파일 처리 중 문제 발생 ({e})")


    # 9. ChromaDB에 새로운 문서들 '주입'하기
    if docs_to_add:
        print(f"\n새로운 단서 {len(docs_to_add)}개를 DB에 추가합니다...")
        try:
            vectorstore.add_texts(
                texts=[doc["text"] for doc in docs_to_add],
                metadatas=[doc["metadata"] for doc in docs_to_add],
                ids=[doc["id"] for doc in docs_to_add]
            )

            # 10. 변경 사항을 디스크에 영구 저장 (필수)
            vectorstore.persist()
            print("지식 베이스 구축 완료! 'chroma_db' 폴더가 생성/업데이트 되었습니다.")

        except Exception as e:
            print(f"오류: DB에 저장하는 중 문제 발생. (API 키 결제 문제일 수 있습니다: {e})")
    else:
        print("\n이미 모든 단서가 최신 상태입니다. 추가할 파일이 없습니다.")

# 이 스크립트 파일을 직접 실행할 때만 main() 함수 실행
if __name__ == "__main__":
    main()
