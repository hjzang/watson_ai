# 1. 파이썬 3.11 버전을 기반으로 시작합니다.
FROM python:3.11-slim

# 2. 캡슐 안의 작업 폴더를 '/app'으로 정합니다.
WORKDIR /app

# 3. 방금 만든 '준비물 목록' 파일을 캡슐 안으로 먼저 복사합니다.
COPY requirements.txt .

# 4. 캡슐 안에서 '준비물 목록'을 보고 pip install을 실행합니다.
RUN pip install --upgrade pip && pip install -r requirements.txt

# 5. 현재 폴더의 '모든 파일'(.py, .env, knowledge_base, chroma_db 등)을
#    캡슐 안의 /app 폴더로 복사합니다.
COPY . .

# 6. AI 서버가 8000번 포트를 사용한다고 외부에 알려줍니다.
EXPOSE 8000

# 7. 캡슐이 시작될 때 'main.py'를 실행하는 최종 명령어입니다.
#    --host 0.0.0.0 은 캡슐 밖(백엔드)에서도 접속할 수 있게 해줍니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
