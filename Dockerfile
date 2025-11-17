# Python 베이스 이미지 설정
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# Ky 디렉토리와 server 디렉토리의 파일을 컨테이너로 복사
COPY ./Ky /app/Ky
COPY ./server /app/server
COPY ./input_data /app/input_data

# Ky/requirements.txt 를 사용하여 의존성 설치
RUN pip install --no-cache-dir -r /app/Ky/requirements.txt

# 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
