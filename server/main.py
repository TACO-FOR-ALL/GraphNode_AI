from fastapi import FastAPI, BackgroundTasks, Request
from typing import Union
import time
import uuid
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


def run_ai_pipeline_task(task_id: str, data: dict):
    """
    AI 파이프라인을 실행하는 가상 함수.
    실제로는 이 함수 내에서 run_pipeline.py를 호출하거나 해당 로직을 실행해야 합니다.
    """
    print(f"[{task_id}] AI 파이프라인 시작.")
    # 입력 데이터를 파일로 저장 (run_pipeline.py가 파일을 읽는다고 가정)
    input_dir = "/app/input_data"
    os.makedirs(input_dir, exist_ok=True)
    input_file_path = os.path.join(input_dir, f"{task_id}_input.json")
    # with open(input_file_path, "w") as f:
    #     json.dump(data, f)

    # 여기서 run_pipeline.py 스크립트를 subprocess로 호출하거나,
    # 해당 모듈의 함수를 직접 임포트하여 실행합니다.
    # 예: subprocess.run(["python", "/app/Ky/src/run_pipeline.py", "--input-json", input_file_path, ...])

    time.sleep(10)  # AI 모듈이 10초간 실행된다고 가정
    print(f"[{task_id}] AI 파이프라인 완료.")
    # 완료 후 Webhook으로 결과를 보내는 로직 추가 필요


@app.post("/analysis")
async def request_analysis(request: Request, background_tasks: BackgroundTasks):
    """
    AI 분석을 요청하는 엔드포인트.
    - 요청 데이터를 받아 유효성을 검사합니다.
    - 백그라운드에서 AI 파이프라인을 실행하도록 작업을 추가합니다.
    - 즉시 작업 ID를 반환합니다.
    """
    task_id = str(uuid.uuid4())
    # body = await request.json() # 데이터 유효성 검사 및 변환 로직 추가 예정

    # background_tasks.add_task(run_ai_pipeline_task, task_id, body)

    return {"task_id": task_id, "status": "accepted"}
