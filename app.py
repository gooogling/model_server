from fastapi import FastAPI
from pydantic import BaseModel
from service.pipeline import run_pipeline
from dotenv import load_dotenv
import os
import requests

load_dotenv()
DJANGO_RESULT_API = os.getenv("DJANGO_RESULT_API")

app = FastAPI()

class AnalyzeRequest(BaseModel):
    url: str
    video_id: int

@app.post("/start-analysis/")
def start_analysis(req: AnalyzeRequest):
    # 전체 모델 파이프라인 실행
    result = run_pipeline(req.url)

    # Django 백엔드로 결과 전송
    payload = {
        "video_id": req.video_id,
        "url": req.url,
        "result": result
    }

    try:
        res = requests.post(DJANGO_RESULT_API, json=payload)
        res.raise_for_status()
        return {"status": "ok", "forwarded": True}
    except Exception as e:
        return {"status": "error", "message": str(e)}
