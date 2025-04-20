import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from service.pipeline import run_pipeline

load_dotenv()
DJANGO_RESULT_API = os.getenv("DJANGO_RESULT_API")

app = FastAPI()

class AnalyzeRequest(BaseModel):
    url: str

    class Config:
        extra = "ignore"  # ✅ 추가 필드 무시

@app.post("/start-analysis/")
def start_analysis(req: AnalyzeRequest):
    result = run_pipeline(req.url)

    payload = {
        "url": req.url,
        "result": result
    }

    try:
        res = requests.post(DJANGO_RESULT_API, json=payload)
        res.raise_for_status()
        return {"status": "ok", "forwarded": True}
    except Exception as e:
        return {"status": "error", "message": str(e)}
