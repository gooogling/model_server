from wrapper.run_ocr import run_ocr
from wrapper.run_stt import run_stt
from wrapper.run_ner import run_ner
from wrapper.run_rag import run_rag

def run_pipeline(url: str, region_code: int):
    stt_result = run_stt(url)
    ocr_result = run_ocr(url)

    ner_result = run_ner(stt_result, ocr_result, region_code=region_code)

    rag_result = run_rag(ner_result)
    return rag_result
