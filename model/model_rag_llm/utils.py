import os
import re
import json
import time
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
from unsloth import FastModel
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# 환경 설정
load_dotenv()
vdb_user = os.getenv("VDB_USER")
vdb_password = os.getenv("VDB_PASSWORD")
vdb_host = os.getenv("VDB_HOST")
vdb_port = int(os.getenv("VDB_PORT", "5432"))
vdb_name = os.getenv("VDB_NAME")
vdb_engine = create_engine(f"postgresql+psycopg2://{vdb_user}:{vdb_password}@{vdb_host}:{vdb_port}/{vdb_name}")

# 모델 경로: /Users/hanseungheon/Desktop/Google_ML_NIPA/Integration/model_server/model/model_rag_llm/gemma-3-reviews-model
# 모델 경로 설정 (환경 변수에서 가져오거나 기본값 사용)
MODEL_PATH = "/Users/hanseungheon/Desktop/Google_ML_NIPA/Integration/model_server/model/model_rag_llm/gemma-3-reviews-model"

# 모델과 토크나이저를 저장할 전역 변수
MODEL = None
TOKENIZER = None

def vector_db_conn():
    """벡터 데이터베이스 연결 함수"""
    try:
        conn = psycopg2.connect(
            host=vdb_host,
            port=vdb_port,
            dbname=vdb_name,
            user=vdb_user,
            password=vdb_password
        )
        return conn
    except Exception as e:
        print(f"🚫 [데이터베이스 연결 오류] {str(e)}")
        raise

def load_finetuned_model(model_path):
    """파인튜닝된 모델 로드 함수"""
    print(f"📂 [모델 로딩] 파인튜닝된 Gemma-3 모델 로딩 중... (경로: {model_path})")
    try:
        # 모델 및 토크나이저 로드
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=True,
            load_in_8bit=False,
        )
        # 채팅 템플릿 설정
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3",
        )
        print("✅ [모델 로딩] 모델 로딩 완료!")
        return model, tokenizer
    except Exception as e:
        print(f"🚫 [모델 로딩 오류] {str(e)}")
        raise

def initialize_model(model_path=None, force_reload=False):
    """모델 초기화 함수 - 캐싱 지원"""
    global MODEL, TOKENIZER
    
    # 모델 경로가 명시적으로 제공되지 않은 경우 전역 변수 사용
    if model_path is None:
        model_path = MODEL_PATH
        
    if MODEL is None or force_reload:
        MODEL, TOKENIZER = load_finetuned_model(model_path)

def clean_review_line(review_line):
    """리뷰 라인에서 앞에 붙은 '-' 기호 등을 제거하는 함수"""
    if review_line is None:
        return None
    # 리뷰 라인의 앞부분에 있는 '-' 기호와 공백 제거
    return re.sub(r'^-\s*', '', review_line.strip())

def process_reviews(reviews):
    """리뷰 전처리 및 형식화 함수"""
    # 빈 리뷰나 의미 없는 리뷰 필터링
    filtered_reviews = [r for r in reviews if r and r.strip() and r != "(리뷰 없음)"]
    
    # 리뷰가 많을 경우 가장 정보량이 많은 리뷰만 선택 (긴 리뷰 우선)
    if len(filtered_reviews) > 7:
        filtered_reviews = sorted(filtered_reviews, key=len, reverse=True)[:7]
    
    # 리뷰가 비어있는 경우 기본값 추가
    if not filtered_reviews:
        filtered_reviews = ["(리뷰 없음)"]
    
    return filtered_reviews

def parse_llm_response(response_text):
    """LLM 응답을 일관된 방식으로 처리하는 함수"""
    if not response_text or not response_text.strip():
        return {"review_1": None, "review_2": None, "review_3": None}
    
    # 리뷰 줄 추출 (줄바꿈으로 구분)
    lines = response_text.split('\n')
    
    # 모든 줄에서 '- ' 접두어 제거 및 빈 줄 제거
    lines = [re.sub(r'^-\s*', '', line.strip()) for line in lines if line.strip()]
    
    # 단일 줄이 여러 문장을 포함하는 경우 처리
    if len(lines) == 1:
        sentences = re.split(r'(?<=[.!?~])\s+', lines[0].strip())
        lines = [s.strip() for s in sentences if s.strip()]
    
    # 정확히 3개의 리뷰 보장
    while len(lines) < 3:
        lines.append(None)
    
    # 최대 3개의 리뷰만 사용
    lines = lines[:3]
    
    return {
        "review_1": lines[0],
        "review_2": lines[1],
        "review_3": lines[2]
    }

def get_review_prompt_template():
    """프롬프트 템플릿 관리"""
    return """
    ## Persona
    당신은 '매장명', '매장리뷰'를 바탕으로 매장에 대한 정보를 3줄(100자 이내)로 답변하는 3명의 리뷰어입니다.

    ## Instruction
    - 모든 문장은 '한국어'로 작성하고, 3줄(100자 이내)의 완결형 문장으로 작성합니다.
    - 반드시 '방문자'의 시점에서 직접 경험한 것처럼 작성하세요.
    - 각 줄은 마치 실제 사람이 말하듯 자연스럽고 생동감 있는 톤으로 답변합니다.
    - 오로지 '매장리뷰'에 있는 정보만을 활용해서 답변합니다.
    - 답변은 오로지 매장 리뷰 3줄만 제공해 주세요. 매장 리뷰를 제외한 이외의 답변은 제거합니다.

    ## Information
    - **매장명**: {store_name}
    - **매장리뷰**:
    {formatted_reviews}

    ## Example
    USER: 매장 정보를 3줄로 제공해 주세요
    ASSISTANT: 
    - 너무 좋았어요! 다음에 친구들이랑 한 번 더 오려구요!
    - 날씨 좋은 날에 처음 가봤는데, 대박 좋았어요~!
    - 매장이 깔끔해서 좋고, 사장님이 엄청 친절해요~
    """

def generate_with_finetuned_model(system_prompt, user_prompt, max_retries=2):
    """파인튜닝된 모델로 추론 수행하는 함수 - 재시도 로직 추가"""
    # 모델이 로드되었는지 확인
    if MODEL is None or TOKENIZER is None:
        raise ValueError("모델이 아직 초기화되지 않았습니다. initialize_model()을 먼저 호출하세요.")
    
    retries = 0
    while retries <= max_retries:
        try:
            # 메시지 포맷팅
            messages = [{
                "role": "user",
                "content": system_prompt + "\nUSER: " + user_prompt
            }]
            
            # 챗 템플릿 적용
            text = TOKENIZER.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            
            # 추론 수행
            outputs = MODEL.generate(
                **TOKENIZER([text], return_tensors="pt").to("cuda"),
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
            )
            
            # 결과 디코딩 및 출력
            decoded_output = TOKENIZER.batch_decode(outputs)[0]
            
            # MODEL 응답 부분 추출
            if "<start_of_turn>model\n" in decoded_output:
                response = decoded_output.split("<start_of_turn>model\n")[1].split("<end_of_turn>")[0]
                return response.strip()
            else:
                print(f"⚠️ [응답 형식 오류] 시도 {retries+1}/{max_retries+1}: 응답 추출 실패")
                retries += 1
                # 마지막 시도였다면 전체 출력 반환
                if retries > max_retries:
                    print("⚠️ [경고] 응답 추출 실패, 전체 출력 반환")
                    return decoded_output
        except Exception as e:
            print(f"⚠️ [추론 오류] 시도 {retries+1}/{max_retries+1}: {str(e)}")
            retries += 1
            if retries > max_retries:
                raise

def process_single_store(keyword, store_info, conn):
    """단일 매장 처리 함수"""
    store_id, store_name, confidence = store_info
    start_time = time.time()
    
    print(f"\n📌 [진행 중] 매장 처리 시작 - '{store_name}' (ID: {store_id}, 키워드: '{keyword}', 신뢰도: {confidence}%)")
    
    try:
        cur = conn.cursor()
        
        # 해당 store_id에 대한 리뷰 조회
        cur.execute("SELECT review_docs FROM stores WHERE store_id = %s", (store_id,))
        review_rows = cur.fetchall()
        cur.close()
        
        reviews = [row[0] for row in review_rows if row[0]]
        
        # 가져온 리뷰 출력
        print("📜 [가져온 리뷰 목록]")
        if reviews:
            for i, review in enumerate(reviews[:3], 1):  # 첫 3개만 출력
                print(f"  {i}. {review}")
            if len(reviews) > 3:
                print(f"  ... 외 {len(reviews)-3}개")
        else:
            print("  (리뷰 없음)")
        
        # 리뷰 처리 및 형식화
        processed_reviews = process_reviews(reviews)
        formatted_reviews = '\n'.join([f" -- {r}" for r in processed_reviews])
        print(f"📝 [프롬프트 생성] '{store_name}' 리뷰 형식화 완료")
        
        # LLM 프롬프트 구성
        prompt_template = get_review_prompt_template()
        system_prompt = prompt_template.format(
            store_name=store_name,
            formatted_reviews=formatted_reviews
        )
        
        # 모델을 사용한 추론
        user_prompt = "매장 정보를 3줄로 제공해 주세요"
        response_raw = generate_with_finetuned_model(system_prompt, user_prompt)
        
        # 결과 처리
        review_parts = parse_llm_response(response_raw)
        
        print(f"✅ [LLM 응답 완료] '{store_name}' 리뷰 요약 생성 완료")
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"⏱️ [처리 시간] {processing_time:.2f}초")
        
        return {
            "keyword": keyword,
            "store_id": store_id,
            "store_name": store_name,
            "confidence": confidence,
            "review_1": review_parts["review_1"],
            "review_2": review_parts["review_2"],
            "review_3": review_parts["review_3"],
            "processing_time": processing_time
        }
        
    except Exception as e:
        print(f"⚠️ [매장 처리 오류] '{store_name}' 처리 중 오류 발생: {str(e)}")
        return {
            "keyword": keyword,
            "store_id": store_id,
            "store_name": store_name,
            "confidence": confidence,
            "review_1": None,
            "review_2": None,
            "review_3": None,
            "error": str(e)
        }

def process_stores_in_batches(stt_results, batch_size=5):
    """매장을 배치로 처리하여 효율성 향상"""
    all_summaries = []
    items = list(stt_results.items())
    
    conn = vector_db_conn()  # 연결 한 번만 생성
    
    for i in range(0, len(items), batch_size):
        batch = dict(items[i:i+batch_size])
        print(f"🔄 [배치 처리] 배치 {i//batch_size + 1}: {i+1}~{min(i+batch_size, len(items))}/{len(items)}")
        
        batch_results = []
        for keyword, store_info in batch.items():
            result = process_single_store(keyword, store_info, conn)
            batch_results.append(result)
        
        all_summaries.extend(batch_results)
    
    conn.close()  # 작업 완료 후 연결 닫기
    return all_summaries

def log_performance_metrics(summaries):
    """성능 지표 로깅"""
    if not summaries:
        return
    
    # 오류가 없는 항목만 필터링
    successful_items = [s for s in summaries if "error" not in s]
    
    if not successful_items:
        print("🕒 [성능 지표] 성공적으로 처리된 항목이 없습니다.")
        return
    
    # 처리 시간이 있는 항목만 필터링
    timed_items = [s for s in successful_items if "processing_time" in s]
    
    if timed_items:
        total_time = sum(s["processing_time"] for s in timed_items)
        avg_time = total_time / len(timed_items)
        print(f"🕒 [성능 지표] 총 처리된 매장 수: {len(successful_items)}")
        print(f"🕒 [성능 지표] 총 소요시간: {total_time:.2f}초")
        print(f"🕒 [성능 지표] 매장당 평균 처리시간: {avg_time:.2f}초")

def generate_store_summaries(stt_results: dict, model_path=None, use_batches=True, batch_size=5):
    """
    리뷰 요약 생성 함수
    
    Args:
        stt_results (dict): STT에서 전달받은 매장 정보 딕셔너리
            {
                '검색어': [store_id, store_name, confidence],
                ...
            }
        model_path (str, optional): 모델 경로 (None인 경우 MODEL_PATH 환경변수 또는 기본값 사용)
        use_batches (bool): 배치 처리 사용 여부
        batch_size (int): 배치 크기
    
    Returns:
        dict: 처리 결과 및 생성된 리뷰 요약
    """
    # 전체 시작 시간 기록
    overall_start_time = time.time()
    
    try:
        # 모델 초기화 (지정된 경로 또는 기본 경로 사용)
        initialize_model(model_path)
        
        print(f"\n🚀 [처리 시작] 총 {len(stt_results)}개 매장 처리 시작")
        
        # 배치 처리 또는 기존 방식 선택
        if use_batches:
            summaries = process_stores_in_batches(stt_results, batch_size)
        else:
            conn = vector_db_conn()
            summaries = []
            
            for keyword, store_info in stt_results.items():
                result = process_single_store(keyword, store_info, conn)
                summaries.append(result)
                
            conn.close()
            
        # 전체 종료 시간 기록
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        # 성능 지표 로깅
        print(f"\n⏱️ [전체 성능] 총 소요시간: {overall_duration:.2f}초")
        log_performance_metrics(summaries)
        
        # 최종 결과 로그 확인
        print("\n🎉 [전체 완료] 모든 매장 요약 작업이 성공적으로 완료되었습니다.")
        print(f"\n🧾 [요약 결과 데이터]:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}")

        # 오류 항목이 있는지 확인
        error_items = [s for s in summaries if "error" in s]
        success_count = len(summaries) - len(error_items)
        
        return {
            "status": "success",
            "code": 200,
            "message": f"매장 요약 완료 ({success_count}/{len(summaries)} 성공)",
            "processed": "ok",
            "total_time": overall_duration,
            "successful": success_count,
            "failed": len(error_items),
            "data": summaries
        }

    except Exception as e:
        print(f"\n❌ [오류 발생] {str(e)}")
        return {
            "status": "error",
            "code": 500,
            "message": str(e),
            "processed": "failed",
            "data": []
        }

# 사용 예시
if __name__ == "__main__":
    # 모델 경로 설정 (실제 경로로 변경 필요)
    model_path = "/path/to/gemma-3-reviews-model"
    
    # 모델 초기화
    initialize_model(model_path)
    
    # STT에서 전달받은 테스트 데이터 예시
    stt_input = {
        '마리우네': [3, '마리오네', 64],
        '성덕': [22, '강별 성수', 62],
        '고기질이': [82, '고깃바', 62],
        '레츠잇칙힌': [1, '레츠잇치킨 성수', 69],
        '성수': [2, '성수노루', 85]
    }

    # 함수 실행
    result = generate_store_summaries(stt_input, batch_size=3)
    print(result)