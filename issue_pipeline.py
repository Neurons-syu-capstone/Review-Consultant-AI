"""
LLM 기반 낮은 평점 리뷰 이슈 탐지 파이프라인
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전체 흐름:
  [1단계] 낮은 평점 필터링     : rating ≤ 2 리뷰만 추출
  [2단계] 대표 리뷰 추출       : TF-IDF + K-Means 클러스터링 → 클러스터별 대표 리뷰
  [3단계] LLM 이슈 분석        : Ollama 로컬 LLM → 속성별 이슈 + 심각도 + 개선 제안
  [4단계] 결과 저장            : issue_report.json

※ 사전 조건:
   1. Ollama 설치: https://ollama.com/download
   2. 모델 다운로드 (RTX 4070 12GB 권장):
      ollama pull mistral-nemo:12b
   3. Ollama 서버 실행:
      ollama serve
   4. 패키지 설치:
      pip install requests scikit-learn pandas tqdm pyarrow
"""

import json
import os
import re
import warnings
import threading
warnings.filterwarnings("ignore")

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 설정값 (필요에 따라 수정)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT_FILE    = "shoes_data/shoes_reviews_cleaned.parquet"
JSON_FILE     = "shoes_data/shoes_reviews_cleaned.json"
OUTPUT_FILE   = "shoes_data/issue_report.json"

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "mistral-nemo:12b"  # RTX 4070 12GB 권장 (VRAM 약 10~11GB 사용, 품질 향상)

LOW_RATING_THRESHOLD = 1            # 평점 1짜리만 → 가장 심각한 리뷰만 분석
N_CLUSTERS           = 10           # TF-IDF 클러스터 수
TOP_PER_CLUSTER      = 2            # 클러스터당 추출할 대표 리뷰 수
OLLAMA_TIMEOUT       = 120          # 큰 모델이므로 여유있게 설정
CHECKPOINT_FILE      = "shoes_data/issue_checkpoint.json"
CHECKPOINT_EVERY     = 10           # N개 상품마다 체크포인트 저장
MAX_WORKERS          = 4            # 동시 LLM 요청 수

ASPECTS = ["comfort", "design", "size", "durability", "price"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0: Ollama 서버 연결 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 55)
print("STEP 0: Ollama 서버 연결 확인")
print("=" * 55)

try:
    resp = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    if not models:
        raise RuntimeError("설치된 모델 없음")
    print(f"  ✅ Ollama 연결 성공")
    print(f"  설치된 모델: {', '.join(models)}")
    if not any(OLLAMA_MODEL in m for m in models):
        print(f"\n  ⚠️  '{OLLAMA_MODEL}' 모델이 없습니다.")
        print(f"     터미널에서 실행하세요: ollama pull {OLLAMA_MODEL}")
        exit(1)
    print(f"  사용 모델: {OLLAMA_MODEL}")
except Exception as e:
    print(f"  ❌ Ollama 연결 실패: {e}")
    print("  Ollama가 실행 중인지 확인하세요: ollama serve")
    exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: 데이터 로드 & 낮은 평점 필터링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 1: 데이터 로드 & 낮은 평점 필터링")
print("=" * 55)

# parquet 없으면 json에서 변환
if not os.path.exists(INPUT_FILE):
    if not os.path.exists(JSON_FILE):
        raise FileNotFoundError(f"'{JSON_FILE}' 파일을 찾을 수 없습니다.")
    print(f"  JSON → Parquet 변환 중...")
    pd.read_json(JSON_FILE).to_parquet(INPUT_FILE, index=False)
    print(f"  변환 완료 → '{INPUT_FILE}'")

df = pd.read_parquet(INPUT_FILE)
df = df.dropna(subset=["text", "rating"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 20]   # 너무 짧은 리뷰 제거

print(f"  전체 리뷰: {len(df):,}건 / 상품 수: {df['parent_asin'].nunique():,}개")

low_df = df[df["rating"] <= LOW_RATING_THRESHOLD].copy()
print(f"  평점 {LOW_RATING_THRESHOLD}이하 리뷰: {len(low_df):,}건 "
      f"({len(low_df)/len(df)*100:.1f}%)")
print(f"  해당 상품 수: {low_df['parent_asin'].nunique():,}개")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: TF-IDF 클러스터링 → 대표 리뷰 추출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 2: TF-IDF 클러스터링 → 대표 리뷰 추출")
print("=" * 55)

def extract_representative_reviews(texts, n_clusters=N_CLUSTERS, top_n=TOP_PER_CLUSTER):
    """
    TF-IDF 벡터화 → K-Means 클러스터링 → 각 클러스터 중심에 가장 가까운 리뷰 추출.
    전체 리뷰의 약 5~10%만 LLM에 전달해 비용과 시간을 절감.
    """
    if len(texts) <= n_clusters * top_n:
        # 리뷰가 적으면 전부 사용
        return list(range(len(texts)))

    actual_clusters = min(n_clusters, len(texts) // 2)

    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 2),         # 2-gram 포함 → 'not comfortable' 같은 부정 표현 포착
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)

    km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    km.fit(tfidf_matrix)

    selected_indices = []
    for cluster_id in range(actual_clusters):
        cluster_mask  = km.labels_ == cluster_id
        cluster_idxs  = np.where(cluster_mask)[0]
        if len(cluster_idxs) == 0:
            continue
        center        = km.cluster_centers_[cluster_id]
        cluster_vecs  = tfidf_matrix[cluster_idxs]
        sims          = cosine_similarity(cluster_vecs, center.reshape(1, -1)).flatten()
        top_local     = np.argsort(sims)[::-1][:top_n]
        selected_indices.extend(cluster_idxs[top_local].tolist())

    return sorted(set(selected_indices))


# 상품별 대표 리뷰 추출
print("  상품별 대표 리뷰 추출 중...")
product_reviews = {}   # { asin: { meta, representative_reviews } }

for asin, group in tqdm(low_df.groupby("parent_asin"), desc="Clustering"):
    texts_list = group["text"].tolist()
    rep_indices = extract_representative_reviews(texts_list)
    rep_reviews = group.iloc[rep_indices]["text"].tolist()

    product_reviews[asin] = {
        "brand":              group["brand"].iloc[0],
        "product_title":      group["product_title"].iloc[0],
        "total_low_reviews":  len(group),
        "avg_low_rating":     round(group["rating"].mean(), 2),
        "representative_reviews": rep_reviews,
    }

total_rep = sum(len(v["representative_reviews"]) for v in product_reviews.values())
total_low = len(low_df)
print(f"  ✅ 대표 리뷰 추출 완료")
print(f"     전체 낮은 평점 리뷰: {total_low:,}건")
print(f"     LLM에 전달할 대표 리뷰: {total_rep:,}건 ({total_rep/total_low*100:.1f}%)")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: LLM 이슈 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 3: LLM 이슈 분석")
print("=" * 55)

SYSTEM_PROMPT = """You are a JSON-only API. You must respond with ONLY a valid JSON object.
Do NOT include any text, explanation, markdown, or code blocks outside the JSON.
Start your response with { and end with }."""

def build_prompt(product_title: str, reviews: list[str]) -> str:
    review_block = "\n".join(f"- {r[:150]}" for r in reviews[:5])
    return f"""Product: {product_title}

Negative reviews:
{review_block}

Respond ONLY with this exact JSON (no other text):
{{"top_issues":[{{"aspect":"comfort|design|size|durability|price|other","issue_summary":"string","severity":"high|medium|low","evidence":"string","improvement_suggestion":"string"}}],"overall_sentiment":"string","critical_issue":"string"}}\
"""


def call_ollama(prompt: str) -> dict | None:
    """Ollama API 호출 → JSON 파싱하여 반환. 실패 시 None."""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.1,     # 낮을수록 일관된 JSON 출력
            "num_predict": 600,     # 300 → 600으로 늘려 JSON 잘림 방지
            "top_p": 0.9,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        raw_text = resp.json().get("response", "")

        # JSON 블록 추출 (```json ... ``` 또는 { ... } 형태 모두 처리)
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            return None
        return json.loads(json_match.group())

    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None


# 체크포인트 로드
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    issue_results   = ckpt["issue_results"]
    done_asins      = set(ckpt["done_asins"])
    print(f"  ✅ 체크포인트 발견 → {len(done_asins)}개 상품 이미 처리됨, 이어서 시작")
else:
    issue_results = {}
    done_asins    = set()
    print("  체크포인트 없음 → 처음부터 시작")

asins_to_process = [a for a in product_reviews if a not in done_asins]
failed_count     = 0
lock             = threading.Lock()   # 체크포인트 저장 시 동시 쓰기 방지


def process_asin(asin):
    """단일 상품에 대해 LLM 분석 수행 → (asin, meta, result) 반환."""
    meta   = product_reviews[asin]
    prompt = build_prompt(meta["product_title"], meta["representative_reviews"])
    result = call_ollama(prompt)
    if result is None:
        result = {
            "top_issues":        [],
            "overall_sentiment": "Analysis failed",
            "critical_issue":    "N/A",
        }
    return asin, meta, result


print(f"  병렬 처리: {MAX_WORKERS}개 동시 요청")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_asin, asin): asin
               for asin in asins_to_process}

    for idx, future in enumerate(
        tqdm(as_completed(futures), total=len(asins_to_process), desc="LLM 분석")
    ):
        asin, meta, result = future.result()

        if not result.get("top_issues"):
            failed_count += 1

        with lock:
            issue_results[asin] = {
                "brand":             meta["brand"],
                "product_title":     meta["product_title"],
                "total_low_reviews": meta["total_low_reviews"],
                "avg_low_rating":    meta["avg_low_rating"],
                "rep_review_count":  len(meta["representative_reviews"]),
                "analysis":          result,
            }
            done_asins.add(asin)

            # 체크포인트 저장
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                tmp = CHECKPOINT_FILE + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"issue_results": issue_results,
                               "done_asins":    list(done_asins)}, f, ensure_ascii=False)
                os.replace(tmp, CHECKPOINT_FILE)
                tqdm.write(f"  💾 체크포인트 저장 ({idx+1}/{len(asins_to_process)})")

print(f"\n  LLM 분석 완료 / 실패 건수: {failed_count}개")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: 결과 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 4: 결과 저장")
print("=" * 55)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(issue_results, f, ensure_ascii=False, indent=2)

# 정상 완료 시 체크포인트 삭제
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("  🗑️  체크포인트 파일 삭제 (정상 종료)")

print(f"  ✅ 저장 완료: {OUTPUT_FILE}")
print(f"  분석 상품 수: {len(issue_results):,}개")
print(f"  LLM 분석 실패: {failed_count}개")
print("\n  다음 단계: streamlit run issue_dashboard.py")
