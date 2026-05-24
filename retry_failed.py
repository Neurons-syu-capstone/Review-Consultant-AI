"""
LLM 분석 실패 상품 재처리 스크립트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
개선 사항:
  1. MAX_RETRY 3 → 5         : 재시도 횟수 증가
  2. 소수 리뷰 상품 별도 처리 : 리뷰 5개 이하는 클러스터링 없이 전체 전달
  3. 소수 리뷰용 프롬프트 완화 : 이슈가 없을 수도 있음을 LLM에 명시

실행 방법:
  python retry_failed.py
"""

import json
import os
import re
import time
import threading
import warnings
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
# 설정값
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT_FILE   = "shoes_data/shoes_reviews_cleaned.parquet"
JSON_FILE    = "shoes_data/shoes_reviews_cleaned.json"
REPORT_FILE  = "shoes_data/issue_report.json"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral-nemo:12b"

LOW_RATING_THRESHOLD  = 1
N_CLUSTERS            = 10
TOP_PER_CLUSTER       = 2
OLLAMA_TIMEOUT        = 120
MAX_WORKERS           = 3
MAX_RETRY             = 5      # 3 → 5로 증가
FEW_REVIEWS_THRESHOLD = 5      # 이 수 이하면 클러스터링 없이 전체 리뷰 전달
CHECKPOINT_FILE       = "shoes_data/retry_checkpoint.json"
CHECKPOINT_EVERY      = 10

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 0: Ollama 서버 연결 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("=" * 55)
print("STEP 0: Ollama 서버 연결 확인")
print("=" * 55)

try:
    resp   = requests.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    if not any(OLLAMA_MODEL in m for m in models):
        print(f"  ⚠️  '{OLLAMA_MODEL}' 없음 → ollama pull {OLLAMA_MODEL}")
        exit(1)
    print(f"  ✅ Ollama 연결 성공 / 모델: {OLLAMA_MODEL}")
except Exception as e:
    print(f"  ❌ Ollama 연결 실패: {e}")
    exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: 기존 결과에서 실패 상품 추출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 1: 실패 상품 추출")
print("=" * 55)

if not os.path.exists(REPORT_FILE):
    print(f"  ❌ '{REPORT_FILE}' 없음. issue_pipeline.py를 먼저 실행하세요.")
    exit(1)

with open(REPORT_FILE, "r", encoding="utf-8") as f:
    report = json.load(f)

# top_issues가 비어있는 상품 = 실패
failed_asins = [
    asin for asin, info in report.items()
    if not info.get("analysis", {}).get("top_issues")
]

print(f"  전체 상품: {len(report):,}개")
print(f"  실패 상품: {len(failed_asins):,}개 ({len(failed_asins)/len(report)*100:.1f}%)")

if not failed_asins:
    print("  ✅ 실패 상품 없음. 재처리 불필요.")
    exit(0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: 실패 상품의 대표 리뷰 재추출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 2: 대표 리뷰 재추출")
print("=" * 55)

if not os.path.exists(INPUT_FILE):
    if not os.path.exists(JSON_FILE):
        raise FileNotFoundError(f"'{JSON_FILE}' 파일을 찾을 수 없습니다.")
    pd.read_json(JSON_FILE).to_parquet(INPUT_FILE, index=False)

df     = pd.read_parquet(INPUT_FILE)
df     = df.dropna(subset=["text", "rating"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df     = df[df["text"].str.len() > 20]
low_df = df[df["rating"] <= LOW_RATING_THRESHOLD].copy()

# 실패 상품만 필터링
low_df = low_df[low_df["parent_asin"].isin(failed_asins)]
print(f"  재처리 대상 리뷰: {len(low_df):,}건")


def extract_representative_reviews(texts, n_clusters=N_CLUSTERS, top_n=TOP_PER_CLUSTER):
    if len(texts) <= n_clusters * top_n:
        return list(range(len(texts)))
    actual_clusters = min(n_clusters, len(texts) // 2)
    vectorizer = TfidfVectorizer(
        max_features=3000, stop_words="english",
        ngram_range=(1, 2), min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    km = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
    km.fit(tfidf_matrix)
    selected = []
    for cid in range(actual_clusters):
        idxs = np.where(km.labels_ == cid)[0]
        if len(idxs) == 0:
            continue
        center = km.cluster_centers_[cid]
        sims   = cosine_similarity(tfidf_matrix[idxs], center.reshape(1, -1)).flatten()
        selected.extend(idxs[np.argsort(sims)[::-1][:top_n]].tolist())
    return sorted(set(selected))


product_reviews = {}
few_review_asins = []   # 리뷰 5개 이하 상품 별도 추적

for asin, group in tqdm(low_df.groupby("parent_asin"), desc="Clustering"):
    texts_list = group["text"].tolist()

    if len(texts_list) <= FEW_REVIEWS_THRESHOLD:
        # [케이스 2] 리뷰가 적으면 클러스터링 없이 전체 전달
        rep_reviews = texts_list
        few_review_asins.append(asin)
    else:
        # [케이스 1] 리뷰가 충분하면 클러스터링으로 대표 추출
        rep_idx     = extract_representative_reviews(texts_list)
        rep_reviews = group.iloc[rep_idx]["text"].tolist()

    product_reviews[asin] = {
        "brand":                  group["brand"].iloc[0],
        "product_title":          group["product_title"].iloc[0],
        "total_low_reviews":      len(group),
        "avg_low_rating":         round(group["rating"].mean(), 2),
        "representative_reviews": rep_reviews,
        "is_few_reviews":         len(texts_list) <= FEW_REVIEWS_THRESHOLD,
    }

print(f"  ✅ 대표 리뷰 추출 완료: {len(product_reviews):,}개 상품")
print(f"     일반 처리 (리뷰 많음): {len(product_reviews) - len(few_review_asins):,}개")
print(f"     소수 리뷰 처리 (전체 전달): {len(few_review_asins):,}개")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: 개선된 설정으로 LLM 재분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 3: LLM 재분석")
print("=" * 55)

# [개선] 프롬프트 강화 — JSON 외 출력 차단
SYSTEM_PROMPT = """You are a JSON-only API. You must respond with ONLY a valid JSON object.
Do NOT include any text, explanation, markdown, or code blocks outside the JSON.
Start your response with { and end with }."""


def build_prompt(product_title: str, reviews: list[str]) -> str:
    """일반 케이스 — 리뷰가 충분한 경우."""
    review_block = "\n".join(f"- {r[:150]}" for r in reviews[:5])
    return f"""Product: {product_title}

Negative reviews:
{review_block}

Respond ONLY with this exact JSON (no other text):
{{"top_issues":[{{"aspect":"comfort|design|size|durability|price|other","issue_summary":"string","severity":"high|medium|low","evidence":"string","improvement_suggestion":"string"}}],"overall_sentiment":"string","critical_issue":"string"}}"""


def build_prompt_few(product_title: str, reviews: list[str]) -> str:
    """소수 리뷰 케이스 — 이슈가 없을 수도 있음을 명시."""
    review_block = "\n".join(f"- {r[:200]}" for r in reviews)   # 전체 리뷰, 더 긴 길이
    return f"""Product: {product_title}

All available negative reviews ({len(reviews)} total):
{review_block}

These reviews may be limited. Analyze what you can and respond ONLY with this JSON.
If there are no clear issues, return an empty top_issues array.

Respond ONLY with this exact JSON (no other text):
{{"top_issues":[{{"aspect":"comfort|design|size|durability|price|other","issue_summary":"string","severity":"high|medium|low","evidence":"string","improvement_suggestion":"string"}}],"overall_sentiment":"string","critical_issue":"string"}}"""


def call_ollama_with_retry(prompt: str, is_few: bool = False,
                           max_retry: int = MAX_RETRY) -> dict | None:
    """
    최대 max_retry회 재시도하는 Ollama 호출.
    is_few=True이면 소수 리뷰 케이스로 top_issues 비어도 성공 처리.
    """
    for attempt in range(1, max_retry + 1):
        try:
            payload = {
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "system": SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "num_predict": 600,
                    "top_p": 0.9,
                },
            }
            resp     = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")

            json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if not json_match:
                raise ValueError("JSON 블록 없음")

            result = json.loads(json_match.group())

            # 소수 리뷰는 top_issues 비어도 성공 (분석할 내용 없는 경우)
            # 일반 케이스는 top_issues 있어야 성공
            if is_few or result.get("top_issues") is not None:
                return result
            raise ValueError("top_issues 없음")

        except Exception:
            if attempt < max_retry:
                time.sleep(1)
            else:
                return None


# 체크포인트 로드
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    retry_results = ckpt["retry_results"]
    done_asins    = set(ckpt["done_asins"])
    print(f"  ✅ 체크포인트 발견 → {len(done_asins)}개 이미 처리됨, 이어서 시작")
else:
    retry_results = {}
    done_asins    = set()
    print("  체크포인트 없음 → 처음부터 시작")

asins_to_retry = [a for a in product_reviews if a not in done_asins]
still_failed   = 0
lock           = threading.Lock()

print(f"  재처리 대상: {len(asins_to_retry):,}개 / 병렬: {MAX_WORKERS}개 / 최대 재시도: {MAX_RETRY}회")


def process_asin(asin):
    meta     = product_reviews[asin]
    is_few   = meta.get("is_few_reviews", False)

    # 케이스에 따라 다른 프롬프트 사용
    if is_few:
        prompt = build_prompt_few(meta["product_title"], meta["representative_reviews"])
    else:
        prompt = build_prompt(meta["product_title"], meta["representative_reviews"])

    result = call_ollama_with_retry(prompt, is_few=is_few)
    if result is None:
        result = {
            "top_issues":        [],
            "overall_sentiment": "Analysis failed after retry",
            "critical_issue":    "N/A",
        }
    return asin, meta, result


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process_asin, asin): asin
               for asin in asins_to_retry}

    for idx, future in enumerate(
        tqdm(as_completed(futures), total=len(asins_to_retry), desc="재분석")
    ):
        asin, meta, result = future.result()
        is_few = meta.get("is_few_reviews", False)

        # 소수 리뷰 상품은 top_issues 비어도 실패로 카운트 안 함
        if not result.get("top_issues") and not is_few:
            still_failed += 1

        with lock:
            retry_results[asin] = {
                "brand":             meta["brand"],
                "product_title":     meta["product_title"],
                "total_low_reviews": meta["total_low_reviews"],
                "avg_low_rating":    meta["avg_low_rating"],
                "rep_review_count":  len(meta["representative_reviews"]),
                "analysis":          result,
            }
            done_asins.add(asin)

            if (idx + 1) % CHECKPOINT_EVERY == 0:
                tmp = CHECKPOINT_FILE + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump({"retry_results": retry_results,
                               "done_asins":    list(done_asins)}, f, ensure_ascii=False)
                os.replace(tmp, CHECKPOINT_FILE)
                tqdm.write(f"  💾 체크포인트 저장 ({idx+1}/{len(asins_to_retry)})")

print(f"\n  재분석 완료 / 여전히 실패: {still_failed}개")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: 기존 결과에 재분석 결과 병합 후 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "=" * 55)
print("STEP 4: 결과 병합 및 저장")
print("=" * 55)

# 기존 report에 재분석 결과 덮어쓰기
recovered = 0
for asin, new_info in retry_results.items():
    is_few    = product_reviews.get(asin, {}).get("is_few_reviews", False)
    has_issue = bool(new_info["analysis"].get("top_issues"))

    # 이슈가 있거나, 소수 리뷰라서 원래 이슈가 없는 게 정상인 경우 모두 복구 성공
    if has_issue or is_few:
        report[asin] = new_info
        recovered += 1

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# 정상 완료 시 체크포인트 삭제
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("  🗑️  체크포인트 파일 삭제 (정상 종료)")

print(f"  ✅ 저장 완료: {REPORT_FILE}")
print(f"  재처리 시도: {len(retry_results):,}개")
print(f"  복구 성공:   {recovered:,}개")
print(f"  여전히 실패: {still_failed:,}개")
print(f"  최종 실패율: {still_failed/len(report)*100:.1f}%")
print("\n  다음 단계: streamlit run issue_dashboard.py")
