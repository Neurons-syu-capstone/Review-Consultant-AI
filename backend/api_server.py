"""
AI 컨설턴트 백엔드 서버
Flask + OpenAI GPT-5.4-mini

실행:
  pip install flask flask-cors openai python-dotenv
  python api_server.py
"""

import json
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client     = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL      = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

SCORE_FILE     = "shoes_data/llm_scores_by_product.json"
SENTENCES_FILE = "shoes_data/llm_sentences.json"

CATEGORIES = ["comfort", "design", "size", "durability", "price"]
CAT_KR     = {
    "comfort":    "착용감",
    "design":     "디자인",
    "size":       "사이즈",
    "durability": "내구성",
    "price":      "가격",
}

# ── 데이터 로드 ────────────────────────────────────────────────
def load_scores():
    with open(SCORE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_sentences():
    if not os.path.exists(SENTENCES_FILE):
        return []
    with open(SENTENCES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

scores_data    = load_scores()
sentences_data = load_sentences()


# ── 상품 목록 API ──────────────────────────────────────────────
@app.route("/api/products", methods=["GET"])
def get_products():
    products = []
    for asin, info in scores_data.items():
        scores = info.get("scores", {})
        product = {
            "asin":          asin,
            "brand":         info.get("brand", "Unknown"),
            "product_title": info.get("product_title", ""),
            "avg_rating":    info.get("avg_rating"),
            "review_count":  info.get("review_count", 0),
        }
        for cat in CATEGORIES:
            s = scores.get(cat, {})
            product[f"{cat}_score"]     = s.get("score")
            product[f"{cat}_pos_count"] = s.get("pos_count", s.get("positive", 0))
            product[f"{cat}_neg_count"] = s.get("neg_count", s.get("negative", 0))
        products.append(product)
    return jsonify(products)


# ── 전체 속성 요약 진단 API ────────────────────────────────────
@app.route("/api/diagnose/overview", methods=["POST"])
def diagnose_overview():
    data  = request.json
    asin  = data.get("asin")
    info  = scores_data.get(asin, {})
    title = info.get("product_title", "")

    # 속성별 점수 수집
    scores = info.get("scores", {})
    score_lines = []
    for cat in CATEGORIES:
        s  = scores.get(cat, {})
        sc = s.get("score")
        if sc is not None:
            score_lines.append(f"- {CAT_KR[cat]}: {sc:.1f}/10점")

    # 부정 문장 수집 (모든 속성)
    neg_sents = [
        s for s in sentences_data
        if str(s.get("asin")) == str(asin) and s.get("sentiment") == "negative"
    ]
    ev_col = "evidence" if neg_sents and "evidence" in neg_sents[0] else "sentence_en"
    review_samples = "\n".join(
        f"- [{CAT_KR.get(s.get('category',''), s.get('category',''))}] {s.get(ev_col,'')}"
        for s in neg_sents[:15]
    )

    prompt = f"""당신은 이커머스 신발 판매자를 위한 상품 품질 분석 전문가입니다.
아래는 '{title}'의 속성별 만족도 점수와 고객 부정 리뷰입니다.

[속성별 점수]
{chr(10).join(score_lines)}

[고객 부정 리뷰 샘플]
{review_samples}

전체 속성을 종합하여 아래 형식으로 한국어로 분석해주세요:

⚡ 가장 시급한 문제
(단 하나의 가장 중요한 문제를 한 문장으로)

📋 속성별 개선 우선순위
(각 속성별로 한 줄씩 우선순위와 이유)

💊 즉시 실행 가이드라인
1. (가장 먼저 해야 할 것)
2. (그 다음)
3. (장기적으로)"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "당신은 이커머스 상품 품질 분석 전문가입니다."},
                {"role": "user",   "content": prompt},
            ],
            max_completion_tokens=600,
        )
        text   = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost   = (response.usage.prompt_tokens / 1_000_000 * 0.75
                  + response.usage.completion_tokens / 1_000_000 * 4.50)
        return jsonify({"result": text, "tokens": tokens, "cost": round(cost, 5)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 속성별 상세 진단 API ───────────────────────────────────────
@app.route("/api/diagnose/aspect", methods=["POST"])
def diagnose_aspect():
    data    = request.json
    asin    = data.get("asin")
    cat     = data.get("category")
    info    = scores_data.get(asin, {})
    title   = info.get("product_title", "")
    scores  = info.get("scores", {})
    sc      = scores.get(cat, {}).get("score", 0)

    neg_sents = [
        s for s in sentences_data
        if (str(s.get("asin")) == str(asin)
            and s.get("category") == cat
            and s.get("sentiment") == "negative")
    ]
    ev_col        = "evidence" if neg_sents and "evidence" in neg_sents[0] else "sentence_en"
    review_block  = "\n".join(f"- {s.get(ev_col, '')}" for s in neg_sents[:10])

    if not review_block:
        return jsonify({"error": "분석할 부정 리뷰 데이터가 없습니다."}), 400

    prompt = f"""당신은 이커머스 신발 판매자를 위한 상품 품질 분석 전문가입니다.
아래는 '{title}'의 [{CAT_KR.get(cat, cat)}] 속성 관련 고객 부정 리뷰입니다.
현재 [{CAT_KR.get(cat, cat)}] 만족도 점수: {sc:.1f}/10점

[부정 리뷰]
{review_block}

다음 형식으로 한국어로 분석해주세요:

🔍 진단 요약
(반복되는 문제 패턴과 고객 불만의 핵심을 2~3문장으로)

⚡ 가장 시급한 문제
(단 하나의 가장 중요한 문제를 한 문장으로)

💊 운영 개선 가이드라인
1. (즉시 조치 가능한 개선안)
2. (중기적 개선안)
3. (장기적 개선안)

📊 예상 효과
(개선 시 기대할 수 있는 구체적 효과)"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "당신은 이커머스 상품 품질 분석 전문가입니다."},
                {"role": "user",   "content": prompt},
            ],
            max_completion_tokens=700,
        )
        text   = response.choices[0].message.content
        tokens = response.usage.total_tokens
        cost   = (response.usage.prompt_tokens / 1_000_000 * 0.75
                  + response.usage.completion_tokens / 1_000_000 * 4.50)
        return jsonify({
            "result":   text,
            "tokens":   tokens,
            "cost":     round(cost, 5),
            "category": cat,
            "cat_kr":   CAT_KR.get(cat, cat),
            "score":    sc,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
