"""
ABSAScorer - LLM 기반 다차원 속성별 감성 분석 클래스
OpenAI gpt-4o-mini로 리뷰에서 속성별 감성을 직접 분류

사용 예시:
    from absa_scorer import ABSAScorer

    scorer = ABSAScorer(model="gpt-4o-mini", max_workers=3)
    scorer.load_data("shoes_data/shoes_sample.json")
    scorer.run()
    scorer.save(
        "shoes_data/llm_scores_by_product.json",
        "shoes_data/llm_sentences.json"
    )

    # 결과 직접 사용
    scores    = scorer.get_scores()    # {asin: {brand, product_title, scores}}
    sentences = scorer.get_sentences() # [{asin, category, sentiment, evidence, ...}]
"""

import json
import os
import time
import threading
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


class ABSAScorer:
    """
    LLM 기반 Aspect-Based Sentiment Analysis 스코어러

    Attributes:
        model (str):       OpenAI 모델명
        batch_size (int):  한 번에 분석할 리뷰 수
        max_workers (int): 병렬 처리 스레드 수
        max_reviews (int): 상품당 최대 리뷰 수 (None = 전체)
    """

    CATEGORIES = ["comfort", "design", "size", "durability", "price"]
    CAT_KR = {
        "comfort":    "착용감",
        "design":     "디자인",
        "size":       "사이즈",
        "durability": "내구성",
        "price":      "가격",
    }

    SYSTEM_PROMPT = """You are an expert e-commerce review analyst specializing in footwear products.
Your task is to analyze shoe reviews and classify sentiment for each aspect.

Aspects to analyze:
- comfort: fit feel, cushioning, arch support, breathability, weight, insole, padding
- design: appearance, style, color, aesthetic, look
- size: sizing accuracy, width, length, fit (too small/large/narrow/wide)
- durability: material quality, sole, stitching, how long it lasts, breaking apart
- price: value for money, affordability, worth the cost

Rules:
1. Only include aspects that are EXPLICITLY mentioned in the review
2. Sentiment must be: "positive", "negative" (NO neutral - skip if unclear)
3. Return ONLY valid JSON, no explanation
4. If no relevant aspects found, return empty aspects array
"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        batch_size: int = 10,
        max_workers: int = 3,
        max_reviews: int = None,
        max_retries: int = 3,
        retry_sleep: int = 10,
        checkpoint_every: int = 10,
        checkpoint_file: str = "shoes_data/llm_checkpoint.json",
    ):
        self.model            = model
        self.batch_size       = batch_size
        self.max_workers      = max_workers
        self.max_reviews      = max_reviews
        self.max_retries      = max_retries
        self.retry_sleep      = retry_sleep
        self.checkpoint_every = checkpoint_every
        self.checkpoint_file  = checkpoint_file

        self.client = OpenAI()
        self.lock   = threading.Lock()

        # 내부 상태
        self._df              = None
        self._product_meta    = {}
        self._product_reviews = {}
        self._scores          = {}
        self._sentences       = []

    # ══════════════════════════════════════════════════════
    # Public Methods
    # ══════════════════════════════════════════════════════

    def load_data(self, input_file: str) -> "ABSAScorer":
        """
        리뷰 데이터 로드 및 상품별 그룹화

        Args:
            input_file: shoes_sample.json 경로

        Returns:
            self (메서드 체이닝 가능)
        """
        print("=" * 55)
        print("데이터 로드")
        print("=" * 55)

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._df = pd.DataFrame(data)
        self._df["parent_asin"] = self._df["parent_asin"].astype(str)

        print(f"  총 리뷰 수 : {len(self._df):,}건")
        print(f"  총 상품 수 : {self._df['parent_asin'].nunique():,}개")

        self._group_by_product()
        return self

    def run(self) -> "ABSAScorer":
        """
        전체 분석 실행
        1. 체크포인트 로드 (이어서 실행 지원)
        2. LLM 병렬 분석
        3. 누락 상품 자동 재분석

        Returns:
            self (메서드 체이닝 가능)
        """
        if not self._product_meta:
            raise RuntimeError("load_data()를 먼저 호출하세요.")

        self._load_checkpoint()
        self._run_analysis()
        self._reanalyze_missing()
        return self

    def save(self, scores_file: str, sentences_file: str) -> "ABSAScorer":
        """
        분석 결과 저장

        Args:
            scores_file:    상품별 점수 JSON 경로
            sentences_file: 문장별 분류 결과 JSON 경로

        Returns:
            self (메서드 체이닝 가능)
        """
        print("\n" + "=" * 55)
        print("저장")
        print("=" * 55)

        os.makedirs(os.path.dirname(scores_file), exist_ok=True)

        with open(scores_file, "w", encoding="utf-8") as f:
            json.dump(self._scores, f, ensure_ascii=False, indent=2)

        with open(sentences_file, "w", encoding="utf-8") as f:
            json.dump(self._sentences, f, ensure_ascii=False, indent=2)

        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

        print(f"  점수 파일 : {scores_file}  ({len(self._scores)}개 상품)")
        print(f"  문장 파일 : {sentences_file}  ({len(self._sentences):,}건)")
        print(f"\n  ✅ 완료!")
        return self

    def get_scores(self) -> dict:
        """
        상품별 속성 점수 반환

        Returns:
            dict: {
                "asin": {
                    "brand": str,
                    "product_title": str,
                    "avg_rating": float,
                    "review_count": int,
                    "scores": {
                        "comfort":    {"score": float, "count": int, "pos_count": int, "neg_count": int},
                        "design":     {...},
                        "size":       {...},
                        "durability": {...},
                        "price":      {...},
                    }
                }
            }
        """
        return self._scores

    def get_sentences(self) -> list:
        """
        문장별 분류 결과 반환

        Returns:
            list: [
                {
                    "asin": str,
                    "brand": str,
                    "product_title": str,
                    "evidence": str,       # LLM이 추출한 근거 문장
                    "full_review": str,    # 원문 리뷰 (앞 300자)
                    "category": str,       # comfort/design/size/durability/price
                    "sentiment": str,      # positive/negative
                    "score": float,        # +1.0 or -1.0
                }
            ]
        """
        return self._sentences

    def get_scores_df(self) -> pd.DataFrame:
        """상품별 점수를 DataFrame으로 반환 (대시보드 연동용)"""
        rows = []
        for asin, info in self._scores.items():
            row = {
                "asin":          asin,
                "brand":         info.get("brand", ""),
                "product_title": info.get("product_title", ""),
                "avg_rating":    info.get("avg_rating"),
                "review_count":  info.get("review_count", 0),
            }
            for cat in self.CATEGORIES:
                s = info.get("scores", {}).get(cat, {})
                row[f"{cat}_score"]     = s.get("score")
                row[f"{cat}_count"]     = s.get("count", 0)
                row[f"{cat}_pos_count"] = s.get("pos_count", 0)
                row[f"{cat}_neg_count"] = s.get("neg_count", 0)
            rows.append(row)
        return pd.DataFrame(rows)

    def get_sentences_df(self) -> pd.DataFrame:
        """문장별 결과를 DataFrame으로 반환 (대시보드 연동용)"""
        return pd.DataFrame(self._sentences)

    # ══════════════════════════════════════════════════════
    # Private Methods
    # ══════════════════════════════════════════════════════

    def _group_by_product(self):
        """상품별 메타 및 리뷰 그룹화"""
        print("\n상품별 그룹화")
        for asin, group in self._df.groupby("parent_asin"):
            self._product_meta[asin] = {
                "brand":         group["brand"].iloc[0],
                "product_title": group["product_title"].iloc[0],
                "avg_rating":    round(group["rating"].mean(), 2),
                "review_count":  len(group),
            }
            sorted_group = group.sort_values("helpful_vote", ascending=False)
            if self.max_reviews:
                sorted_group = sorted_group.head(self.max_reviews)
            self._product_reviews[asin] = sorted_group[["rating", "text"]].to_dict("records")

        total_reviews = sum(len(v) for v in self._product_reviews.values())
        mode_str      = "전체" if self.max_reviews is None else f"상위 {self.max_reviews}개"
        print(f"  상품 수        : {len(self._product_meta):,}개")
        print(f"  분석 리뷰 수   : {total_reviews:,}건 ({mode_str})")
        print(f"  동시 요청 수   : {self.max_workers}개")

    def _load_checkpoint(self):
        """체크포인트 로드"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            self._scores    = checkpoint.get("scores", {})
            self._sentences = checkpoint.get("sentences", [])
            print(f"\n  체크포인트 발견 → {len(self._scores)}개 완료, 이어서 실행")

    def _save_checkpoint(self):
        """체크포인트 저장"""
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(
                {"scores": self._scores, "sentences": self._sentences},
                f, ensure_ascii=False
            )

    def _make_user_prompt(self, reviews: list) -> str:
        reviews_text = ""
        for i, rev in enumerate(reviews, 1):
            reviews_text += f"\n[Review {i}] Rating: {rev['rating']}/5\n{rev['text']}\n"
        return f"""Analyze these shoe reviews and extract aspect sentiments.
{reviews_text}
Return JSON in this exact format:
{{
  "reviews": [
    {{
      "review_id": 1,
      "aspects": [
        {{"aspect": "comfort", "sentiment": "positive", "evidence": "very comfortable to wear"}},
        {{"aspect": "size", "sentiment": "negative", "evidence": "runs too small"}}
      ]
    }}
  ]
}}"""

    def _call_openai(self, reviews: list) -> list:
        """OpenAI API 호출 (Rate Limit 자동 재시도)"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": self._make_user_prompt(reviews)},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=2000,
                )
                raw    = response.choices[0].message.content.strip()
                result = json.loads(raw)
                if isinstance(result, list):
                    return result
                return result.get("reviews", [])
            except Exception as e:
                err_str = str(e)
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    wait = self.retry_sleep * (attempt + 1)
                    tqdm.write(f"  [Rate Limit] {wait}초 대기 후 재시도 ({attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    tqdm.write(f"  [WARN] API 오류: {e}")
                    return []
        return []

    @staticmethod
    def _aggregate(score_list: list) -> dict:
        """감성 점수 리스트 → 10점 척도"""
        if not score_list:
            return {"score": None, "count": 0, "pos_count": 0, "neg_count": 0}
        pos = sum(1 for s in score_list if s > 0)
        neg = sum(1 for s in score_list if s < 0)
        avg = sum(score_list) / len(score_list)
        return {
            "score":     round((avg + 1) / 2 * 10, 2),
            "count":     len(score_list),
            "pos_count": pos,
            "neg_count": neg,
        }

    def _process_product(self, asin: str) -> dict:
        """상품 1개 전체 배치 처리"""
        reviews    = self._product_reviews[asin]
        meta       = self._product_meta[asin]
        cat_scores = {cat: [] for cat in self.CATEGORIES}
        sentences  = []

        batches = [reviews[i:i + self.batch_size] for i in range(0, len(reviews), self.batch_size)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as batch_executor:
            future_to_batch = {
                batch_executor.submit(self._call_openai, batch): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch   = future_to_batch[future]
                results = future.result()

                for rev_result in results:
                    if not isinstance(rev_result, dict):
                        continue
                    rev_idx = rev_result.get("review_id", 1) - 1
                    if rev_idx < 0 or rev_idx >= len(batch):
                        continue
                    original_text = batch[rev_idx]["text"]

                    for aspect_item in rev_result.get("aspects", []):
                        if not isinstance(aspect_item, dict):
                            continue
                        cat       = aspect_item.get("aspect", "")
                        sentiment = aspect_item.get("sentiment", "")
                        evidence  = aspect_item.get("evidence", "")

                        if cat not in self.CATEGORIES or sentiment not in ("positive", "negative"):
                            continue

                        score = 1.0 if sentiment == "positive" else -1.0
                        cat_scores[cat].append(score)
                        sentences.append({
                            "asin":          str(asin),
                            "brand":         meta["brand"],
                            "product_title": meta["product_title"],
                            "evidence":      evidence,
                            "full_review":   original_text[:300],
                            "category":      cat,
                            "sentiment":     sentiment,
                            "score":         score,
                        })

        return {
            "asin": asin,
            "result": {
                "brand":         meta["brand"],
                "product_title": meta["product_title"],
                "avg_rating":    meta["avg_rating"],
                "review_count":  meta["review_count"],
                "scores":        {cat: self._aggregate(cat_scores[cat]) for cat in self.CATEGORIES},
            },
            "sentences": sentences,
        }

    def _run_analysis(self):
        """병렬 분석 실행"""
        print("\n" + "=" * 55)
        print(f"LLM 분석 (병렬 {self.max_workers}개)")
        print("=" * 55)

        done_asins       = set(self._scores.keys())
        asins_to_process = [a for a in self._product_meta if a not in done_asins]
        processed_count  = 0
        pbar             = tqdm(total=len(asins_to_process), desc="Products")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_asin = {
                executor.submit(self._process_product, asin): asin
                for asin in asins_to_process
            }
            for future in as_completed(future_to_asin):
                try:
                    output = future.result()
                except Exception as e:
                    tqdm.write(f"  [WARN] 상품 처리 오류: {e}")
                    pbar.update(1)
                    continue

                with self.lock:
                    self._scores[output["asin"]]  = output["result"]
                    self._sentences              += output["sentences"]
                    processed_count              += 1

                    if processed_count % self.checkpoint_every == 0:
                        self._save_checkpoint()
                        tqdm.write(
                            f"  [체크포인트] {processed_count}개 완료 / "
                            f"{len(self._sentences):,}건 문장 저장"
                        )
                pbar.update(1)

        pbar.close()

    def _reanalyze_missing(self):
        """누락 상품 자동 감지 및 재분석"""
        print("\n" + "=" * 55)
        print("누락 상품 확인 및 재분석")
        print("=" * 55)

        sample_asins  = set(self._df["parent_asin"].unique())
        scores_asins  = set(self._scores.keys())
        absent_asins  = sample_asins - scores_asins
        none_asins    = {
            asin for asin, info in self._scores.items()
            if all(s["score"] is None for s in info["scores"].values())
        }
        missing_asins = list(absent_asins | none_asins)

        print(f"  전체 상품 수       : {len(sample_asins)}개")
        print(f"  분석 완료 상품 수  : {len(scores_asins)}개")
        print(f"  JSON에 없는 상품   : {len(absent_asins)}개")
        print(f"  점수 전부 None     : {len(none_asins)}개")
        print(f"  재분석 대상        : {len(missing_asins)}개")

        if not missing_asins:
            print("  ✅ 누락 상품 없음!")
            return

        for asin in tqdm(missing_asins, desc="Re-analyzing"):
            try:
                output = self._process_product(asin)
                with self.lock:
                    self._scores[asin]  = output["result"]
                    self._sentences    += output["sentences"]
                tqdm.write(
                    f"  ✅ [{output['result']['brand']}] "
                    f"{output['result']['product_title'][:40]}"
                )
            except Exception as e:
                tqdm.write(f"  [WARN] {asin} 재분석 오류: {e}")


# ══════════════════════════════════════════════════════════
# 직접 실행 시 (python absa_scorer.py)
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    scorer = ABSAScorer(
        model       = "gpt-4o-mini",
        batch_size  = 10,
        max_workers = 3,
        max_reviews = None,   # None = 전체 리뷰 분석
    )

    scorer.load_data("shoes_data/shoes_sample.json")
    scorer.run()
    scorer.save(
        scores_file    = "shoes_data/llm_scores_by_product.json",
        sentences_file = "shoes_data/llm_sentences.json",
    )

    # 결과 확인
    df_scores = scorer.get_scores_df()
    print(f"\n분석 완료: {len(df_scores)}개 상품")
    print(df_scores[["brand", "product_title", "comfort_score", "size_score"]].head())