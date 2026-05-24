"""
PyABSA 기반 다차원 만족도 스코어보드 - 상품별
- 속성(Aspect) 추출 + 감성 분류 → 항목별 점수 산출
- 대상 속성: comfort(착용감), design(디자인), size(사이즈), durability(내구성), price(가격)

개선 사항:
  1. numpy 호환성 안정화   : numpy 버전 체크 후 경고 출력
  2. Aspect 임계값 필터    : confidence < CONF_THRESHOLD 인 결과는 none 처리
  3. 중간 저장(체크포인트) : CHECKPOINT_EVERY 배치마다 자동 저장, 재실행 시 이어서 처리
  4. 배치 처리 최적화      : 텍스트 길이 기준 정렬 후 배치 구성 → GPU 패딩 낭비 최소화

※ 사전 조건: Python 3.10 환경에서 실행 권장
   pip install pyabsa==2.4.3 pandas tqdm pyarrow streamlit plotly numpy==1.24.4
   GPU (CUDA 12.1): pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
   GPU (CUDA 11.8): pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
"""

import json
import os
import warnings
warnings.filterwarnings("ignore")

# ── [개선 1] numpy 호환성 안정화 ──────────────────────────────
import numpy as np
_np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
if _np_ver >= (1, 25):
    print(f"  ⚠️  numpy {np.__version__} 감지 → pyabsa 호환 문제 발생 가능")
    print("     권장 버전: pip install numpy==1.24.4")
else:
    print(f"  ✅ numpy {np.__version__} (호환 버전)")

import torch
import pandas as pd
from tqdm import tqdm
from pyabsa import AspectTermExtraction as ATEPC

# ── 경로 설정 ─────────────────────────────────────────────────
JSON_FILE        = "shoes_data/shoes_reviews_cleaned.json"
INPUT_FILE       = "shoes_data/shoes_reviews_cleaned.parquet"
OUTPUT_FILE      = "shoes_data/absa_scores_by_product.json"
CHECKPOINT_FILE  = "shoes_data/absa_checkpoint.json"   # 중간 저장 파일

# ── [개선 2] Aspect 신뢰도 임계값 ────────────────────────────
# confidence 가 이 값 미만이면 해당 aspect를 none(무시) 처리
CONF_THRESHOLD = 0.75

# ── 체크포인트 저장 주기 ──────────────────────────────────────
# 배치 N개마다 중간 저장 (RTX 4070 기준 배치128 → 약 2분마다 저장)
CHECKPOINT_EVERY = 50

# ── GPU / CPU 디바이스 설정 ───────────────────────────────────
print("=" * 55)
print("GPU 환경 확인")
print("=" * 55)

if torch.cuda.is_available():
    DEVICE   = "cuda"
    GPU_NAME = torch.cuda.get_device_name(0)
    VRAM_GB  = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if VRAM_GB < 4:
        BATCH_SIZE = 32
    elif VRAM_GB < 8:
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 128
    print(f"  ✅ GPU 사용: {GPU_NAME}")
    print(f"  VRAM: {VRAM_GB:.1f} GB  →  배치 사이즈 자동 설정: {BATCH_SIZE}")
    torch.cuda.empty_cache()
else:
    DEVICE     = "cpu"
    BATCH_SIZE = 32
    print("  ⚠️  GPU를 찾을 수 없습니다. CPU로 실행합니다.")
    print("     (CUDA PyTorch 설치 여부와 nvidia-smi 출력을 확인하세요.)")

# ── STEP 0: JSON → Parquet 변환 ──────────────────────────────
print("\n" + "=" * 55)
print("STEP 0: JSON → Parquet 변환")
print("=" * 55)

if not os.path.exists(INPUT_FILE):
    if not os.path.exists(JSON_FILE):
        raise FileNotFoundError(
            f"'{JSON_FILE}' 파일을 찾을 수 없습니다.\n"
            "shoes_data/ 폴더에 shoes_reviews_cleaned.json을 넣어 주세요."
        )
    print(f"  '{JSON_FILE}' 을 읽어 parquet으로 변환 중...")
    _df_tmp = pd.read_json(JSON_FILE, encoding="utf-8")
    _df_tmp.to_parquet(INPUT_FILE, index=False)
    print(f"  변환 완료 → '{INPUT_FILE}'")
else:
    print(f"  '{INPUT_FILE}' 이미 존재 → 변환 생략")

# ── 속성 키워드 사전 ──────────────────────────────────────────
ASPECT_CATEGORIES = {
    "comfort":    ["comfort", "comfortable", "cushion", "soft", "padding",
                   "support", "arch", "insole", "cozy", "feel", "wearing",
                   "breathable", "lightweight", "heavy"],
    "design":     ["design", "style", "look", "color", "colour", "appearance",
                   "aesthetic", "beautiful", "ugly", "stylish", "sleek"],
    "size":       ["size", "sizing", "fit", "width", "narrow", "wide",
                   "small", "large", "tight", "loose", "length", "toe"],
    "durability": ["durable", "durability", "quality", "material", "sole",
                   "stitching", "last", "wear", "broke", "falling", "glue",
                   "leather", "construction", "sturdy", "flimsy"],
    "price":      ["price", "value", "worth", "expensive", "cheap",
                   "affordable", "cost", "money", "overpriced", "budget"],
}

SAMPLE_N = None   # 테스트 시 숫자로 변경 (예: 500)

# ── STEP 1: 데이터 로드 ───────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 1: 데이터 로드")
print("=" * 55)

df = pd.read_parquet(INPUT_FILE)

if SAMPLE_N:
    df = df.sample(n=min(SAMPLE_N, len(df)), random_state=42).reset_index(drop=True)
    print(f"  샘플링: {len(df):,}건 (테스트 모드)")
else:
    print(f"  전체 데이터: {len(df):,}건")

print(f"  상품 수: {df['parent_asin'].nunique():,}개")

# ── [개선 4] 텍스트 길이 기준 정렬 → GPU 패딩 낭비 최소화 ──────
# 길이가 비슷한 텍스트끼리 같은 배치에 묶이면
# 모델 내부 패딩이 줄어들어 VRAM 사용량과 처리 시간이 감소
print("  텍스트 길이 기준 정렬 중 (배치 최적화)...")
df["_text_len"] = df["text"].str.len()
df = df.sort_values("_text_len").reset_index(drop=True)
df.drop(columns=["_text_len"], inplace=True)
print("  정렬 완료")

# ── STEP 2: PyABSA 모델 로드 ─────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: PyABSA 모델 로드")
print("=" * 55)

aspect_extractor = ATEPC.AspectExtractor(
    "multilingual",
    auto_device=True,
)
print(f"  모델 로드 완료 (디바이스: {DEVICE.upper()})")

# ── STEP 3: Aspect 추출 및 감성 분류 ─────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Aspect 추출 및 감성 분류")
print("=" * 55)

# ── [개선 3] 체크포인트 로드 ──────────────────────────────────
# 이전 실행이 중단된 경우 체크포인트에서 이어서 처리
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    product_results = ckpt["product_results"]
    start_batch_idx = ckpt["next_batch_idx"]
    print(f"  ✅ 체크포인트 발견 → 배치 {start_batch_idx}번부터 이어서 처리합니다.")
    print(f"     (이미 처리된 상품 수: {len(product_results):,}개)")
else:
    product_results = {}
    start_batch_idx = 0
    print("  체크포인트 없음 → 처음부터 시작합니다.")


def map_to_category(aspect_term):
    """aspect 단어를 5개 속성 중 하나로 매핑. 해당 없으면 'other' 반환."""
    term = aspect_term.lower()
    for category, keywords in ASPECT_CATEGORIES.items():
        if any(kw in term for kw in keywords):
            return category
    return "other"


def sentiment_to_score(sentiment, confidence):
    """
    감성 + 신뢰도를 -1 ~ +1 점수로 변환.
    [개선 2] confidence < CONF_THRESHOLD 이면 None 반환 → 집계에서 제외
    """
    if confidence < CONF_THRESHOLD:
        return None   # 임계값 미만 → none 처리
    if sentiment == "Positive":
        return confidence
    elif sentiment == "Negative":
        return -confidence
    return None       # Neutral 도 제외


def save_checkpoint(product_results, next_batch_idx):
    """
    중간 결과를 체크포인트 파일에 저장.
    임시 파일에 먼저 쓴 뒤 이름 변경(원자적 저장)
    → 저장 도중 중단되더라도 기존 체크포인트 파일이 손상되지 않음.
    """
    ckpt = {
        "next_batch_idx": next_batch_idx,
        "product_results": product_results,
    }
    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)
    os.replace(tmp_path, CHECKPOINT_FILE)


texts        = df["text"].tolist()
parent_asins = df["parent_asin"].tolist()
brands       = df["brand"].tolist()
titles       = df["product_title"].tolist()
ratings      = df["rating"].tolist()

total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
skipped_count = 0   # 임계값 미만으로 제외된 aspect 수

for batch_num, i in enumerate(
    tqdm(range(0, len(texts), BATCH_SIZE), desc="Processing", total=total_batches),
    start=0,
):
    # 체크포인트 이후 배치부터 처리
    if batch_num < start_batch_idx:
        continue

    batch_texts   = texts[i:i + BATCH_SIZE]
    batch_asins   = parent_asins[i:i + BATCH_SIZE]
    batch_brands  = brands[i:i + BATCH_SIZE]
    batch_titles  = titles[i:i + BATCH_SIZE]
    batch_ratings = ratings[i:i + BATCH_SIZE]

    # ── 모델 추론 ─────────────────────────────────────────────
    try:
        results = aspect_extractor.predict(
            batch_texts, save_result=False, print_result=False, ignore_error=True,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            BATCH_SIZE = max(8, BATCH_SIZE // 2)
            tqdm.write(f"  [WARN] GPU 메모리 부족 → 배치 사이즈를 {BATCH_SIZE}로 줄입니다.")
            try:
                results = aspect_extractor.predict(
                    batch_texts, save_result=False, print_result=False, ignore_error=True,
                )
            except Exception as e2:
                tqdm.write(f"  [WARN] 재시도 실패, 배치 {batch_num} 건너뜀: {e2}")
                continue
        else:
            tqdm.write(f"  [WARN] 배치 {batch_num} 오류: {e}")
            continue
    except Exception as e:
        tqdm.write(f"  [WARN] 배치 {batch_num} 오류: {e}")
        continue

    # ── 결과 누적 ─────────────────────────────────────────────
    for result, asin, brand, title, rating in zip(
        results, batch_asins, batch_brands, batch_titles, batch_ratings
    ):
        if asin not in product_results:
            product_results[asin] = {
                "meta": {"brand": brand, "product_title": title, "avg_rating": []},
                **{cat: [] for cat in ASPECT_CATEGORIES},
            }
        product_results[asin]["meta"]["avg_rating"].append(float(rating))

        for aspect, sentiment, conf in zip(
            result.get("aspect", []),
            result.get("sentiment", []),
            result.get("confidence", []),
        ):
            category = map_to_category(aspect)
            if category == "other":
                continue

            score = sentiment_to_score(sentiment, float(conf))
            if score is None:
                # [개선 2] 임계값 미만 또는 중립 → 제외
                skipped_count += 1
                continue

            product_results[asin][category].append(score)

    # ── [개선 3] 주기적 체크포인트 저장 ──────────────────────
    if (batch_num + 1) % CHECKPOINT_EVERY == 0:
        save_checkpoint(product_results, batch_num + 1)
        tqdm.write(f"  💾 체크포인트 저장 (배치 {batch_num + 1}/{total_batches})")

print(f"\n  임계값({CONF_THRESHOLD}) 미만으로 제외된 aspect 수: {skipped_count:,}건")

# ── STEP 4: 점수 집계 ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: 점수 집계")
print("=" * 55)


def aggregate_scores(score_list):
    """
    -1 ~ +1 범위의 점수 리스트를 0 ~ 10점으로 환산.
    데이터 없으면 None 반환.
    """
    if not score_list:
        return {"score": None, "count": 0}
    avg = sum(score_list) / len(score_list)
    return {"score": round((avg + 1) / 2 * 10, 2), "count": len(score_list)}


final_output = {}
for asin, data in product_results.items():
    meta = data["meta"]
    final_output[asin] = {
        "brand":         meta["brand"],
        "product_title": meta["product_title"],
        "avg_rating":    round(sum(meta["avg_rating"]) / len(meta["avg_rating"]), 2),
        "review_count":  len(meta["avg_rating"]),
        "scores":        {cat: aggregate_scores(data[cat]) for cat in ASPECT_CATEGORIES},
    }

# ── 최종 결과 저장 ────────────────────────────────────────────
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

# 정상 완료 시 체크포인트 파일 삭제
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("  🗑️  체크포인트 파일 삭제 완료 (정상 종료)")

print(f"  저장 완료: {OUTPUT_FILE}")
print(f"  총 상품 수: {len(final_output):,}개")
print(f"  실행 디바이스: {DEVICE.upper()}")
if DEVICE == "cuda":
    used_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
    print(f"  GPU 최대 사용 VRAM: {used_gb:.2f} GB")
print(f"  신뢰도 임계값({CONF_THRESHOLD}) 미만 제외 건수: {skipped_count:,}건")
print("\n※ 전체 데이터 실행 시 SAMPLE_N = None 으로 변경하세요.")
print("※ 임계값 조정: CONF_THRESHOLD 값을 변경하세요. (기본값: 0.75)")
