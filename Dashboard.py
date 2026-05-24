"""
다차원 만족도 스코어보드 Streamlit 대시보드
llm_scores_by_product.json + llm_sentences.json 시각화

실행: streamlit run dashboard.py
pip install streamlit plotly pandas
"""

import json
import os
import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ══════════════════════════════════════════════════════════
# 설정
# ══════════════════════════════════════════════════════════
SCORE_FILE     = "shoes_data/llm_scores_by_product.json"
SENTENCES_FILE = "shoes_data/llm_sentences.json"
OPENAI_MODEL   = "gpt-5.4-mini"   # AI 이슈 진단에 사용할 모델

CATEGORIES = ["comfort", "design", "size", "durability", "price"]
CAT_KR = {
    "comfort":    "착용감",
    "design":     "디자인",
    "size":       "사이즈",
    "durability": "내구성",
    "price":      "가격",
}
CAT_EMOJI = {
    "comfort": "🦶", "design": "🎨",
    "size": "📏", "durability": "🔩", "price": "💰",
}
COLOR_MAP = {
    "Skechers": "#0057A8", "Nike": "#FF6B35", "adidas": "#1A1A1A",
    "Crocs": "#F6C700",    "ASICS": "#003DA5", "Merrell": "#4A7C3F",
    "PUMA": "#E31837",     "KEEN": "#F4A100",  "Clarks": "#8B6914",
    "NINE WEST": "#C71585",
}

# ══════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="신발 리뷰 다차원 만족도 스코어보드",
    page_icon="👟",
    layout="wide",
)

# ══════════════════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════════════════
@st.cache_data
def load_scores(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for asin, info in raw.items():
        row = {
            "asin":          asin,
            "brand":         info.get("brand", "Unknown"),
            "product_title": info.get("product_title", ""),
            "avg_rating":    info.get("avg_rating"),
            "review_count":  info.get("review_count", 0),
        }
        for cat in CATEGORIES:
            s = info.get("scores", {}).get(cat, {})
            row[f"{cat}_score"]     = s.get("score")
            row[f"{cat}_pos_count"] = s.get("pos_count", s.get("positive", 0))
            row[f"{cat}_neg_count"] = s.get("neg_count", s.get("negative", 0))
            row[f"{cat}_count"]     = s.get("count", row[f"{cat}_pos_count"] + row[f"{cat}_neg_count"])
        rows.append(row)
    return pd.DataFrame(rows)

@st.cache_data
def load_sentences(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

try:
    df = load_scores(SCORE_FILE)
except FileNotFoundError:
    st.error(f"❌ '{SCORE_FILE}' 없음. absa_scorer.py를 먼저 실행하세요.")
    st.stop()

sent_df = load_sentences(SENTENCES_FILE)

# ══════════════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════════════
st.title("👟 신발 리뷰 다차원 만족도 스코어보드")
st.caption("LLM(gpt-4o-mini) 기반 Aspect-Level 감성 분석 | 점수 범위: 0 ~ 10점")

with st.expander("ℹ️ 점수 계산 방식 및 기준 안내"):
    st.markdown("""
    **1. 점수 계산 방법**
    - 각 속성에 대한 리뷰 문장을 LLM이 긍정(Positive) 또는 부정(Negative)으로 분류합니다.
    - **속성별 만족도 점수 (10점 만점) = (긍정 수 / 전체 수) × 10**
    - 해당 속성에 대한 리뷰가 없으면 `N/A`로 표시됩니다.

    **2. 점수 기준표**
    - 🟢 **7.0 ~ 10.0점** : 긍정 우세 (우수)
    - 🟡 **4.0 ~ 6.9점** : 긍정·부정 혼재 (보통)
    - 🔴 **0.0 ~ 3.9점** : 부정 우세 (미흡)
    - ⚪ **N/A** : 관련 리뷰 없음
    """)

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("총 상품 수",  f"{len(df):,}개")
c2.metric("브랜드 수",   f"{df['brand'].nunique()}개")
c3.metric("평균 별점",   f"{df['avg_rating'].mean():.2f} ⭐")
c4.metric("총 리뷰 수",  f"{int(df['review_count'].sum()):,}건")
st.divider()

# ══════════════════════════════════════════════════════════
# 섹션 1: 상품 선택 + 레이더 차트 + 속성 점수 카드
# ══════════════════════════════════════════════════════════
st.subheader("📊 상품별 속성 만족도")

brands = sorted(df["brand"].dropna().unique().tolist())
col_search, col_brand = st.columns([3, 1])
with col_brand:
    brand_filter = st.selectbox("브랜드로 좁히기", ["전체"] + brands, key="brand_filter")
with col_search:
    search_query = st.text_input("상품명 검색", placeholder="예: Classic Clog, Air Max ...")

display_df = df.copy()
if brand_filter != "전체":
    display_df = display_df[display_df["brand"] == brand_filter]
if search_query:
    display_df = display_df[
        display_df["product_title"].str.contains(search_query, case=False, na=False)
    ]

if display_df.empty:
    st.warning("조건에 맞는 상품이 없습니다.")
    st.stop()

product_options = {
    f"[{r['brand']}] {r['product_title'][:55]}  (리뷰 {r['review_count']}건 / ⭐{r['avg_rating']})": r["asin"]
    for _, r in display_df.iterrows()
}
selected_label = st.selectbox("상품 선택", list(product_options.keys()))
selected_asin  = product_options[selected_label]
prod           = display_df[display_df["asin"] == selected_asin].iloc[0]

# 문장 데이터에서 긍정/부정 카운트 계산
prod_sent = pd.DataFrame()
if not sent_df.empty:
    prod_sent = sent_df[sent_df["asin"] == str(selected_asin)]

radar_col, score_col = st.columns([3, 2])

with radar_col:
    scores_clean = [
        prod.get(f"{cat}_score") if pd.notna(prod.get(f"{cat}_score")) else 0
        for cat in CATEGORIES
    ]
    labels      = [CAT_KR[cat] for cat in CATEGORIES]
    brand_color = COLOR_MAP.get(prod["brand"], "#6366F1")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_clean + [scores_clean[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(99,110,250,0.2)",
        line=dict(color=brand_color, width=2.5),
        name=prod["product_title"][:30],
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=11))),
        showlegend=False,
        margin=dict(t=30, b=30, l=20, r=20),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

with score_col:
    st.markdown(f"**{prod['product_title'][:50]}**")
    st.markdown(f"브랜드: `{prod['brand']}` | ⭐ {prod['avg_rating']} | 리뷰 {prod['review_count']}건")
    st.markdown("---")

    for cat in CATEGORIES:
        score = prod.get(f"{cat}_score")

        if not prod_sent.empty:
            cat_sent  = prod_sent[prod_sent["category"] == cat]
            pos_count = len(cat_sent[cat_sent["sentiment"] == "positive"])
            neg_count = len(cat_sent[cat_sent["sentiment"] == "negative"])
        else:
            pos_count = int(prod.get(f"{cat}_pos_count", 0) or 0)
            neg_count = int(prod.get(f"{cat}_neg_count", 0) or 0)

        emoji = CAT_EMOJI[cat]

        if pd.notna(score):
            filled = int(score / 10 * 20)
            bar    = "█" * filled + "░" * (20 - filled)
            icon   = "🔴" if score < 4 else ("🟡" if score < 7 else "🟢")
            st.markdown(
                f"{icon} {emoji} **{CAT_KR[cat]}** &nbsp; `{score:.1f} / 10` "
                f"&nbsp; 👍{pos_count} 👎{neg_count}"
            )
            st.markdown(f"`{bar}`")
        else:
            st.markdown(f"⚪ {emoji} **{CAT_KR[cat]}** &nbsp; `N/A`")

st.divider()

# ══════════════════════════════════════════════════════════
# 섹션 2: 속성 버튼 클릭 → 관련 문장 표시
# ══════════════════════════════════════════════════════════
st.subheader("🔍 속성별 리뷰 문장 드릴다운")

if sent_df.empty:
    st.info(f"'{SENTENCES_FILE}' 없음.")
elif prod_sent.empty:
    st.warning("해당 상품의 문장 데이터가 없습니다.")
else:
    st.markdown("**속성 선택 (클릭하면 관련 문장이 아래에 나타납니다)**")
    btn_cols = st.columns(5)

    for i, cat in enumerate(CATEGORIES):
        cat_sent  = prod_sent[prod_sent["category"] == cat]
        pos_count = len(cat_sent[cat_sent["sentiment"] == "positive"])
        neg_count = len(cat_sent[cat_sent["sentiment"] == "negative"])
        score     = prod.get(f"{cat}_score")

        if pd.notna(score):
            score_str = f"{score:.1f}점"
            icon      = "🟢" if score >= 7 else ("🟡" if score >= 4 else "🔴")
        else:
            score_str = "N/A"
            icon      = "⚪"

        with btn_cols[i]:
            if st.button(
                f"{CAT_EMOJI[cat]} {CAT_KR[cat]}\n{icon} {score_str}\n👍{pos_count} 👎{neg_count}",
                key=f"btn_{cat}",
                use_container_width=True,
            ):
                st.session_state["selected_cat"] = cat

    # 버튼을 한 번도 누르지 않았으면 안내 문구만 표시
    if "selected_cat" not in st.session_state:
        st.markdown("")
        st.info("위 속성 버튼을 클릭하면 관련 리뷰 문장이 표시됩니다.")
    else:
        current_cat = st.session_state["selected_cat"]
        cat_sent_df = prod_sent[prod_sent["category"] == current_cat]

        st.markdown(
            f"---\n#### {CAT_EMOJI[current_cat]} {CAT_KR[current_cat]} "
            f"관련 문장 ({len(cat_sent_df)}건)"
        )

        pos_df = cat_sent_df[cat_sent_df["sentiment"] == "positive"]
        neg_df = cat_sent_df[cat_sent_df["sentiment"] == "negative"]

        pos_tab, neg_tab = st.tabs([
            f"👍 긍정 ({len(pos_df)}건)",
            f"👎 부정 ({len(neg_df)}건)",
        ])

        with pos_tab:
            if pos_df.empty:
                st.info("긍정 문장이 없습니다.")
            else:
                ev_col = "evidence" if "evidence" in pos_df.columns else "sentence_en"
                for _, row in pos_df.iterrows():
                    evidence = row.get(ev_col, "") or ""
                    st.markdown(
                        f"<div style='background:#f0fdf4;border-left:4px solid #22c55e;"
                        f"padding:10px 14px;border-radius:6px;margin-bottom:8px'>"
                        f"✅ {evidence}</div>",
                        unsafe_allow_html=True,
                    )

        with neg_tab:
            if neg_df.empty:
                st.info("부정 문장이 없습니다.")
            else:
                ev_col = "evidence" if "evidence" in neg_df.columns else "sentence_en"
                for _, row in neg_df.iterrows():
                    evidence = row.get(ev_col, "") or ""
                    st.markdown(
                        f"<div style='background:#fef2f2;border-left:4px solid #ef4444;"
                        f"padding:10px 14px;border-radius:6px;margin-bottom:8px'>"
                        f"❌ {evidence}</div>",
                        unsafe_allow_html=True,
                    )

st.divider()

# ══════════════════════════════════════════════════════════
# 섹션 3: AI 이슈 진단 및 실행 처방전
# ══════════════════════════════════════════════════════════
st.subheader("🤖 AI 이슈 진단 및 실행 처방전")
st.caption("스코어 하락 속성 + 해당 관련 부정 리뷰 → 하락 원인 분석 요약 + 운영 개선 가이드라인")

# API 키 입력
with st.expander("🔑 OpenAI API 키 설정", expanded="openai_api_key" not in st.session_state):
    api_key_input = st.text_input(
        "API 키를 입력하세요",
        type="password",
        placeholder="sk-...",
        help="OpenAI Platform(platform.openai.com)에서 발급한 API 키를 입력하세요.",
    )
    if st.button("저장", key="save_api_key"):
        if api_key_input.startswith("sk-"):
            st.session_state["openai_api_key"] = api_key_input
            st.success("✅ API 키가 저장됐습니다.")
        else:
            st.error("올바른 API 키 형식이 아닙니다. sk- 로 시작해야 합니다.")

# 분석 대상 속성: 낮은 점수 순으로 정렬
weak_cats = sorted(
    [cat for cat in CATEGORIES if pd.notna(prod.get(f"{cat}_score"))],
    key=lambda c: prod.get(f"{cat}_score", 10),
)

diag_col, result_col = st.columns([1, 2], gap="large")

with diag_col:
    st.markdown("**분석 설정**")
    sel_cat = st.selectbox(
        "진단할 속성 선택",
        weak_cats,
        format_func=lambda c: f"{CAT_EMOJI[c]} {CAT_KR[c]} ({prod.get(f'{c}_score', 0):.1f}점)",
    )
    neg_count_limit = st.slider("분석할 부정 리뷰 수", 3, 15, 7)

    # 선택 속성 현황
    sc = prod.get(f"{sel_cat}_score")
    pos_n = int(prod.get(f"{sel_cat}_pos_count", 0) or 0)
    neg_n = int(prod.get(f"{sel_cat}_neg_count", 0) or 0)
    icon  = "🔴" if sc < 4 else ("🟡" if sc < 7 else "🟢")
    st.markdown(f"""
    <div style='background:#f8fafc;border-radius:10px;padding:.8rem 1rem;margin-top:.5rem;
                border-left:4px solid {"#ef4444" if sc<4 else "#f59e0b" if sc<7 else "#10b981"}'>
      <div style='font-size:.8rem;font-weight:700;color:#1e1b4b;'>
        {CAT_EMOJI[sel_cat]} {CAT_KR[sel_cat]} 현황</div>
      <div style='font-size:1.4rem;font-weight:900;color:{"#dc2626" if sc<4 else "#d97706" if sc<7 else "#16a34a"};'>
        {icon} {sc:.1f}점</div>
      <div style='font-size:.75rem;color:#6b7280;margin-top:3px;'>
        👍 긍정 {pos_n}건 &nbsp; 👎 부정 {neg_n}건</div>
    </div>""", unsafe_allow_html=True)

    run_btn = st.button(
        f"🔍 {CAT_KR[sel_cat]} 진단 실행",
        use_container_width=True,
        type="primary",
        disabled="openai_api_key" not in st.session_state,
    )
    if "openai_api_key" not in st.session_state:
        st.caption("⚠️ API 키를 먼저 저장하세요.")

with result_col:
    result_key = f"diag_{selected_asin}_{sel_cat}"

    if run_btn and "openai_api_key" in st.session_state:
        # 부정 문장 수집
        if not prod_sent.empty:
            neg_sents = prod_sent[
                (prod_sent["category"] == sel_cat) &
                (prod_sent["sentiment"] == "negative")
            ]
            ev_col_name = "evidence" if "evidence" in neg_sents.columns else "sentence_en"
            neg_texts = neg_sents[ev_col_name].dropna().tolist()[:neg_count_limit]
        else:
            neg_texts = []

        if not neg_texts:
            st.warning("분석할 부정 문장 데이터가 없습니다.")
        else:
            review_block = "\n".join(f"- {t}" for t in neg_texts)
            prompt = f"""당신은 이커머스 신발 판매자를 위한 상품 품질 분석 전문가입니다.
아래는 '{prod["product_title"]}'의 [{CAT_KR[sel_cat]}] 속성에 대한 고객 부정 리뷰입니다.
현재 [{CAT_KR[sel_cat]}] 만족도 점수는 {sc:.1f}/10점입니다.

[부정 리뷰]
{review_block}

다음 형식으로 한국어로 분석해주세요:

🔍 진단 요약
(반복되는 문제 패턴과 고객 불만의 핵심을 2~3문장으로 요약)

⚡ 가장 시급한 문제
(단 하나의 가장 중요한 문제를 한 문장으로)

💊 운영 개선 가이드라인
1. (즉시 조치 가능한 개선안)
2. (중기적 개선안)
3. (장기적 개선안)

📊 예상 효과
(개선 시 기대할 수 있는 구체적 효과)"""

            with st.spinner(f"GPT-5.4 mini가 {CAT_KR[sel_cat]} 속성을 진단 중..."):
                try:
                    client   = OpenAI(api_key=st.session_state["openai_api_key"])
                    response = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "당신은 이커머스 상품 품질 분석 전문가입니다. 판매자가 즉시 실행할 수 있는 구체적이고 실용적인 분석을 제공합니다."},
                            {"role": "user", "content": prompt},
                        ],
                        max_completion_tokens=800,
                    )
                    result_text = response.choices[0].message.content
                    # 토큰 사용량 계산
                    usage       = response.usage
                    cost_input  = usage.prompt_tokens / 1_000_000 * 0.75
                    cost_output = usage.completion_tokens / 1_000_000 * 4.50
                    cost_total  = cost_input + cost_output
                    st.session_state[result_key] = {
                        "text":   result_text,
                        "tokens": usage.total_tokens,
                        "cost":   cost_total,
                    }
                except Exception as e:
                    st.error(f"API 호출 실패: {e}")

    if result_key in st.session_state:
        r = st.session_state[result_key]
        st.markdown(f"""
        <div style='background:white;border-radius:14px;padding:1.2rem 1.5rem;
                    box-shadow:0 2px 12px rgba(99,102,241,.1);border-top:4px solid #6366f1;'>
          <div style='font-size:.7rem;color:#9ca3af;font-weight:700;letter-spacing:.06em;margin-bottom:.6rem;'>
            AI 이슈 진단 결과 — {CAT_EMOJI[sel_cat]} {CAT_KR[sel_cat]} | {OPENAI_MODEL}
          </div>
          <div style='font-size:.88rem;color:#1e1b4b;line-height:1.9;white-space:pre-wrap;'>
{r["text"]}
          </div>
          <div style='font-size:.7rem;color:#9ca3af;margin-top:.8rem;border-top:1px solid #f3f4f6;padding-top:.5rem;'>
            토큰: {r["tokens"]:,}개 &nbsp;|&nbsp; 예상 비용: ${r["cost"]:.5f}
          </div>
        </div>""", unsafe_allow_html=True)
    elif not run_btn:
        st.markdown("""
        <div style='background:#f8fafc;border-radius:14px;padding:2rem;text-align:center;color:#9ca3af;'>
          <div style='font-size:2rem;margin-bottom:.5rem;'>🤖</div>
          <div style='font-size:.85rem;'>왼쪽에서 속성을 선택하고<br>진단 실행 버튼을 누르세요.</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════
# 섹션 3: 전체 데이터 테이블
# ══════════════════════════════════════════════════════════
with st.expander("📋 전체 데이터 보기"):
    show_cols = ["brand", "product_title", "avg_rating", "review_count"] + \
                [f"{cat}_score" for cat in CATEGORIES]
    show_df = df[show_cols].copy()
    show_df.columns = (
        ["브랜드", "상품명", "평균 별점", "리뷰 수"] +
        [f"{CAT_KR[cat]} 점수" for cat in CATEGORIES]
    )
    show_df["상품명"] = show_df["상품명"].str[:50]
    st.dataframe(show_df.reset_index(drop=True), use_container_width=True)