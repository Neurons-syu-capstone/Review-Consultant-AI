"""
뉴런스 팀 | LLM 기반 신발 리뷰 이슈 탐지 대시보드
방법론: LLM 이슈 분석 + VADER Sentiment + LDA Topic Modeling

필요 파일:
  - shoes_data/issue_report.json       (issue_pipeline.py 결과)
  - shoes_data/shoes_reviews_cleaned.parquet  또는
    shoes_reviews_cleaned.json          (원본 리뷰)

nlp_cache.pkl 불필요 — 실행 시 자동으로 VADER/LDA 계산 후 캐시
"""

import streamlit as st
import pandas as pd
import numpy as np
import re, json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

# ═══════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="리뷰 인사이트 대시보드 | 뉴런스 팀",
    page_icon="👟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════
ASPECTS     = ['comfort', 'fit_size', 'durability', 'design', 'price']
ASPECT_KO   = {'comfort':'착용감','fit_size':'사이즈','durability':'내구성',
                'design':'디자인','price':'가격'}
ASPECT_ICON = {'comfort':'👟','fit_size':'📏','durability':'🔩',
                'design':'🎨','price':'💰'}
ASPECT_KW   = {
    'comfort':    ['comfort','comfortable','comfy','cushion','soft','padded',
                   'arch support','support','pain','hurt','blister','sore','cozy','plush'],
    'fit_size':   ['fit','size','sizing','small','large','wide','narrow','tight',
                   'loose','true to size','length','width','snug','roomy'],
    'durability': ['durable','durability','last','lasting','worn out','broke','broken',
                   'fall apart','fell apart','sole','quality','cheap','sturdy','weak','hold up'],
    'design':     ['design','style','stylish','look','looks','beautiful','ugly','color',
                   'colour','appearance','cute','pretty','fashionable','attractive'],
    'price':      ['price','expensive','worth','value','cost','affordable',
                   'overpriced','money','dollar','budget','deal'],
}
STOP_WORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'is','it','this','that','was','are','be','have','has','had','as','from',
    'they','we','you','i','my','me','he','she','his','her','its','our','their',
    'not','no','so','if','by','do','get','got','just','can','will','would',
    'could','should','very','too','also','more','one','two','all','been',
    'than','then','when','what','which','there','these','those','them','us',
    'shoe','shoes','boot','boots','pair','wear','wearing','wore','worn',
])
BRAND_COLOR = {
    'Skechers':'#6366f1','adidas':'#f59e0b','Nike':'#ef4444',
    'ASICS':'#10b981','Clarks':'#3b82f6','Crocs':'#ec4899',
    'KEEN':'#8b5cf6','Merrell':'#14b8a6','PUMA':'#f97316','NINE WEST':'#64748b',
}
SEV_COLOR = {'high':'#ef4444','medium':'#f59e0b','low':'#10b981'}
SEV_KO    = {'high':'높음','medium':'중간','low':'낮음'}

def score_color(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return '#9ca3af'
    return '#16a34a' if s >= 7.0 else ('#d97706' if s >= 6.0 else '#dc2626')
def score_bg(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return '#f3f4f6'
    return '#dcfce7' if s >= 7.0 else ('#fef9c3' if s >= 6.0 else '#fee2e2')
def score_label(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return '데이터 없음'
    return '높음 ▲' if s >= 7.0 else ('보통 →' if s >= 6.0 else '낮음 ▼')

# ═══════════════════════════════════════════════════════
#  데이터 로드 (nlp_cache.pkl 불필요)
# ═══════════════════════════════════════════════════════
@st.cache_resource
def load_raw() -> pd.DataFrame:
    """리뷰 원본 데이터 로드 (parquet 우선, 없으면 json)"""
    parquet = "shoes_data/shoes_reviews_cleaned.parquet"
    json_f  = "shoes_reviews_cleaned.json"
    import os
    if os.path.exists(parquet):
        df = pd.read_parquet(parquet)
    elif os.path.exists(json_f):
        with open(json_f, "r", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
    else:
        st.error("리뷰 데이터 파일을 찾을 수 없습니다.\n"
                 "shoes_data/shoes_reviews_cleaned.parquet 또는\n"
                 "shoes_reviews_cleaned.json 이 필요합니다.")
        st.stop()
    df = df.dropna(subset=["text","rating"]).copy()
    df["text"]   = df["text"].astype(str).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    elif "year" not in df.columns:
        df["year"] = 2020
    return df


@st.cache_resource
def load_issue_report() -> dict:
    try:
        with open("shoes_data/issue_report.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_resource
def build_product_pivot(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    nlp_cache의 product_pivot을 대체.
    VADER로 속성별 점수를 계산하여 상품별 피벗 테이블 생성.
    처음 한 번만 실행되고 이후는 캐시에서 반환.
    """
    analyzer = SentimentIntensityAnalyzer()
    records  = {}

    for asin, grp in raw_df.groupby("parent_asin"):
        brand = grp["brand"].iloc[0] if "brand" in grp.columns else "Unknown"
        title = grp["product_title"].iloc[0] if "product_title" in grp.columns else asin
        avg_r = grp["rating"].mean()
        rv_cnt= len(grp)

        asp_scores   = {}
        asp_mentions = {}
        asp_compound = {}

        for asp, kws in ASPECT_KW.items():
            sents = []
            for txt in grp["text"]:
                for sent in re.split(r"[.!?]+", str(txt)):
                    sent = sent.strip()
                    if any(kw in sent.lower() for kw in kws) and len(sent) > 20:
                        sents.append(sent)
            if sents:
                compounds = [analyzer.polarity_scores(s)["compound"] for s in sents]
                avg_c = float(np.mean(compounds))
                asp_scores[asp]   = round((avg_c + 1) / 2 * 10, 2)
                asp_mentions[asp] = len(sents)
                asp_compound[asp] = avg_c
            else:
                asp_scores[asp]   = None
                asp_mentions[asp] = 0
                asp_compound[asp] = None

        valid = [v for v in asp_scores.values() if v is not None]
        overall = round(float(np.mean(valid)), 2) if valid else None

        rec = {
            "parent_asin":   asin,
            "brand":         brand,
            "product_title": title,
            "avg_rating":    round(avg_r, 2),
            "review_count":  rv_cnt,
            "overall_score": overall,
        }
        for asp in ASPECTS:
            rec[asp]                  = asp_scores[asp]
            rec[f"{asp}_mentions"]    = asp_mentions[asp]
            rec[f"{asp}_compound"]    = asp_compound[asp]

        records[asin] = rec

    return pd.DataFrame(list(records.values()))


@st.cache_data
def get_vader_stats(asin: str, raw_df: pd.DataFrame) -> dict:
    """상품 1개의 VADER 통계 (yearly_rating, yearly_compound, rating_dist 등)"""
    analyzer = SentimentIntensityAnalyzer()
    grp = raw_df[raw_df["parent_asin"] == asin].copy()
    if grp.empty:
        return {}

    compounds = grp["text"].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    pos_mask  = compounds >= 0.05
    neg_mask  = compounds <= -0.05

    yearly_rating   = grp.groupby("year")["rating"].mean().round(3).to_dict()
    yearly_compound = grp.assign(compound=compounds).groupby("year")["compound"].mean().round(4).to_dict()
    rating_dist     = grp["rating"].value_counts().to_dict()

    return {
        "pos_n":           int(pos_mask.sum()),
        "neg_n":           int(neg_mask.sum()),
        "neu_n":           int((~pos_mask & ~neg_mask).sum()),
        "pos_pct":         round(pos_mask.mean() * 100, 1),
        "neg_pct":         round(neg_mask.mean() * 100, 1),
        "avg_compound":    round(float(compounds.mean()), 4),
        "yearly_rating":   {int(k): round(v, 2) for k, v in yearly_rating.items()},
        "yearly_compound": {int(k): round(v, 4) for k, v in yearly_compound.items()},
        "rating_dist":     {int(k): int(v) for k, v in rating_dist.items()},
    }


@st.cache_data
def get_clean_texts(asin: str, raw_df: pd.DataFrame) -> dict:
    """LDA용 긍/부정 정제 텍스트 (단어 리스트)"""
    analyzer = SentimentIntensityAnalyzer()
    grp = raw_df[raw_df["parent_asin"] == asin]
    if grp.empty:
        return {"pos_clean": [], "neg_clean": []}

    def clean(text):
        words = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower()).split()
        return " ".join(w for w in words if w not in STOP_WORDS and len(w) > 2)

    pos_texts, neg_texts = [], []
    for txt in grp["text"]:
        c = analyzer.polarity_scores(str(txt))["compound"]
        cleaned = clean(txt)
        if cleaned:
            if c >= 0.05:  pos_texts.append(cleaned)
            elif c <= -0.05: neg_texts.append(cleaned)

    return {"pos_clean": pos_texts, "neg_clean": neg_texts}


@st.cache_data
def get_aspect_reviews(asin: str, raw_df: pd.DataFrame, aspect: str) -> pd.DataFrame:
    """속성별 관련 문장 + VADER 점수"""
    analyzer = SentimentIntensityAnalyzer()
    kws      = ASPECT_KW[aspect]
    grp      = raw_df[raw_df["parent_asin"] == asin]
    rows = []
    for _, r in grp.iterrows():
        for sent in re.split(r"[.!?]+", str(r["text"])):
            sent = sent.strip()
            if any(kw in sent.lower() for kw in kws) and len(sent) > 20:
                c = analyzer.polarity_scores(sent)["compound"]
                rows.append({"text": sent, "compound": c,
                             "rating": r["rating"], "year": r.get("year", 2020)})
    return pd.DataFrame(rows)


def compute_lda(pos_texts, neg_texts, n_pos=3, n_neg=3):
    def _lda(texts, n):
        if len(texts) < 8: return {}
        try:
            vec = CountVectorizer(max_df=0.92, min_df=2, max_features=250, ngram_range=(1,2))
            dtm = vec.fit_transform(texts)
            if dtm.shape[1] < n: return {}
            lda = LatentDirichletAllocation(n_components=n, random_state=42,
                                             max_iter=15, learning_method='batch')
            lda.fit(dtm)
            feat = vec.get_feature_names_out()
            return {i: [feat[j] for j in comp.argsort()[-8:][::-1]]
                    for i, comp in enumerate(lda.components_)}
        except:
            return {}
    return _lda(pos_texts, n_pos), _lda(neg_texts, n_neg)


# ═══════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════
def main():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700;900&display=swap');
    html, body, [data-testid], .stMarkdown, p, div, span {
        font-family: 'Noto Sans KR', sans-serif !important;
    }
    .main .block-container{padding:1.2rem 2rem 2rem;max-width:1500px;}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0c29 0%,#302b63 60%,#24243e 100%);}
    [data-testid="stSidebar"] *{color:#e0e7ff!important;}
    [data-testid="stSidebar"] label{color:#a5b4fc!important;font-size:.78rem;font-weight:600;letter-spacing:.04em;}
    /* 탭 텍스트 삐져나옴 방지 */
    .stTabs [data-baseweb="tab-list"]{gap:4px;background:#f1f5f9;border-radius:12px;padding:4px;flex-wrap:nowrap;overflow-x:auto;}
    .stTabs [data-baseweb="tab"]{border-radius:8px;padding:6px 10px;font-weight:600;color:#64748b;
        font-family:'Noto Sans KR',sans-serif!important;white-space:nowrap;font-size:.80rem;flex-shrink:0;}
    .stTabs [aria-selected="true"]{background:#4f46e5!important;color:white!important;}
    .sec{font-size:.95rem;font-weight:800;color:#1e1b4b;
         border-left:4px solid #6366f1;padding-left:10px;margin:1.2rem 0 .7rem;}
    /* 속성 카드 5번째 잘림 방지: minmax(0,1fr) */
    .asp-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:.6rem 0 1rem;overflow:visible;}
    .asp-card{background:white;border-radius:14px;padding:.8rem .4rem;text-align:center;
              box-shadow:0 2px 12px rgba(0,0,0,.07);transition:transform .15s;min-width:0;overflow:hidden;}
    .asp-card:hover{transform:translateY(-3px);}
    .asp-icon{font-size:1.3rem;margin-bottom:3px;}
    .asp-name{font-size:.66rem;color:#6b7280;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
    .asp-score{font-size:1.85rem;font-weight:900;line-height:1.1;}
    .asp-mention{font-size:.63rem;color:#9ca3af;margin-top:3px;}
    .asp-lv{display:inline-block;padding:2px 7px;border-radius:20px;font-size:.63rem;font-weight:700;margin-top:4px;}
    .kpi{background:white;border-radius:14px;padding:1rem 1.1rem;
         box-shadow:0 2px 12px rgba(99,102,241,.09);border-top:4px solid #6366f1;}
    .kpi-lbl{font-size:.68rem;color:#9ca3af;font-weight:700;letter-spacing:.05em;}
    .kpi-val{font-size:1.75rem;font-weight:900;color:#1e1b4b;line-height:1.1;margin:2px 0;}
    .kpi-sub{font-size:.7rem;color:#9ca3af;}
    .issue-card{background:white;border-radius:12px;padding:.9rem 1.1rem;
                margin-bottom:8px;box-shadow:0 1px 8px rgba(0,0,0,.06);}
    .pr-row{display:flex;align-items:center;gap:12px;background:white;border-radius:10px;
            padding:.7rem 1rem;margin-bottom:7px;box-shadow:0 1px 8px rgba(0,0,0,.06);}
    .topic-chip{display:inline-block;padding:3px 10px;border-radius:20px;
                font-size:.75rem;font-weight:600;margin:2px;background:#ede9fe;color:#4f46e5;}
    .topic-chip-neg{background:#fee2e2;color:#dc2626;}
    .rv-card{background:white;border-radius:10px;padding:.7rem 1rem;
             margin-bottom:6px;box-shadow:0 1px 8px rgba(0,0,0,.05);font-size:.82rem;line-height:1.6;}
    .critical-banner{background:linear-gradient(135deg,#fef2f2,#fee2e2);
                     border-radius:14px;padding:1rem 1.3rem;margin:.5rem 0 1rem;
                     border-left:5px solid #ef4444;}
    /* 우측 상단 메뉴 글씨 겹침 방지 */
    [data-testid="stMainMenu"] button span,[data-testid="stMainMenu"] li,
    [data-testid="stMainMenu"] div{font-family:sans-serif!important;font-size:14px!important;line-height:1.5!important;}
    /* 랭킹 테이블 글씨 깨짐 방지 */
    .stDataFrame,.stDataFrame *,[data-testid="stDataFrameResizable"] *{
        font-family:'Noto Sans KR',sans-serif!important;font-size:.82rem!important;}
    </style>
    """, unsafe_allow_html=True)

    # ── 데이터 로드
    raw_df       = load_raw()
    issue_report = load_issue_report()

    # product_pivot 빌드 (첫 실행 시 약 1~3분 소요, 이후 캐시)
    with st.spinner("📊 상품 데이터 분석 중... (첫 실행 시 1~3분 소요)"):
        pp = build_product_pivot(raw_df)

    BRANDS = sorted(pp["brand"].dropna().unique().tolist())

    # ═══════════════════════════════════════════════════════
    #  SIDEBAR
    # ═══════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 👟 리뷰 인사이트 대시보드")
        st.markdown("---")
        sel_brand   = st.selectbox("🏷️ 브랜드", ["전체"] + BRANDS)
        min_rev     = st.slider("📊 최소 리뷰 수", 10, 300, 50, 10)
        min_rat     = st.slider("⭐ 최소 평균 별점", 1.0, 5.0, 1.0, 0.5)
        sel_aspects = st.multiselect("🔍 분석 속성", ASPECTS, default=ASPECTS,
                                      format_func=lambda x: f"{ASPECT_ICON[x]} {ASPECT_KO[x]}")
        sort_opt    = st.selectbox("📐 정렬 기준",
                                   ["종합 점수↓","리뷰 수↓","별점↓","착용감↓",
                                    "사이즈↓","내구성↓","디자인↓","가격↓"])
        sort_col = {
            "종합 점수↓":"overall_score","리뷰 수↓":"review_count","별점↓":"avg_rating",
            "착용감↓":"comfort","사이즈↓":"fit_size","내구성↓":"durability",
            "디자인↓":"design","가격↓":"price",
        }[sort_opt]
        if not sel_aspects: sel_aspects = ASPECTS
        st.markdown("---")
        st.markdown("""
        <div style='font-size:.72rem;color:#a5b4fc;line-height:1.8;'>
        🔬 <b>분석 방법론</b><br>
        ① VADER Sentiment Analysis<br>
        &nbsp;&nbsp;— 리뷰 전체 감성 스코어<br>
        ② LDA Topic Modeling<br>
        &nbsp;&nbsp;— 긍/부정 리뷰 핵심 토픽<br>
        ③ LLM 이슈 분석<br>
        &nbsp;&nbsp;— 속성별 이슈·심각도·개선 제안<br>
        ④ 속성별 키워드 매칭<br>
        &nbsp;&nbsp;— 5개 속성 문장 추출<br>
        </div>""", unsafe_allow_html=True)

    # ── 필터 & 정렬
    filt = pp.copy()
    if sel_brand != "전체":
        filt = filt[filt["brand"] == sel_brand]
    filt = filt[filt["review_count"] >= min_rev]
    filt = filt[filt["avg_rating"]   >= min_rat]
    filt = filt.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)

    # ── 헤더
    _bc  = BRAND_COLOR.get(sel_brand, "#6366f1")
    _dot = (f"<span style='display:inline-block;width:11px;height:11px;border-radius:50%;"
            f"background:{_bc};margin-right:7px;vertical-align:middle;'></span>"
            if sel_brand != "전체" else "")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f0c29,#302b63 55%,#1e1b4b);
        border-radius:18px;padding:1.4rem 2rem 1.2rem;margin-bottom:1rem;'>
      <div style='color:#a5b4fc;font-size:.72rem;font-weight:700;letter-spacing:.1em;'>
        REVIEW INSIGHT &nbsp;·&nbsp; NLP ANALYSIS &nbsp;·&nbsp; ASPECT SCOREBOARD
      </div>
      <div style='color:white;font-size:1.6rem;font-weight:900;margin:4px 0 3px;'>
        {_dot}{'전체 브랜드' if sel_brand=='전체' else sel_brand} — 다차원 만족도 스코어보드
      </div>
      <div style='color:#818cf8;font-size:.82rem;'>
        VADER 감성 분석 + LDA 토픽 모델링 + LLM 이슈 분석 &nbsp;|&nbsp;
        착용감·사이즈·내구성·디자인·가격 속성 점수화
        &nbsp;|&nbsp; 분석 상품 <b style='color:#c7d2fe;'>{len(filt):,}개</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    search_q = st.text_input("", placeholder="🔍  상품명 검색  (예: Nike Air, Merrell Moab, Crocs...)",
                              label_visibility="collapsed")
    if search_q:
        filt = filt[filt["product_title"].str.contains(search_q, case=False, na=False)]

    if filt.empty:
        st.warning("조건에 맞는 상품이 없습니다. 필터를 조정해 보세요.")
        st.stop()

    # ── KPI
    k_weak = filt[sel_aspects].mean().idxmin()
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val, sub, bc in [
        (c1, "분석 상품",  f"{len(filt):,}",                  "개",    "#6366f1"),
        (c2, "총 리뷰",   f"{int(filt['review_count'].sum()):,}", "건", "#8b5cf6"),
        (c3, "평균 별점", f"⭐ {filt['avg_rating'].mean():.2f}", "/ 5.0","#f59e0b"),
        (c4, "평균 종합", f"{filt['overall_score'].mean():.2f}", "/ 10", "#10b981"),
        (c5, "최약 속성", f"{ASPECT_ICON[k_weak]} {ASPECT_KO[k_weak]}",
             f"평균 {filt[k_weak].mean():.2f}", "#ef4444"),
    ]:
        col.markdown(f"""
        <div class="kpi" style="border-top-color:{bc};">
          <div class="kpi-lbl">{lbl}</div>
          <div class="kpi-val">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 상품 선택
    sel_title = st.selectbox(
        "📦  상세 분석할 상품을 선택하세요",
        filt["product_title"].tolist(),
        format_func=lambda x: x[:80] + ("…" if len(x) > 80 else ""),
    )
    prod = filt[filt["product_title"] == sel_title].iloc[0]

    # ── 선택 상품 VADER 통계 & LDA 텍스트 (상품별 캐시)
    with st.spinner("리뷰 분석 중..."):
        vs         = get_vader_stats(prod["parent_asin"], raw_df)
        clean_data = get_clean_texts(prod["parent_asin"], raw_df)

    # ── issue_report
    ir           = issue_report.get(prod["parent_asin"], {})
    ir_anal      = ir.get("analysis", {})
    ir_issues    = ir_anal.get("top_issues", [])
    ir_critical  = ir_anal.get("critical_issue", "")
    ir_sentiment = ir_anal.get("overall_sentiment", "")

    # ── 상품 헤더
    _pbc = BRAND_COLOR.get(prod["brand"], "#6366f1")
    overall_avg = pp[sel_aspects].mean()
    brand_avg   = pp[pp["brand"] == prod["brand"]][sel_aspects].mean()

    st.markdown(f"""
    <div style='background:white;border-radius:14px;padding:1rem 1.5rem;
        box-shadow:0 2px 14px rgba(99,102,241,.1);margin:1rem 0;
        border-left:5px solid {_pbc};'>
      <div style='font-size:.7rem;color:#9ca3af;font-weight:700;letter-spacing:.06em;'>선택 상품</div>
      <div style='font-size:1.15rem;font-weight:800;color:#1e1b4b;margin:3px 0 5px;'>
        {prod["product_title"]}
      </div>
      <span style='font-size:.8rem;color:#6b7280;'>🏷️ {prod["brand"]}</span>
      &nbsp;·&nbsp;
      <span style='font-size:.8rem;color:#6b7280;'>📝 {int(prod["review_count"]):,}건</span>
      &nbsp;·&nbsp;
      <span style='font-size:.8rem;color:#f59e0b;font-weight:700;'>⭐ {prod["avg_rating"]:.2f} / 5.0</span>
      &nbsp;·&nbsp;
      <span style='font-size:.8rem;color:#10b981;font-weight:700;'>종합점수 {prod["overall_score"]:.2f} / 10.0</span>
    </div>
    """, unsafe_allow_html=True)

    # ── 가장 시급한 이슈 배너 (한/영 토글)
    if ir_critical and ir_critical != "N/A":
        banner_col, toggle_col = st.columns([6, 1])
        with toggle_col:
            show_kr = st.toggle("🇰🇷 한국어", value=False, key="critical_lang")

        # 한국어 번역 (session_state 기반 간단 캐시)
        cache_key = f"kr_{prod['parent_asin']}"
        if show_kr:
            if cache_key not in st.session_state:
                try:
                    from deep_translator import GoogleTranslator
                    t_critical  = GoogleTranslator(source="en", target="ko").translate(ir_critical)
                    t_sentiment = GoogleTranslator(source="en", target="ko").translate(ir_sentiment)
                    st.session_state[cache_key] = (t_critical, t_sentiment)
                except Exception:
                    st.session_state[cache_key] = (ir_critical + " (번역 실패 — deep_translator 설치 필요)", ir_sentiment)
            disp_critical, disp_sentiment = st.session_state[cache_key]
        else:
            disp_critical, disp_sentiment = ir_critical, ir_sentiment

        with banner_col:
            st.markdown(f"""
            <div class="critical-banner">
              <div style='font-size:.75rem;font-weight:700;color:#991b1b;letter-spacing:.05em;'>
                ⚡ 가장 시급한 이슈 {"🇰🇷" if show_kr else "🇺🇸"}
              </div>
              <div style='font-size:1rem;font-weight:800;color:#7f1d1d;margin:4px 0 2px;'>
                {disp_critical}
              </div>
              <div style='font-size:.78rem;color:#b91c1c;'>{disp_sentiment}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── 속성 점수 카드
    st.markdown('<div class="sec">📊 속성별 만족도 점수</div>', unsafe_allow_html=True)
    html = '<div class="asp-grid">'
    for asp in sel_aspects:
        sc  = prod.get(asp)
        mn  = int(prod.get(f"{asp}_mentions", 0))
        clr = score_color(sc)
        bg  = score_bg(sc)
        lv  = score_label(sc)
        html += f"""
        <div class="asp-card" style="background:{bg};border-bottom:4px solid {clr};">
          <div class="asp-icon">{ASPECT_ICON[asp]}</div>
          <div class="asp-name">{ASPECT_KO[asp]}</div>
          <div class="asp-score" style="color:{clr};">
            {"N/A" if sc is None or (isinstance(sc,float) and np.isnan(sc)) else f"{sc:.1f}"}
          </div>
          <div class="asp-mention">{mn:,}건 언급</div>
          <span class="asp-lv" style="background:{bg};color:{clr};">{lv}</span>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    #  TABS
    # ═══════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕸️  레이더 차트",
        "🔤  LDA 토픽 분석",
        "📈  트렌드 & 분포",
        "🎯  이슈 & 개선 우선순위",
        "📋  속성별 리뷰",
    ])

    # ── TAB 1: 레이더 차트
    with tab1:
        col_L, col_R = st.columns([1, 1], gap="large")
        with col_L:
            st.markdown('<div class="sec">속성별 만족도 레이더 차트</div>', unsafe_allow_html=True)
            asp_labels   = [f"{ASPECT_ICON[a]} {ASPECT_KO[a]}" for a in sel_aspects]
            scores_prod  = [float(prod[a]) if pd.notna(prod.get(a)) else 5.0 for a in sel_aspects]
            scores_brand = [float(brand_avg[a]) if pd.notna(brand_avg.get(a)) else 5.0 for a in sel_aspects]
            scores_all   = [float(overall_avg[a]) if pd.notna(overall_avg.get(a)) else 5.0 for a in sel_aspects]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=scores_all+[scores_all[0]], theta=asp_labels+[asp_labels[0]],
                fill='toself', name='전체 평균',
                line=dict(color='#d1d5db', width=1.2, dash='dot'),
                fillcolor='rgba(209,213,219,0.12)',
            ))
            fig.add_trace(go.Scatterpolar(
                r=scores_brand+[scores_brand[0]], theta=asp_labels+[asp_labels[0]],
                fill='toself', name=f'{prod["brand"]} 평균',
                line=dict(color=_pbc, width=1.8, dash='dash'),
                fillcolor='rgba(99,102,241,0.07)',
            ))
            fig.add_trace(go.Scatterpolar(
                r=scores_prod+[scores_prod[0]], theta=asp_labels+[asp_labels[0]],
                fill='toself', name='선택 상품',
                line=dict(color='#4f46e5', width=3.2),
                fillcolor='rgba(79,70,229,0.22)',
                marker=dict(size=9, color='#4f46e5'),
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[3,10], tickfont_size=9,
                                    gridcolor='#e5e7eb', linecolor='#e5e7eb'),
                    angularaxis=dict(tickfont_size=12, gridcolor='#e5e7eb'),
                    bgcolor='#f9fafb',
                ),
                showlegend=True,
                legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center', font_size=11),
                margin=dict(l=30,r=30,t=30,b=80), height=420, paper_bgcolor='white',
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("🔵 선택 상품  |  점선: 브랜드 평균  |  회색: 전체 평균")

        with col_R:
            st.markdown('<div class="sec">전체 평균 대비 Gap 분석</div>', unsafe_allow_html=True)
            gap_rows = []
            for asp in sel_aspects:
                sc = prod.get(asp)
                av = float(overall_avg[asp]) if pd.notna(overall_avg.get(asp)) else 0
                if sc is not None and not np.isnan(float(sc)):
                    gap_rows.append({'aspect': f"{ASPECT_ICON[asp]} {ASPECT_KO[asp]}",
                                     'score': float(sc), 'avg': av,
                                     'gap': round(float(sc)-av, 2)})
            gap_df = pd.DataFrame(gap_rows).sort_values('gap')
            fig_gap = go.Figure(go.Bar(
                x=gap_df['gap'], y=gap_df['aspect'], orientation='h',
                marker_color=['#10b981' if g>=0 else '#ef4444' for g in gap_df['gap']],
                text=[f"{'+' if g>=0 else ''}{g:.2f}" for g in gap_df['gap']],
                textposition='outside',
                textfont=dict(size=12, color=['#16a34a' if g>=0 else '#dc2626' for g in gap_df['gap']]),
                customdata=list(zip(gap_df['score'], gap_df['avg'])),
                hovertemplate="<b>%{y}</b><br>이 상품: %{customdata[0]:.2f}<br>전체평균: %{customdata[1]:.2f}<br>Gap: %{x:.2f}<extra></extra>",
            ))
            fig_gap.add_vline(x=0, line_color='#374151', line_width=1.5)
            fig_gap.update_layout(
                xaxis=dict(title='전체 평균 대비 Gap', gridcolor='#f3f4f6', zeroline=False),
                yaxis=dict(tickfont_size=12),
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=10,r=80,t=10,b=40), height=250,
            )
            st.plotly_chart(fig_gap, use_container_width=True)

            st.markdown('<div class="sec" style="margin-top:.5rem;">속성별 긍정/부정 비율</div>',
                        unsafe_allow_html=True)
            for asp in sel_aspects:
                asp_c = prod.get(f'{asp}_compound')
                if asp_c is not None and not (isinstance(asp_c, float) and np.isnan(float(asp_c))):
                    asp_c = float(asp_c)
                    pos_p = max(0, min(100, (asp_c+1)/2*100))
                    neg_p = max(0, min(100-pos_p, (1-abs(asp_c))*30+(1-pos_p/100)*40))
                    neg_p = round(neg_p, 1); pos_p = round(pos_p, 1)
                else:
                    pos_p, neg_p = 0.0, 0.0
                neu_p = max(0, 100-pos_p-neg_p)
                label = f"{ASPECT_ICON[asp]} {ASPECT_KO[asp]}"
                st.markdown(f"""
                <div style='margin-bottom:5px;'>
                  <span style='font-size:.78rem;font-weight:600;color:#374151;'>{label}</span>
                  <span style='float:right;font-size:.72rem;color:#6b7280;'>
                    긍정 {pos_p:.0f}% / 부정 {neg_p:.0f}%</span>
                </div>
                <div style='display:flex;height:10px;border-radius:6px;overflow:hidden;margin-bottom:8px;'>
                  <div style='width:{pos_p}%;background:#10b981;'></div>
                  <div style='width:{neu_p}%;background:#e5e7eb;'></div>
                  <div style='width:{neg_p}%;background:#ef4444;'></div>
                </div>""", unsafe_allow_html=True)

    # ── TAB 2: 시맨틱 리뷰 요약 & 키워드 맵
    with tab2:
        st.markdown('<div class="sec">시맨틱 리뷰 요약 & 키워드 맵</div>',
                    unsafe_allow_html=True)
        st.caption("전체 리뷰의 맥락을 3줄로 요약하고, 긍정/부정 핵심 키워드를 클러스터 맵으로 표시합니다.")

        # ── 3줄 요약 ──────────────────────────────────────────
        pos_n_s   = vs.get("pos_n", 0)
        neg_n_s   = vs.get("neg_n", 0)
        avg_cmp_s = vs.get("avg_compound", 0)
        total_s   = pos_n_s + neg_n_s + vs.get("neu_n", 0)
        overall_lbl = ("전반적으로 긍정적" if avg_cmp_s > 0.1
                       else ("전반적으로 부정적" if avg_cmp_s < -0.1 else "혼재된"))
        pos_words_all = " ".join(clean_data.get("pos_clean", []))
        neg_words_all = " ".join(clean_data.get("neg_clean", []))
        top_pos = [w for w,_ in Counter(pos_words_all.split()).most_common(3)] if pos_words_all else []
        top_neg = [w for w,_ in Counter(neg_words_all.split()).most_common(3)] if neg_words_all else []
        pos_kw_str = ", ".join(top_pos) if top_pos else "없음"
        neg_kw_str = ", ".join(top_neg) if top_neg else "없음"
        st.markdown(
            f"""
            <div style='background:#f8fafc;border-radius:14px;padding:1rem 1.4rem;
                        margin-bottom:1.2rem;border:1px solid #e2e8f0;'>
              <div style='font-size:.78rem;font-weight:700;color:#6366f1;
                          letter-spacing:.05em;margin-bottom:.5rem;'>핵심 3줄 요약</div>
              <div style='font-size:.88rem;color:#1e1b4b;line-height:1.9;'>
                📌 {total_s:,}건의 리뷰를 분석한 결과, 이 상품은 <b>{overall_lbl}</b> 평가를 받고 있습니다.<br>
                ✅ 고객들이 주로 언급하는 <b>강점</b>은
                <b style='color:#059669;'>{pos_kw_str}</b> 등입니다.<br>
                ⚠️ <b>개선이 필요한 부분</b>으로는
                <b style='color:#dc2626;'>{neg_kw_str}</b> 등이 자주 언급됩니다.
              </div>
            </div>
            """, unsafe_allow_html=True)

        n_pos_t = st.slider("긍정 토픽 수", 2, 5, 3, key="n_pos")
        n_neg_t = st.slider("부정 토픽 수", 2, 4, 3, key="n_neg")

        with st.spinner("LDA 분석 중..."):
            pos_topics, neg_topics = compute_lda(
                clean_data.get("pos_clean", []),
                clean_data.get("neg_clean", []),
                n_pos_t, n_neg_t,
            )

        col_pos, col_neg = st.columns(2, gap="large")

        with col_pos:
            pos_n   = vs.get('pos_n', 0)
            pos_pct = vs.get('pos_pct', 0)
            st.markdown(f"""
            <div style='background:#f0fdf4;border-radius:12px;padding:.8rem 1rem;margin-bottom:.7rem;
                        border-left:4px solid #10b981;'>
              <div style='font-size:.8rem;font-weight:700;color:#166534;'>
                ✅ 긍정 리뷰 — {pos_n:,}건 ({pos_pct:.1f}%)</div>
              <div style='font-size:.72rem;color:#15803d;margin-top:2px;'>고객이 좋아하는 포인트</div>
            </div>""", unsafe_allow_html=True)
            if pos_topics:
                for t_i, words in pos_topics.items():
                    st.markdown(f"**Topic {t_i+1}**")
                    chips = "".join([f'<span class="topic-chip">{w}</span>' for w in words[:7]])
                    st.markdown(chips, unsafe_allow_html=True)
                    st.markdown("")
            else:
                st.info("긍정 리뷰가 부족하여 토픽을 추출하지 못했습니다.")

            pos_all = " ".join(clean_data.get("pos_clean", []))
            if pos_all:
                wc_pos = Counter(pos_all.split()).most_common(20)
                wdf    = pd.DataFrame(wc_pos, columns=["word","count"])
                max_c  = wdf["count"].max() or 1
                fig_wc = go.Figure(go.Scatter(
                    x=wdf["count"], y=list(range(len(wdf))),
                    mode='markers+text',
                    marker=dict(
                        size=[max(18, int(c/max_c*60)) for c in wdf["count"]],
                        color='#10b981', opacity=0.78,
                        line=dict(width=1.5, color='white'),
                    ),
                    text=wdf["word"],
                    textposition='middle center',
                    textfont=dict(size=10, color="white"),
                    hovertemplate="<b>%{text}</b>: %{x}회<extra></extra>",
))
                fig_wc.update_layout(
                    showlegend=False,
                    xaxis=dict(title='언급 빈도', gridcolor='#f3f4f6'),
                    yaxis=dict(visible=False),
                    plot_bgcolor='white', paper_bgcolor='white',
                    margin=dict(l=10,r=10,t=5,b=30), height=380,
                )
                st.plotly_chart(fig_wc, use_container_width=True)

        with col_neg:
            neg_n   = vs.get('neg_n', 0)
            neg_pct = vs.get('neg_pct', 0)
            st.markdown(f"""
            <div style='background:#fef2f2;border-radius:12px;padding:.8rem 1rem;margin-bottom:.7rem;
                        border-left:4px solid #ef4444;'>
              <div style='font-size:.8rem;font-weight:700;color:#991b1b;'>
                ❌ 부정 리뷰 — {neg_n:,}건 ({neg_pct:.1f}%)</div>
              <div style='font-size:.72rem;color:#b91c1c;margin-top:2px;'>고객이 불만족한 포인트</div>
            </div>""", unsafe_allow_html=True)
            if neg_topics:
                for t_i, words in neg_topics.items():
                    st.markdown(f"**Topic {t_i+1}**")
                    chips = "".join([f'<span class="topic-chip-neg">{w}</span>' for w in words[:7]])
                    st.markdown(chips, unsafe_allow_html=True)
                    st.markdown("")
            else:
                st.info("부정 리뷰가 부족하여 토픽을 추출하지 못했습니다.")

            neg_all = " ".join(clean_data.get("neg_clean", []))
            if neg_all:
                wc_neg = Counter(neg_all.split()).most_common(20)
                wdf2   = pd.DataFrame(wc_neg, columns=["word","count"])
                max_c2 = wdf2["count"].max() or 1
                fig_wc2 = go.Figure(go.Scatter(
                    x=wdf2["count"], y=list(range(len(wdf2))),
                    mode='markers+text',
                    marker=dict(
                        size=[max(18, int(c/max_c2*60)) for c in wdf2["count"]],
                        color='#ef4444', opacity=0.78,
                        line=dict(width=1.5, color='white'),
                    ),
                    text=wdf2["word"],
                    textposition='middle center',
                    textfont=dict(size=10, color="white"),
                    hovertemplate="<b>%{text}</b>: %{x}회<extra></extra>",
))
                fig_wc2.update_layout(
                    showlegend=False,
                    xaxis=dict(title='언급 빈도', gridcolor='#f3f4f6'),
                    yaxis=dict(visible=False),
                    plot_bgcolor='white', paper_bgcolor='white',
                    margin=dict(l=10,r=10,t=5,b=30), height=380,
                )
                st.plotly_chart(fig_wc2, use_container_width=True)

    # ── TAB 3: 트렌드 & 분포
    with tab3:
        col3L, col3R = st.columns(2, gap="large")

        with col3L:
            st.markdown('<div class="sec">연도별 별점 및 감성 추이</div>', unsafe_allow_html=True)
            yearly_rating   = vs.get('yearly_rating', {})
            yearly_compound = vs.get('yearly_compound', {})
            years = sorted(set(list(yearly_rating.keys()) + list(yearly_compound.keys())))

            if years:
                fig_yr = make_subplots(specs=[[{"secondary_y": True}]])
                fig_yr.add_trace(go.Scatter(
                    x=years, y=[yearly_rating.get(y) for y in years],
                    name='평균 별점', mode='lines+markers',
                    line=dict(color='#f59e0b', width=2.5), marker=dict(size=8),
                ), secondary_y=False)
                fig_yr.add_trace(go.Scatter(
                    x=years, y=[yearly_compound.get(y) for y in years],
                    name='감성 점수', mode='lines+markers',
                    line=dict(color='#6366f1', width=2.5, dash='dot'), marker=dict(size=8),
                ), secondary_y=True)
                fig_yr.update_layout(
                    plot_bgcolor='white', paper_bgcolor='white',
                    xaxis=dict(title='연도', tickmode='linear', dtick=1, gridcolor='#f3f4f6'),
                    legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center', font_size=11),
                    margin=dict(l=10,r=10,t=10,b=70), height=280,
                )
                fig_yr.update_yaxes(title_text="평균 별점", range=[1,5],
                                     gridcolor='#f3f4f6', secondary_y=False)
                fig_yr.update_yaxes(title_text="VADER compound", range=[-1,1], secondary_y=True)
                st.plotly_chart(fig_yr, use_container_width=True)
            else:
                st.info("연도별 데이터가 없습니다.")

        with col3R:
            st.markdown('<div class="sec">별점 분포</div>', unsafe_allow_html=True)
            rating_dist = vs.get('rating_dist', {})
            stars  = [1,2,3,4,5]
            counts = [rating_dist.get(s, 0) for s in stars]
            total_rv = sum(counts)
            fig_rd = go.Figure(go.Bar(
                y=[f"⭐{'★'*s} {s}점" for s in stars], x=counts, orientation='h',
                marker_color=['#ef4444','#f97316','#f59e0b','#84cc16','#10b981'],
                text=[f"{c:,}건 ({c/total_rv*100:.1f}%)" if total_rv else "0건" for c in counts],
                textposition='outside', textfont=dict(size=11),
            ))
            fig_rd.update_layout(
                xaxis=dict(title='리뷰 수', gridcolor='#f3f4f6'),
                yaxis=dict(tickfont_size=11),
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=10,r=130,t=10,b=30), height=260,
            )
            st.plotly_chart(fig_rd, use_container_width=True)

            st.markdown('<div class="sec">VADER 감성 분포</div>', unsafe_allow_html=True)
            avg_cmp = vs.get('avg_compound', 0)
            fig_pie = go.Figure(go.Pie(
                labels=['긍정','중립','부정'],
                values=[vs.get('pos_n',0), vs.get('neu_n',0), vs.get('neg_n',0)],
                marker_colors=['#10b981','#d1d5db','#ef4444'],
                hole=0.58, textinfo='label+percent', textfont_size=11, pull=[0.03,0,0.03],
            ))
            fig_pie.update_layout(
                showlegend=False, paper_bgcolor='white',
                margin=dict(l=0,r=0,t=10,b=0), height=220,
                annotations=[dict(text=f"compound<br><b>{avg_cmp:+.3f}</b>",
                                  x=0.5, y=0.5, font_size=12, showarrow=False)],
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── TAB 4: 이슈 & 개선 우선순위
    with tab4:
        t4L, t4R = st.columns([1, 1], gap="large")

        with t4L:
            st.markdown('<div class="sec">AI 이슈 진단 & 실행 처방전</div>', unsafe_allow_html=True)
            st.caption("점수가 왜 낮아졌는지, 그리고 무엇을 해야 하는지 알려드립니다.")
            if not ir_issues:
                st.success("✅ LLM이 탐지한 이슈가 없습니다.")
            else:
                for i, issue in enumerate(ir_issues, 1):
                    sev    = issue.get('severity', 'low')
                    asp    = issue.get('aspect', 'other')
                    asp_kr = ASPECT_KO.get(asp, ASPECT_KO.get('fit_size','기타')
                                           if asp == 'size' else asp)
                    sev_clr = SEV_COLOR.get(sev, '#9ca3af')
                    sev_kr  = SEV_KO.get(sev, sev)
                    sev_bg  = {'high':'#fef2f2','medium':'#fffbeb','low':'#f0fdf4'}.get(sev,'#f9fafb')
                    # 점수 가져오기 (size → fit_size 매핑)
                    asp_key = 'fit_size' if asp == 'size' else asp
                    asp_sc  = prod.get(asp_key)
                    sc_txt  = f'{asp_sc:.1f}점' if asp_sc and not (isinstance(asp_sc,float) and __import__('math').isnan(asp_sc)) else 'N/A'
                    st.markdown(f"""
                    <div class="issue-card" style="border-left:5px solid {sev_clr};background:{sev_bg};">
                      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                        <div>
                          <span style='font-size:.72rem;font-weight:700;color:{sev_clr};
                                background:{sev_clr}22;padding:2px 9px;border-radius:20px;
                                margin-right:6px;'>심각도 {sev_kr}</span>
                          <span style='font-size:.88rem;font-weight:800;color:#1e1b4b;'>
                            {asp_kr} &nbsp;
                            <span style='color:{sev_clr};'>{sc_txt}</span>
                          </span>
                        </div>
                        <span style='font-size:1.2rem;font-weight:900;color:{sev_clr};'>#{i}</span>
                      </div>
                      <div style='font-size:.83rem;color:#1e1b4b;font-weight:600;margin-bottom:5px;'>
                        🔍 진단: {issue.get('issue_summary','')}</div>
                      <div style='font-size:.75rem;color:#6b7280;font-style:italic;margin-bottom:6px;
                                  padding:.4rem .6rem;background:rgba(0,0,0,.04);border-radius:6px;'>
                        "{issue.get('evidence','')}"</div>
                      <div style='font-size:.82rem;color:#059669;font-weight:700;
                                  padding:.4rem .7rem;background:#f0fdf4;border-radius:8px;
                                  border-left:3px solid #10b981;'>
                        💊 처방: {issue.get('improvement_suggestion','')}</div>
                    </div>""", unsafe_allow_html=True)

        with t4R:
            st.markdown('<div class="sec">🎯 속성별 개선 우선순위</div>', unsafe_allow_html=True)
            st.caption("긴급도: 낮은 점수일수록 높음  |  중요도: 언급 빈도가 높을수록 높음")

            pr_rows = []
            for asp in sel_aspects:
                sc  = prod.get(asp)
                mn  = int(prod.get(f"{asp}_mentions", 0))
                asp_c = prod.get(f"{asp}_compound")
                np_  = 0.0
                if asp_c is not None and not (isinstance(asp_c, float) and np.isnan(float(asp_c))):
                    np_ = max(0, (1-(float(asp_c)+1)/2)*100)
                if sc is None or np.isnan(float(sc)): continue
                sc = float(sc)
                urgency    = max(0, 10-sc)
                max_mn     = int(filt[[f"{a}_mentions" for a in ASPECTS]].max().max() or 1)
                importance = mn/max_mn*10
                priority   = urgency*0.55 + importance*0.45
                status     = ('🔴 즉시 개선' if sc<6.0 else ('🟡 모니터링' if sc<7.0 else '🟢 유지·강화'))
                actions = {
                    'durability':('내구성 향상: 소재 품질 및 봉제 강화','현재 내구성 강점 유지'),
                    'fit_size':  ('사이즈 가이드 개선 및 폭 다양화 검토','정확한 사이즈 정보 유지'),
                    'comfort':   ('쿠셔닝·아치 서포트 개선','현재 착용감 강점 홍보 강화'),
                    'design':    ('색상·스타일 라인업 재검토','현재 디자인 강점 마케팅 활용'),
                    'price':     ('가격 경쟁력 점검 및 가치 전달 강화','현재 가성비 포지셔닝 유지'),
                }
                action = actions[asp][1 if sc >= 7.0 else 0]
                pr_rows.append({'asp':asp,'score':sc,'mentions':mn,'neg_pct':np_,
                                'urgency':urgency,'importance':importance,
                                'priority':priority,'status':status,'action':action})

            pr_df = pd.DataFrame(pr_rows).sort_values('priority', ascending=False)

            for rank, (_, row) in enumerate(pr_df.iterrows(), 1):
                pct = int(row['score']/10*100)
                bc  = '#ef4444' if row['score']<6 else ('#f59e0b' if row['score']<7 else '#10b981')
                st.markdown(f"""
                <div class="pr-row" style="border-left:5px solid {bc};">
                  <div style='font-size:1.3rem;font-weight:900;color:{bc};min-width:26px;'>#{rank}</div>
                  <div style='font-size:1.4rem;min-width:28px;'>{ASPECT_ICON[row["asp"]]}</div>
                  <div style='flex:1;min-width:0;'>
                    <div style='font-size:.9rem;font-weight:700;color:#1e1b4b;
                         white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>
                      {ASPECT_KO[row["asp"]]} &nbsp; {row["status"]}</div>
                    <div style='background:#f3f4f6;border-radius:6px;height:8px;margin:4px 0;overflow:hidden;'>
                      <div style='width:{pct}%;height:100%;background:{bc};border-radius:6px;'></div>
                    </div>
                    <div style='font-size:.74rem;color:#6b7280;'>💬 {row["action"]}</div>
                  </div>
                  <div style='text-align:right;min-width:80px;flex-shrink:0;'>
                    <div style='font-size:1.55rem;font-weight:900;color:{bc};'>{row["score"]:.1f}</div>
                    <div style='font-size:.7rem;color:#9ca3af;'>{row["mentions"]:,}건</div>
                    <div style='font-size:.7rem;color:#ef4444;'>부정 {row["neg_pct"]:.0f}%</div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # 중요도 × 긴급도 매트릭스
            st.markdown('<div class="sec">중요도 × 긴급도 매트릭스</div>', unsafe_allow_html=True)
            fig_mx = go.Figure()
            for _, row in pr_df.iterrows():
                clr = '#ef4444' if row['score']<6 else ('#f59e0b' if row['score']<7 else '#10b981')
                fig_mx.add_trace(go.Scatter(
                    x=[row['importance']], y=[row['urgency']],
                    mode='markers+text', name=ASPECT_KO[row['asp']],
                    marker=dict(size=30, color=clr, opacity=0.85,
                                line=dict(width=2, color='white')),
                    text=[f"{ASPECT_ICON[row['asp']]} {ASPECT_KO[row['asp']]}"],
                    textposition='top center', textfont=dict(size=11, color='#1e1b4b'),
                    hovertemplate=(f"<b>{ASPECT_KO[row['asp']]}</b><br>"
                                   f"점수:{row['score']:.2f}<br>언급:{row['mentions']:,}건<extra></extra>"),
                    showlegend=False,
                ))
            for x0,x1,y0,y1,label,lc in [
                (5,10,5,10,"⚠️ 즉시 개선","#dc2626"),
                (0, 5,5,10,"📋 개선 검토","#d97706"),
                (5,10,0, 5,"💪 강점 활용","#059669"),
                (0, 5,0, 5,"📌 낮은 우선순위","#9ca3af"),
            ]:
                fig_mx.add_shape(type='rect', x0=x0,x1=x1,y0=y0,y1=y1,
                                  fillcolor='rgba(0,0,0,0.02)', line_width=0)
                fig_mx.add_annotation(x=(x0+x1)/2, y=y1-0.5, text=label,
                                       showarrow=False, font=dict(size=10, color=lc))
            fig_mx.add_shape(type='line',x0=5,x1=5,y0=0,y1=10,
                              line=dict(color='#d1d5db',width=1,dash='dash'))
            fig_mx.add_shape(type='line',x0=0,x1=10,y0=5,y1=5,
                              line=dict(color='#d1d5db',width=1,dash='dash'))
            fig_mx.update_layout(
                xaxis=dict(title='중요도 (언급 빈도)', range=[-0.5,10.5], gridcolor='#f3f4f6'),
                yaxis=dict(title='긴급도 (낮은 점수)', range=[-0.5,10.5], gridcolor='#f3f4f6'),
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=10,r=10,t=20,b=40), height=360,
            )
            st.plotly_chart(fig_mx, use_container_width=True)

    # ── TAB 5: 속성별 리뷰
    with tab5:
        st.markdown('<div class="sec">📋 속성별 대표 리뷰 탐색</div>', unsafe_allow_html=True)
        c5L, c5R = st.columns([1, 2])
        with c5L:
            drill_asp = st.selectbox("속성", sel_aspects,
                                      format_func=lambda x: f"{ASPECT_ICON[x]} {ASPECT_KO[x]}")
            sent_filt = st.radio("감성 필터", ["전체","긍정","부정","중립"], horizontal=False)
            show_n    = st.slider("표시 리뷰 수", 5, 20, 10)

        with st.spinner("리뷰 추출 중..."):
            drill_df = get_aspect_reviews(prod["parent_asin"], raw_df, drill_asp)

        if drill_df.empty:
            with c5R:
                st.info("해당 속성 관련 리뷰가 없습니다.")
        else:
            d_pos = (drill_df["compound"] > 0.2).sum()
            d_neg = (drill_df["compound"] < -0.1).sum()
            d_neu = len(drill_df) - d_pos - d_neg
            with c5L:
                fig_dp = go.Figure(go.Pie(
                    labels=['긍정','중립','부정'], values=[d_pos,d_neu,d_neg],
                    marker_colors=['#10b981','#d1d5db','#ef4444'],
                    hole=0.55, textinfo='label+percent', textfont_size=11,
                ))
                fig_dp.update_layout(showlegend=False, paper_bgcolor='white',
                                      margin=dict(l=0,r=0,t=10,b=0), height=180,
                                      annotations=[dict(text=f"{len(drill_df)}<br>문장",
                                                        x=0.5,y=0.5,font_size=11,showarrow=False)])
                st.plotly_chart(fig_dp, use_container_width=True)

            with c5R:
                if sent_filt   == "긍정": show = drill_df[drill_df["compound"] >  0.2]
                elif sent_filt == "부정": show = drill_df[drill_df["compound"] < -0.1]
                elif sent_filt == "중립": show = drill_df[
                    (drill_df["compound"]>=-0.1)&(drill_df["compound"]<=0.2)]
                else: show = drill_df
                show = show.sort_values("compound", ascending=(sent_filt=="부정")).head(show_n)
                if show.empty:
                    st.info("해당 조건의 리뷰가 없습니다.")
                else:
                    for _, rv in show.iterrows():
                        c  = rv["compound"]
                        bg = "#f0fdf4" if c>0.2 else ("#fef2f2" if c<-0.1 else "#f8fafc")
                        bc = "#86efac" if c>0.2 else ("#fca5a5" if c<-0.1 else "#e2e8f0")
                        ic = "✅" if c>0.2 else ("❌" if c<-0.1 else "➖")
                        st.markdown(f"""
                        <div class="rv-card" style="background:{bg};border-left:4px solid {bc};">
                          {ic} <b>{int(rv['year'])}년 &nbsp;⭐{rv['rating']:.0f}</b>
                          <span style='float:right;font-size:.72rem;color:#9ca3af;'>
                            VADER {c:+.2f}</span><br>{rv['text']}
                        </div>""", unsafe_allow_html=True)

            st.markdown('<div class="sec">연도별 감성 점수 추이</div>', unsafe_allow_html=True)
            yr_data = drill_df.groupby("year").agg(
                score=("compound", lambda c: round((c.mean()+1)/2*10,2)),
                cnt=("compound","count"),
            ).reset_index()
            fig_yt = go.Figure(go.Scatter(
                x=yr_data["year"], y=yr_data["score"],
                mode='lines+markers+text',
                text=[f"{s:.1f}" for s in yr_data["score"]],
                textposition='top center', textfont_size=11,
                line=dict(color='#6366f1', width=2.5), marker=dict(size=9, color='#6366f1'),
                fill='tozeroy', fillcolor='rgba(99,102,241,0.07)',
                customdata=yr_data["cnt"],
                hovertemplate="%{x}년: %{y:.2f} (%{customdata:,}문장)<extra></extra>",
            ))
            fig_yt.add_hline(y=7.0, line_dash='dash', line_color='#10b981',
                              annotation_text='양호 기준 7.0', annotation_position='right')
            fig_yt.update_layout(
                xaxis=dict(title='연도', tickmode='linear', dtick=1, gridcolor='#f3f4f6'),
                yaxis=dict(title='감성 점수 (0-10)', range=[3,10], gridcolor='#f3f4f6'),
                plot_bgcolor='white', paper_bgcolor='white',
                margin=dict(l=10,r=100,t=20,b=40), height=240,
            )
            st.plotly_chart(fig_yt, use_container_width=True)

    # ── 전체 랭킹 테이블
    st.markdown("---")
    with st.expander("📊 전체 상품 랭킹 테이블", expanded=False):
        disp_cols = ['product_title','brand','review_count','avg_rating','overall_score'] + sel_aspects
        rename_map = {
            'product_title':'상품명','brand':'브랜드',
            'review_count':'리뷰수','avg_rating':'별점','overall_score':'종합점수',
            **{a: ASPECT_KO[a] for a in ASPECTS},
        }
        disp = filt[disp_cols].rename(columns=rename_map).copy()
        for c in ['별점','종합점수'] + list(ASPECT_KO.values()):
            if c in disp.columns:
                disp[c] = pd.to_numeric(disp[c], errors='coerce').apply(
                    lambda x: f"{x:.1f}" if pd.notna(x) else '-')
        disp['상품명'] = disp['상품명'].str[:50]
        disp = disp.reset_index(drop=True)
        disp.index += 1
        # column_config으로 헤더 글씨 깨짐 방지 (이모지 제거)
        col_cfg = {
            '상품명':    st.column_config.TextColumn('상품명',    width='large'),
            '브랜드':    st.column_config.TextColumn('브랜드',    width='small'),
            '리뷰수':    st.column_config.NumberColumn('리뷰수',  width='small', format='%d'),
            '별점':      st.column_config.TextColumn('별점',      width='small'),
            '종합점수':  st.column_config.TextColumn('종합점수',  width='small'),
            **{ASPECT_KO[a]: st.column_config.TextColumn(ASPECT_KO[a], width='small')
               for a in sel_aspects if ASPECT_KO[a] in disp.columns},
        }
        st.dataframe(disp, use_container_width=True, height=400, column_config=col_cfg)

    # ── 푸터
    st.markdown("""
    <div style='text-align:center;color:#9ca3af;font-size:.72rem;
        margin-top:2rem;padding-top:1rem;border-top:1px solid #e5e7eb;'>
      🔬 뉴런스 팀 (마세은&nbsp;·&nbsp;김준형&nbsp;·&nbsp;윤지웅&nbsp;·&nbsp;정채원)
      &nbsp;·&nbsp; 2026-1 인공지능 종합 설계 캡스톤디자인
      &nbsp;·&nbsp; VADER Sentiment Analysis + LDA Topic Modeling
    </div>
    """, unsafe_allow_html=True)


main()
