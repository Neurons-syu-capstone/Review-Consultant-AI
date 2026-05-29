import { useState, useEffect, useCallback } from "react";
import "./App.css";

const API = "http://localhost:8000";

const STATUS_CONFIG = {
  위험: { color: "#ef4444", bg: "#fef2f2", border: "#fecaca", label: "위험" },
  주의: { color: "#f59e0b", bg: "#fffbeb", border: "#fde68a", label: "주의" },
  양호: { color: "#10b981", bg: "#f0fdf4", border: "#a7f3d0", label: "양호" },
};

function ScoreBar({ score }) {
  const pct   = Math.round((score / 10) * 100);
  const color = score < 4 ? "#ef4444" : score < 7 ? "#f59e0b" : "#10b981";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: "#e5e7eb", borderRadius: 99, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color,
                      borderRadius: 99, transition: "width .6s ease" }} />
      </div>
      <span style={{ fontSize: 13, fontWeight: 700, color, minWidth: 36 }}>{score.toFixed(1)}</span>
    </div>
  );
}

function OverviewCard({ item, selected, onClick }) {
  const cfg = STATUS_CONFIG[item.status];
  return (
    <div className={`overview-card ${selected ? "selected" : ""}`}
         style={{ borderColor: selected ? cfg.color : "#e5e7eb",
                  background: selected ? cfg.bg : "white" }}
         onClick={onClick}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <span style={{ fontSize: 22 }}>{item.emoji}</span>
        <span className="status-badge" style={{ background: cfg.bg, color: cfg.color, border: `1px solid ${cfg.border}` }}>
          {cfg.label}
        </span>
      </div>
      <div style={{ fontWeight: 700, fontSize: 15, color: "#1e1b4b", marginBottom: 6 }}>{item.cat_kr}</div>
      <ScoreBar score={item.score} />
      <div style={{ display: "flex", gap: 10, marginTop: 6, fontSize: 12, color: "#9ca3af" }}>
        <span>👍 {item.pos}</span>
        <span>👎 {item.neg}</span>
      </div>
    </div>
  );
}

function DiagnosisResult({ result, loading }) {
  if (loading) return (
    <div className="diagnosis-loading">
      <div className="spinner" />
      <p>GPT가 분석 중입니다...</p>
    </div>
  );
  if (!result) return (
    <div className="diagnosis-empty">
      <span style={{ fontSize: 48 }}>🤖</span>
      <p>왼쪽에서 속성을 선택하고<br/>진단 실행 버튼을 누르세요.</p>
    </div>
  );

  const { sections, usage, cat_kr, score } = result;
  const color = score < 4 ? "#ef4444" : score < 7 ? "#f59e0b" : "#10b981";

  return (
    <div className="diagnosis-result">
      <div className="result-header">
        <span style={{ fontSize: 13, fontWeight: 700, color: "#9ca3af", letterSpacing: "0.06em" }}>
          AI 진단 결과 — {cat_kr}
        </span>
        <span className="score-badge" style={{ background: color + "22", color }}>
          {score.toFixed(1)}점
        </span>
      </div>

      {sections.urgent && (
        <div className="urgent-banner">
          <span style={{ fontWeight: 800, color: "#991b1b", fontSize: 13 }}>⚡ 가장 시급한 문제</span>
          <p style={{ margin: "4px 0 0", color: "#7f1d1d", fontSize: 14, fontWeight: 600 }}>
            {sections.urgent}
          </p>
        </div>
      )}

      {sections.diagnosis && (
        <div className="section-block">
          <div className="section-title">🔍 진단 요약</div>
          <p className="section-body">{sections.diagnosis}</p>
        </div>
      )}

      {sections.guidelines?.length > 0 && (
        <div className="section-block">
          <div className="section-title">💊 운영 개선 가이드라인</div>
          <ol className="guidelines-list">
            {sections.guidelines.map((g, i) => (
              <li key={i}>{g.replace(/^\d+\.\s*/, "")}</li>
            ))}
          </ol>
        </div>
      )}

      {sections.effect && (
        <div className="section-block">
          <div className="section-title">📊 예상 효과</div>
          <p className="section-body">{sections.effect}</p>
        </div>
      )}

      <div className="usage-footer">
        토큰 {usage.tokens.toLocaleString()}개 &nbsp;|&nbsp; 예상 비용 ${usage.cost}
      </div>
    </div>
  );
}

export default function App() {
  const [brands,      setBrands]      = useState([]);
  const [products,    setProducts]    = useState([]);
  const [selBrand,    setSelBrand]    = useState("전체");
  const [search,      setSearch]      = useState("");
  const [selAsin,     setSelAsin]     = useState("");
  const [overview,    setOverview]    = useState([]);
  const [selCat,      setSelCat]      = useState("");
  const [diagnosis,   setDiagnosis]   = useState(null);
  const [diagLoading, setDiagLoading] = useState(false);
  const [error,       setError]       = useState("");

  // 브랜드 목록
  useEffect(() => {
    fetch(`${API}/brands`)
      .then(r => r.json())
      .then(data => setBrands(["전체", ...data]))
      .catch(() => setError("서버에 연결할 수 없습니다. api_server.py를 실행해주세요."));
  }, []);

  // 상품 목록
  useEffect(() => {
    const params = new URLSearchParams();
    if (selBrand !== "전체") params.set("brand", selBrand);
    if (search)              params.set("search", search);
    fetch(`${API}/products?${params}`)
      .then(r => r.json())
      .then(data => { setProducts(data); setSelAsin(""); setOverview([]); setDiagnosis(null); })
      .catch(() => {});
  }, [selBrand, search]);

  // 상품 선택 → 개요
  const selectProduct = useCallback((asin) => {
    setSelAsin(asin);
    setDiagnosis(null);
    setSelCat("");
    fetch(`${API}/overview/${asin}`)
      .then(r => r.json())
      .then(setOverview)
      .catch(() => {});
  }, []);

  // 진단 실행
  const runDiagnosis = useCallback(() => {
    if (!selAsin || !selCat) return;
    setDiagLoading(true);
    setDiagnosis(null);
    fetch(`${API}/diagnose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ asin: selAsin, category: selCat }),
    })
      .then(r => r.json())
      .then(data => { setDiagnosis(data); setDiagLoading(false); })
      .catch(() => { setDiagLoading(false); setError("진단 중 오류가 발생했습니다."); });
  }, [selAsin, selCat]);

  const selectedProd = products.find(p => p.asin === selAsin);

  return (
    <div className="app">
      {/* 헤더 */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <div>
              <div className="logo-title">AI Consultant</div>
              <div className="logo-sub">리뷰 기반 이슈 진단 & 실행 처방전</div>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        {error && <div className="error-banner">{error}</div>}

        {/* 필터 */}
        <div className="filter-row">
          <select className="select" value={selBrand} onChange={e => setSelBrand(e.target.value)}>
            {brands.map(b => <option key={b}>{b}</option>)}
          </select>
          <input className="input" placeholder="🔍  상품명 검색..."
                 value={search} onChange={e => setSearch(e.target.value)} />
          <span className="count-badge">{products.length.toLocaleString()}개 상품</span>
        </div>

        <div className="layout">
          {/* 상품 목록 */}
          <aside className="sidebar">
            <div className="sidebar-title">상품 선택</div>
            <div className="product-list">
              {products.slice(0, 100).map(p => (
                <div key={p.asin}
                     className={`product-item ${selAsin === p.asin ? "active" : ""}`}
                     onClick={() => selectProduct(p.asin)}>
                  <div className="product-brand">{p.brand}</div>
                  <div className="product-name">{p.product_title.slice(0, 50)}{p.product_title.length > 50 ? "…" : ""}</div>
                  <div className="product-meta">⭐ {p.avg_rating} &nbsp;·&nbsp; {p.review_count.toLocaleString()}건</div>
                </div>
              ))}
              {products.length === 0 && (
                <div style={{ padding: "2rem", textAlign: "center", color: "#9ca3af", fontSize: 13 }}>
                  상품이 없습니다.
                </div>
              )}
            </div>
          </aside>

          {/* 메인 콘텐츠 */}
          <section className="content">
            {!selAsin ? (
              <div className="empty-state">
                <span style={{ fontSize: 64 }}>👟</span>
                <h2>상품을 선택해주세요</h2>
                <p>왼쪽 목록에서 분석할 상품을 선택하면<br/>AI 이슈 진단을 시작할 수 있습니다.</p>
              </div>
            ) : (
              <>
                {/* 상품 헤더 */}
                <div className="product-header">
                  <div>
                    <div className="product-header-brand">{selectedProd?.brand}</div>
                    <div className="product-header-title">{selectedProd?.product_title}</div>
                    <div className="product-header-meta">
                      ⭐ {selectedProd?.avg_rating} &nbsp;·&nbsp;
                      리뷰 {selectedProd?.review_count?.toLocaleString()}건
                    </div>
                  </div>
                </div>

                {/* 전체 개요 */}
                <div className="section-card">
                  <div className="card-title">📊 전체 속성 개요</div>
                  <div className="overview-grid">
                    {overview.map(item => (
                      <OverviewCard key={item.category} item={item}
                                    selected={selCat === item.category}
                                    onClick={() => setSelCat(item.category)} />
                    ))}
                  </div>
                  {overview.length > 0 && (
                    <p className="overview-hint">카드를 클릭하면 해당 속성을 선택합니다.</p>
                  )}
                </div>

                {/* 진단 */}
                <div className="section-card">
                  <div className="card-title">🤖 AI 이슈 진단 & 실행 처방전</div>
                  <div className="diagnose-row">
                    <div className="diagnose-controls">
                      <div style={{ fontSize: 13, color: "#6b7280", marginBottom: 8 }}>
                        진단할 속성을 선택하세요
                      </div>
                      <div className="cat-buttons">
                        {overview.map(item => {
                          const cfg = STATUS_CONFIG[item.status];
                          return (
                            <button key={item.category}
                                    className={`cat-btn ${selCat === item.category ? "cat-btn-active" : ""}`}
                                    style={selCat === item.category
                                      ? { background: cfg.color, color: "white", borderColor: cfg.color }
                                      : { borderColor: cfg.color + "88", color: cfg.color }}
                                    onClick={() => setSelCat(item.category)}>
                              {item.emoji} {item.cat_kr}
                              <span style={{ fontSize: 11, opacity: 0.85 }}> {item.score.toFixed(1)}점</span>
                            </button>
                          );
                        })}
                      </div>
                      <button className={`run-btn ${(!selCat || diagLoading) ? "run-btn-disabled" : ""}`}
                              disabled={!selCat || diagLoading}
                              onClick={runDiagnosis}>
                        {diagLoading ? "분석 중..." : `🔍 ${selCat ? overview.find(o => o.category === selCat)?.cat_kr : ""} 진단 실행`}
                      </button>
                    </div>
                    <div className="diagnose-result-area">
                      <DiagnosisResult result={diagnosis} loading={diagLoading} />
                    </div>
                  </div>
                </div>
              </>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
