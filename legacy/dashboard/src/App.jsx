/* ============================================================
   APP SHELL  (ES module port of science_culture_frontend/app.jsx)
   ============================================================ */
import { useState, useEffect } from 'react'
import { KPICard, Tabs, MethodLadder, DecircleCard, MediationCard, CaveatsPanel } from './components.jsx'
import { Tab1, Tab2, Tab3, Tab4 } from './tabs.jsx'
import { DataProvider, useData, ModeSelector } from './DataContext.jsx'

const TABS = [
  { id: "cat",  label: "카테고리 기여도", Comp: Tab1 },
  { id: "kr",   label: "한국 병목 진단",  Comp: Tab2 },
  { id: "sim",  label: "What-if 시뮬레이터", Comp: Tab3 },
  { id: "lag",  label: "Time-lag · 국가비교", Comp: Tab4 },
];

function AppShell() {
  const { data: DATA, mode } = useData();
  const [active, setActive] = useState("cat");
  const [theme, setTheme] = useState(() => localStorage.getItem("xai-theme") || "dark");
  const [caveatsOpen, setCaveatsOpen] = useState(false);
  const Current = TABS.find((t) => t.id === active).Comp;

  useEffect(() => {
    document.documentElement.dataset.theme = theme === "light" ? "light" : "";
    localStorage.setItem("xai-theme", theme);
  }, [theme]);
  const isLight = theme === "light";

  // 동적 헤더: 활성 데이터셋의 KPI / 부제(label·subtitle)를 자동 바인딩.
  const culturePct = DATA.kpis.find((k) => k.id === "culture")?.value || "—";
  const cvR2 = DATA.kpis.find((k) => k.id === "r2")?.value || "—";
  const nCtry = DATA.kpis.find((k) => k.id === "scope")?.value || "—";
  const nYrUnit = DATA.kpis.find((k) => k.id === "scope")?.unit || "개국×12년";
  const isV1 = mode === "v1";
  // v2 통제 결과 부제 1줄 — categoriesControlled 가 있을 때만 동적 보강
  const ctrl = DATA.categoriesControlled || [];
  const ctrlCult = ctrl.find((c) => c.id === "culture")?.value;
  const ctrlPhys = ctrl.find((c) => c.id === "physical")?.value;
  const hasCtrl = !isV1 && ctrlCult != null && ctrlPhys != null;

  return (
    <div className="wrap" style={{ maxWidth: 1280, margin: "0 auto", padding: "28px 28px 40px", display: "flex", flexDirection: "column", gap: 22 }}>
      {/* ---- Header ---- */}
      <header className="app-header" style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 24 }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <div style={{
            width: 52, height: 52, borderRadius: 14, flex: "none",
            background: "linear-gradient(150deg, var(--accent-bright), var(--c-culture-d))",
            display: "grid", placeItems: "center", boxShadow: "0 10px 30px -8px var(--accent-glow)",
          }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="2.2" /><path d="M12 2a10 10 0 0 1 0 20M12 2a10 10 0 0 0 0 20" />
              <ellipse cx="12" cy="12" rx="10" ry="4" /><ellipse cx="12" cy="12" rx="4" ry="10" />
            </svg>
          </div>
          <div>
            <h1 style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.025em", color: "var(--text)", lineHeight: 1.1 }}>
              과학문화 <span style={{ color: "var(--text-dim)", fontWeight: 600 }}>×</span> 국가혁신 <span style={{ color: "var(--accent-bright)" }}>XAI 분석</span>
            </h1>
            <p style={{ fontSize: 14.5, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.55 }}>
              {isV1
                ? <>📜 <b style={{ color: "var(--c-culture)" }}>초기 사양 재현</b> — XGBoost + SHAP, 과학문화 <b style={{ color: "var(--accent-bright)" }}>{culturePct}%</b> (합성 PISA·WVS·ILOSTAT, 20국·17변수·5시나리오)</>
                : <>XGBoost + SHAP — 과학소양(PISA)이 국가혁신과 <b style={{ color: "var(--accent-bright)" }}>{culturePct}% 기여(raw)</b>로 횡단 연관 (R&amp;D 통제 후 견고; 특허 직접측정에서는 within +0.349*** 검출 — §탈순환)</>}
              {hasCtrl && (
                <>
                  <br />
                  <span style={{ color: "var(--text-dim)" }}>
                    발전수준 통제 시 <b style={{ color: "var(--text-soft)" }}>비등</b>
                    {" "}(과학문화 ~<b className="num" style={{ color: "var(--c-culture)" }}>{Math.round(ctrlCult)}%</b>
                    {" "}vs 물적 ~<b className="num">{Math.round(ctrlPhys)}%</b>)
                  </span>
                </>
              )}
            </p>
          </div>
        </div>
        <div className="header-controls" style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <span className="no-print"><ModeSelector /></span>
          <div className="no-print" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 999, background: "var(--surface)", border: "1px solid var(--line)", whiteSpace: "nowrap" }}>
            <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--pos)", boxShadow: "0 0 10px var(--pos)" }} />
            <span className="num" style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text-muted)" }}>CV R² {cvR2} · {isV1 ? "논문 0.509" : "처음 보는 국가 기준"}</span>
          </div>
          <button onClick={() => setTheme(isLight ? "dark" : "light")} aria-label="테마 전환"
            className="no-print"
            style={{ cursor: "pointer", font: "inherit", display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", borderRadius: 999, background: "var(--surface)", border: "1px solid var(--line)", color: "var(--text-muted)", fontSize: 12.5, fontWeight: 700, whiteSpace: "nowrap" }}>
            {isLight ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
            )}
            {isLight ? "다크" : "라이트"}
          </button>
        </div>
      </header>

      {/* ---- KPI row ---- */}
      <div className="kpi-row" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14 }}>
        {DATA.kpis.map((k, i) => <KPICard key={k.id} kpi={k} index={i} />)}
      </div>

      {/* ---- 3층위 종합표 (v2 만; v1 은 표시 안 함) ---- */}
      {DATA.methods && DATA.methods.length > 0 && (
        <MethodLadder methods={DATA.methods} takeaway={DATA.methodsTakeaway} />
      )}

      {/* ---- §7.7·§7.8 탈순환 검증 카드 (v2 만; v1 은 decircle=null) ---- */}
      {DATA.decircle && <DecircleCard data={DATA.decircle} />}

      {/* ---- §7.5 매개분석 카드 (v2 만; v1 은 mediation 키 없음) ---- */}
      {DATA.mediation && DATA.mediation.pathways && DATA.mediation.pathways.length > 0 && (
        <MediationCard data={DATA.mediation} />
      )}

      {/* ---- Tabs ---- */}
      <div className="no-print">
        <Tabs tabs={TABS} active={active} onChange={setActive} />
      </div>

      {/* ---- Content ---- */}
      <main key={active + theme}>
        <Current />
      </main>

      {/* ---- Data·Methods·Caveats 상설 패널 ---- */}
      <CaveatsPanel caveats={DATA.caveats} open={caveatsOpen} onToggle={() => setCaveatsOpen((s) => !s)} />

      <footer className="app-footer" style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", fontSize: 11.5, color: "var(--text-dim)", paddingTop: 4 }}>
        <span>XGBoost · SHAP(TreeExplainer) · GroupKFold(국가) · {nCtry} {nYrUnit} 패널 · 커스텀 SVG · <b>{DATA.label || mode}</b></span>
        <span className="num no-print">© 2026 과학문화 정책연구 · 발표용 데모</span>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <DataProvider>
      <AppShell />
    </DataProvider>
  );
}
