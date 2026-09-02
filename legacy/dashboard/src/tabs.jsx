/* ============================================================
   TAB SCREENS  (ES module port of science_culture_frontend/tabs.jsx)
   ============================================================ */
import { useState } from 'react'
import { ChartCard, CardHead, Callout, Legend, CountryPill, SliderRow, useCountUp } from './components.jsx'
import { HBar, Tornado, LineChart, ComparisonBars } from './charts.jsx'
import { useData } from './DataContext.jsx'

/* ---------- TAB 1 — 카테고리 기여도 ---------- */
export function Tab1() {
  const { data: D } = useData();
  const catRows = D.categories.map((c) => ({ label: c.name, value: c.value, color: c.color, sub: c.desc }));
  const ctrl = D.categoriesControlled || [];
  const ctrlRows = ctrl.map((c) => ({ label: c.name, value: c.value, color: c.color }));
  const featRows = D.features.map((f) => ({ label: f.name, value: f.value, color: D.catColor[f.cat] }));
  const legend = D.categories.map((c) => ({ label: c.name, color: c.color }));
  const axMax = Math.max(...catRows.map((r) => r.value), ...ctrlRows.map((r) => r.value)) + 4;
  const rawPhys = D.categories.find((c) => c.id === "physical")?.value ?? 0;
  const ctrlCult = ctrl.find((c) => c.id === "culture")?.value;
  const ctrlPhys = ctrl.find((c) => c.id === "physical")?.value;
  const nCult = D.features.filter((f) => f.cat === "culture").length;

  return (
    <div className="tab-grid tab-grid-2col" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22, animation: "fadeUp .4s both" }}>
      <ChartCard>
        <CardHead kicker="SHAP · 4대 카테고리 기여 분해" title="카테고리별 혁신 기여도"
          right={<span className="num" style={{ fontSize: 12, color: "var(--text-dim)" }}>합계 100%</span>} />
        <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.1em", color: "var(--text-dim)", marginBottom: 8 }}>① 전체 (raw)</div>
        <HBar rows={catRows} max={axMax} unit="%" layout="stacked" barH={16} fmt={(v) => v.toFixed(1)} />
        {ctrlRows.length > 0 && (
          <>
            <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.1em", color: "var(--text-dim)", margin: "18px 0 8px" }}>② 발전수준(1인당 GDP)·GII 순환 통제 후</div>
            <HBar rows={ctrlRows} max={axMax} unit="%" layout="stacked" barH={16} fmt={(v) => v.toFixed(1)} />
          </>
        )}
        <Callout tone="culture">
          raw에선 <b>물적규모 {rawPhys.toFixed(0)}%</b>가 크나, 대부분 <b>발전수준(1인당 GDP)·GII 순환</b> 탓이다. {ctrlCult != null && ctrlPhys != null ? <>이를 통제하면 격차가 <b style={{ color: "var(--c-culture)" }}>과학문화 {ctrlCult.toFixed(0)}%</b> vs 물적규모 {ctrlPhys.toFixed(0)}%로 <b>크게 축소</b></> : "이를 통제하면 격차가 크게 축소"} — 단 순서는 실행·표본에 민감해 <b>역전 단정은 불가</b>(점추정 스냅샷).
        </Callout>
      </ChartCard>

      <ChartCard>
        <CardHead kicker="개별 피처 기여 · 평균 |SHAP|" title="피처 중요도 Top 10"
          right={<Legend items={legend} />} />
        <HBar rows={featRows} max={Math.max(...featRows.map((r) => r.value)) * 1.05} layout="inline" barH={13} fmt={(v) => v.toFixed(3)} />
        <Callout tone="accent">
          상위 10개 중 <b style={{ color: "var(--c-culture)" }}>{nCult}개가 과학문화 피처</b> — 과학소양(PISA)이 1인당 GDP·R&D 다음 핵심 동인.
        </Callout>
      </ChartCard>
    </div>
  );
}

/* ---------- TAB 2 — 한국 병목 진단 ---------- */
export function Tab2() {
  const { data: D } = useData();
  const rows = [...D.tornado].sort((a, b) => b.value - a.value);
  const top = rows.filter((r) => r.value > 0).slice(0, 3);
  const bottlenecks = rows.filter((r) => r.value < 0).sort((a, b) => a.value - b.value).slice(0, 3);
  const topStrength = top[0];
  const minNeg = bottlenecks[0];

  const MiniStat = ({ r, pos }) => (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "11px 14px",
      background: pos ? "var(--pos-soft)" : "var(--neg-soft)", borderRadius: 10,
      border: `1px solid ${pos ? "rgba(47,211,154,0.25)" : "rgba(255,107,107,0.25)"}` }}>
      <span style={{ fontSize: 13.5, fontWeight: 600, color: "var(--text-soft)", whiteSpace: "nowrap" }}>{r.name}</span>
      <span className="num" style={{ fontSize: 15, fontWeight: 800, color: pos ? "var(--pos)" : "var(--neg)" }}>{pos ? "+" : "−"}{Math.abs(r.value).toFixed(2)}</span>
    </div>
  );

  return (
    <div className="tab-grid tab-grid-2col" style={{ display: "grid", gridTemplateColumns: "1.55fr 1fr", gap: 22, animation: "fadeUp .4s both" }}>
      <ChartCard>
        <CardHead kicker="한국 · 로컬 SHAP 양/음 기여 (Tornado, n=1)" title="혁신 성과 — 강점과 병목 진단" />
        <Tornado rows={rows} />
        <Callout tone="accent">
          {bottlenecks.length === 0
            ? <>한국은 <b style={{ color: "var(--pos)" }}>음의 SHAP 기여가 거의 없다</b> — 표시 피처 전부 양의 기여. 최대 강점은 <b>{topStrength?.name} (+{topStrength?.value.toFixed(2)})</b>.</>
            : bottlenecks.length === 1
              ? <>한국은 양의 기여가 두텁고 유의미한 음의 SHAP은 거의 없다 — 가장 작은 양수 또는 약한 음수는 <b style={{ color: "var(--neg)" }}>{minNeg.name} ({minNeg.value > 0 ? "+" : "−"}{Math.abs(minNeg.value).toFixed(2)})</b>.</>
              : <>한국의 최대 음의 SHAP은 <b style={{ color: "var(--neg)" }}>{minNeg.name} ({minNeg.value > 0 ? "+" : "−"}{Math.abs(minNeg.value).toFixed(2)})</b>.</>
          }
        </Callout>
        <div style={{ marginTop: 10, fontSize: 11.5, color: "var(--text-dim)", lineHeight: 1.55 }}>
          ⚠ <b>SHAP ≠ 정책 진단</b>: 음의 SHAP은 "이 피처값으로 모델이 *기대했던* 점수보다 실제가 낮음"을 뜻하지 *현실의 부족*이 아니다.
          예: 인터넷 보급 SHAP −0.44(2023) → 한국 실측 <b>97.4%·44국 중 5위</b>(세계 최상위권)이므로 정책적 병목 아님.
          단일 국가 local SHAP은 <b>n=1 해석주의</b> · 정책 단정 금지.
        </div>
      </ChartCard>

      <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
        <ChartCard pad={22}>
          <div style={{ fontSize: 12, fontWeight: 800, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--pos)", marginBottom: 14 }}>강점 TOP {Math.min(3, top.length)}</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>{top.map((r) => <MiniStat key={r.name} r={r} pos />)}</div>
        </ChartCard>
        <ChartCard pad={22}>
          <div style={{ fontSize: 12, fontWeight: 800, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--neg)", marginBottom: 14 }}>음의 기여 {bottlenecks.length}개</div>
          {bottlenecks.length === 0 ? (
            <div style={{ fontSize: 13, color: "var(--text-muted)", padding: "10px 0" }}>음의 기여 항목 없음 (모든 피처가 양의 기여).</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>{bottlenecks.map((r) => <MiniStat key={r.name} r={r} pos={false} />)}</div>
          )}
          <div style={{ marginTop: 14, fontSize: 12.5, color: "var(--text-dim)", lineHeight: 1.5 }}>
            한국 표시 데이터상 뚜렷한 다중 병목은 관측되지 않음 — 강점 기반 국가 프로필.
          </div>
        </ChartCard>
      </div>
    </div>
  );
}

/* ---------- TAB 3 — What-if 정책 시뮬레이터 (HIGHLIGHT) ---------- */
export function Tab3() {
  const { data: D, mode } = useData();
  const [country, setCountry] = useState("kr");
  const [vals, setVals] = useState(() => Object.fromEntries(D.policies.map((p) => [p.id, 0])));

  const c = D.countries.find((x) => x.id === country) || D.countries[0];
  // 실 모델 시나리오 효과: 슬라이더 ±20% = scenario_matrix 의 해당 카테고리 delta_pct(%) × base.
  // v1 모드는 시나리오 키가 S1~S5(scenario_key)고, v2 는 카테고리(cat)다.
  const wf = D.whatif?.[country] || {};
  const isV1 = mode === "v1";
  const delta = D.policies.reduce((s, p) => {
    const key = isV1 ? p.scenario_key : p.cat;
    return s + (vals[p.id] / 100) * ((wf[key] ?? 0) / 100) * (c?.base || 0);
  }, 0);
  const predicted = c.base + delta;
  const shown = useCountUp(predicted, [predicted]);
  const up = delta > 0.05, down = delta < -0.05;

  const reset = () => setVals(Object.fromEntries(D.policies.map((p) => [p.id, 0])));

  return (
    <div className="tab-grid tab-grid-2col" style={{ display: "grid", gridTemplateColumns: "1.25fr 1fr", gap: 22, animation: "fadeUp .4s both" }}>
      {/* controls */}
      <ChartCard>
        <CardHead kicker="국가 선택" title="정책 시나리오 시뮬레이터"
          right={<button onClick={reset} style={{ cursor: "pointer", font: "inherit", fontSize: 13, fontWeight: 700, color: "var(--text-muted)", background: "var(--inset)", border: "1px solid var(--line)", borderRadius: 8, padding: "8px 14px", whiteSpace: "nowrap" }}>초기화</button>} />
        <div style={{ display: "flex", flexWrap: "wrap", gap: 9, marginBottom: 26 }}>
          {D.countries.map((cc) => <CountryPill key={cc.id} c={cc} active={cc.id === country} onClick={() => setCountry(cc.id)} />)}
        </div>
        <div style={{ fontSize: 12, fontWeight: 800, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--text-dim)", marginBottom: 18 }}>정책 투자 조정 (현행 대비)</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {D.policies.map((p) => (
            <SliderRow key={p.id} policy={p} value={vals[p.id]} color={D.catColor[p.cat]}
              onChange={(v) => setVals((s) => ({ ...s, [p.id]: v }))} />
          ))}
        </div>
        <Callout tone="neg" label="모델 한계">
          슬라이더 ±20%는 scenario_matrix(XGBoost, +20% 단일 충격) 실효과를 그대로 사용한다.
          <b> 효과가 작거나 음수일 수 있다</b> — 모델은 나라 간 연관은 잡지만 <b>국가 내 개입효과는 통계적으로 미검출</b>(within n.s.).
          정책 효과 예측 신뢰도는 낮다(참고용).
        </Callout>
      </ChartCard>

      {/* result */}
      <ChartCard style={{ background: "linear-gradient(165deg, rgba(94,148,255,0.16), var(--surface-2))", border: "1px solid var(--line-accent)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: 26 }}>{c.flag}</span>
          <span style={{ fontSize: 16, fontWeight: 700, color: "var(--text-soft)", whiteSpace: "nowrap" }}>{c.name} · 예측 혁신점수</span>
        </div>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", textAlign: "center", gap: 18, padding: "10px 0" }}>
          <div className="num" style={{ fontSize: 96, fontWeight: 800, lineHeight: 0.9, color: "var(--text)", textShadow: "0 0 50px var(--accent-glow)" }}>
            {shown.toFixed(2)}
          </div>
          <div className="num" style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            fontSize: 24, fontWeight: 800,
            color: up ? "var(--pos)" : down ? "var(--neg)" : "var(--text-dim)",
            background: up ? "var(--pos-soft)" : down ? "var(--neg-soft)" : "var(--inset)",
            padding: "8px 18px", borderRadius: 999,
            border: `1px solid ${up ? "rgba(47,211,154,0.3)" : down ? "rgba(255,107,107,0.3)" : "var(--line)"}`,
          }}>
            <span style={{ fontSize: 18 }}>{up ? "▲" : down ? "▼" : "—"}</span>
            {delta >= 0 ? "+" : "−"}{Math.abs(delta).toFixed(2)} pt
          </div>
          <div style={{ fontSize: 14, color: "var(--text-muted)", whiteSpace: "nowrap" }}>
            기준점수 <span className="num" style={{ fontWeight: 700, color: "var(--text-soft)" }}>{c.base.toFixed(2)}</span> 대비
          </div>
        </div>
        {/* base vs predicted mini bar */}
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {[{ l: "현행", v: c.base, col: "var(--surface-3)" }, { l: "예측", v: predicted, col: "var(--c-culture)" }].map((b) => (
            <div key={b.l} style={{ display: "grid", gridTemplateColumns: "44px 1fr 60px", alignItems: "center", gap: 12 }}>
              <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-muted)" }}>{b.l}</span>
              <div style={{ height: 14, borderRadius: 7, background: "var(--inset)", overflow: "hidden" }}>
                <div style={{ height: "100%", width: `${Math.min(100, (b.v / 100) * 100)}%`, background: b.col, borderRadius: 7, transition: "width .5s cubic-bezier(.22,1,.36,1)" }} />
              </div>
              <span className="num" style={{ textAlign: "right", fontSize: 14, fontWeight: 800, color: "var(--text)" }}>{b.v.toFixed(1)}</span>
            </div>
          ))}
        </div>
      </ChartCard>
    </div>
  );
}

/* ---------- TAB 4 — Time-lag / 국가비교 ---------- */
export function Tab4() {
  const { data: D } = useData();
  const tl = D.timelag;
  const cMin = Math.min(...tl.culture), cMax = Math.max(...tl.culture);
  const pMin = Math.min(...tl.physical), pMax = Math.max(...tl.physical);
  const series = [
    { name: "과학문화(sci_literacy) → 혁신", color: "var(--c-culture)", data: tl.culture, area: true },
    { name: "물적규모(idx_hard_rd) → 혁신", color: "var(--c-physical)", data: tl.physical, dashed: true },
  ];
  const koreaRow = D.cultureIndex.find((r) => r.korea);
  const koreaVal = koreaRow ? koreaRow.value : 0;
  const lead = Math.max(...D.cultureIndex.map((r) => r.value));
  const leadName = D.cultureIndex.find((r) => r.value === lead)?.name || "선도국";
  const pisaYr = D.pisaRefYear || 2018;

  return (
    <div className="tab-grid tab-grid-2col" style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 22, animation: "fadeUp .4s both" }}>
      <ChartCard>
        <CardHead kicker="시차 Pearson r · 횡단(cross-country) 상관 — within 아님"
          title="시차에 따른 상관 안정성"
          right={<Legend items={series.map((s) => ({ label: s.name, color: s.color }))} />} />
        <LineChart lags={tl.lags} series={series} peak={null} />
        <Callout tone="accent">
          시차 0~7년 내내 과학문화 r≈<b className="num">{cMin.toFixed(2)}~{cMax.toFixed(2)}</b>, 물적규모 r≈<b className="num">{pMin.toFixed(2)}~{pMax.toFixed(2)}</b>로 거의 평탄.
          <b> 특정 지연(lead) 효과 신호는 약하다</b> — 안정적 횡단 연관일 뿐, 인과·시간내 효과 주장 못함(within n.s., findings §6).
        </Callout>
      </ChartCard>

      <ChartCard>
        <CardHead kicker={`PISA 과학소양 (${pisaYr}, 0~100 정규화)`} title="국가 비교 — 단일 지표" />
        <ComparisonBars rows={D.cultureIndex} />
        <Callout tone="culture">
          한국 PISA 과학소양 <b className="num">{koreaVal}</b>점 — {leadName} {lead}점 대비 <b className="num">−{lead - koreaVal}점</b>(나라 간 격차).
          ※ '과학문화 종합지수'가 아닌 <b>PISA 과학소양 단일 지표의 min–max 정규화</b>.
        </Callout>
      </ChartCard>
    </div>
  );
}
