/* ============================================================
   CORE COMPONENTS  (ES module port of science_culture_frontend)
   ============================================================ */
import { useState, useEffect, useRef } from 'react'

/* ---- animated count-up hook ---- */
export function useCountUp(target, deps = [], dur = 650) {
  const [val, setVal] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    const start = performance.now();
    const a = from.current, b = target;
    let raf;
    const tick = (t) => {
      const p = Math.min(1, (t - start) / dur);
      const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
      setVal(a + (b - a) * e);
      if (p < 1) raf = requestAnimationFrame(tick);
      else from.current = b;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, deps); // eslint-disable-line
  return val;
}

/* ---- Section eyebrow + title ---- */
export function CardHead({ kicker, title, right }) {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, marginBottom: 18 }}>
      <div>
        {kicker && <div className="eyebrow" style={{ marginBottom: 8 }}>{kicker}</div>}
        <h3 style={{ fontSize: 22, fontWeight: 800, letterSpacing: "-0.02em", color: "var(--text)" }}>{title}</h3>
      </div>
      {right}
    </div>
  );
}

/* ---- Chart card shell ---- */
export function ChartCard({ children, style, pad = 26, span }) {
  return (
    <section style={{
      background: "linear-gradient(180deg, var(--surface), var(--surface-2))",
      border: "1px solid var(--line)",
      borderRadius: "var(--r-lg)",
      padding: pad,
      boxShadow: "var(--shadow-card)",
      gridColumn: span ? `span ${span}` : undefined,
      display: "flex", flexDirection: "column",
      ...style,
    }}>
      {children}
    </section>
  );
}

/* ---- KPI card ---- */
export function KPICard({ kpi, index }) {
  const isBig = kpi.big;
  return (
    <div style={{
      position: "relative",
      background: isBig
        ? "linear-gradient(150deg, rgba(94,148,255,0.22), rgba(94,148,255,0.04))"
        : "linear-gradient(170deg, var(--surface), var(--surface-2))",
      border: isBig ? "1px solid var(--line-accent)" : "1px solid var(--line)",
      borderRadius: "var(--r-md)",
      padding: "18px 20px",
      overflow: "hidden",
      boxShadow: "var(--shadow-card)",
      animation: `fadeUp .5s ${index * 0.06}s both`,
    }}>
      {isBig && <div style={{ position: "absolute", inset: 0, background: "radial-gradient(140px 90px at 90% 0%, var(--accent-glow), transparent 70%)", pointerEvents: "none" }} />}
      <div style={{ position: "relative" }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: isBig ? "var(--accent-bright)" : "var(--text-muted)", marginBottom: 12, whiteSpace: "nowrap" }}>{kpi.label}</div>
        <div className="num" style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
          <span style={{ fontSize: isBig ? 44 : 34, fontWeight: 800, color: "var(--text)", lineHeight: 1 }}>{kpi.value}</span>
          <span style={{ fontSize: isBig ? 18 : 13, fontWeight: 700, color: isBig ? "var(--accent-bright)" : "var(--text-muted)", whiteSpace: "nowrap" }}>{kpi.unit}</span>
        </div>
        <div style={{ fontSize: 12, color: "var(--text-dim)", marginTop: 10 }}>{kpi.note}</div>
      </div>
    </div>
  );
}

/* ---- Tabs ---- */
export function Tabs({ tabs, active, onChange }) {
  return (
    <div role="tablist" style={{
      display: "flex", gap: 6, padding: 6,
      background: "var(--inset)", border: "1px solid var(--line)",
      borderRadius: "var(--r-md)",
    }}>
      {tabs.map((t, i) => {
        const on = active === t.id;
        return (
          <button key={t.id} role="tab" aria-selected={on} onClick={() => onChange(t.id)}
            style={{
              flex: 1, position: "relative", cursor: "pointer",
              border: "none", borderRadius: 10,
              padding: "13px 16px",
              background: on ? "linear-gradient(180deg, var(--surface-3), var(--surface-2))" : "transparent",
              boxShadow: on ? "0 1px 0 rgba(255,255,255,0.06) inset, 0 8px 20px -12px rgba(0,0,0,0.8)" : "none",
              color: on ? "var(--text)" : "var(--text-muted)",
              font: "inherit", transition: "all .18s",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
            }}>
            <span style={{
              width: 26, height: 26, borderRadius: 7, display: "grid", placeItems: "center",
              fontSize: 13, fontWeight: 800,
              background: on ? "var(--accent)" : "var(--surface-3)",
              color: on ? "#fff" : "var(--text-dim)",
            }} className="num">{i + 1}</span>
            <span style={{ fontSize: 15.5, fontWeight: on ? 800 : 600, letterSpacing: "-0.01em", whiteSpace: "nowrap" }}>{t.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/* ---- Callout (핵심 메시지) ---- */
export function Callout({ tone = "accent", children, label = "핵심 메시지" }) {
  const map = {
    accent:   { c: "var(--accent)",     bg: "rgba(94,148,255,0.10)",  br: "rgba(94,148,255,0.30)" },
    culture:  { c: "var(--c-culture)",  bg: "rgba(94,148,255,0.10)",  br: "rgba(94,148,255,0.30)" },
    pos:      { c: "var(--pos)",        bg: "rgba(47,211,154,0.10)",  br: "rgba(47,211,154,0.30)" },
    neg:      { c: "var(--neg)",        bg: "rgba(255,107,107,0.10)", br: "rgba(255,107,107,0.30)" },
  };
  const s = map[tone] || map.accent;
  return (
    <div style={{
      display: "flex", gap: 16, alignItems: "center",
      marginTop: "auto", padding: "16px 18px",
      background: s.bg, border: `1px solid ${s.br}`,
      borderRadius: "var(--r-md)",
    }}>
      <div style={{
        flex: "none", width: 40, height: 40, borderRadius: 11, display: "grid", placeItems: "center",
        background: s.c, color: "#0a1120", boxShadow: `0 8px 20px -8px ${s.c}`,
      }}>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 18h6M10 21h4M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V17h6v-.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2Z" />
        </svg>
      </div>
      <div>
        <div style={{ fontSize: 10.5, fontWeight: 800, letterSpacing: "0.18em", textTransform: "uppercase", color: s.c, marginBottom: 4 }}>{label}</div>
        <div style={{ fontSize: 16.5, fontWeight: 600, color: "var(--text-soft)", lineHeight: 1.45, textWrap: "pretty" }}>{children}</div>
      </div>
    </div>
  );
}

/* ---- Legend ---- */
export function Legend({ items }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 18px", alignItems: "center" }}>
      {items.map((it) => (
        <div key={it.label} style={{ display: "flex", alignItems: "center", gap: 7 }}>
          <span style={{ width: 11, height: 11, borderRadius: 3, background: it.color }} />
          <span style={{ fontSize: 12.5, color: "var(--text-muted)", fontWeight: 600, whiteSpace: "nowrap" }}>{it.label}</span>
        </div>
      ))}
    </div>
  );
}

/* ---- Country pill ---- */
export function CountryPill({ c, active, onClick }) {
  return (
    <button onClick={onClick} style={{
      display: "flex", alignItems: "center", gap: 9, cursor: "pointer",
      padding: "10px 16px", borderRadius: 999,
      border: active ? "1px solid var(--line-accent)" : "1px solid var(--line)",
      background: active ? "linear-gradient(180deg, rgba(94,148,255,0.22), rgba(94,148,255,0.06))" : "var(--surface)",
      color: active ? "var(--accent-bright)" : "var(--text-muted)",
      font: "inherit", fontWeight: active ? 800 : 600, fontSize: 15, whiteSpace: "nowrap",
      boxShadow: active ? "0 8px 22px -12px var(--accent-glow)" : "none",
      transition: "all .16s",
    }}>
      <span style={{ fontSize: 18, filter: active ? "none" : "grayscale(0.4) opacity(0.8)" }}>{c.flag}</span>
      {c.name}
    </button>
  );
}

/* ---- Slider row (policy scenario) ---- */
export function SliderRow({ policy, value, onChange, color }) {
  const pos = value >= 0;
  return (
    <div style={{ display: "grid", gridTemplateColumns: "172px 1fr 78px", alignItems: "center", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
        <span style={{ width: 9, height: 9, borderRadius: 3, background: color, flex: "none" }} />
        <span style={{ fontSize: 14.5, fontWeight: 600, color: "var(--text-soft)", whiteSpace: "nowrap" }}>{policy.label}</span>
      </div>
      <div style={{ position: "relative", height: 30, display: "flex", alignItems: "center" }}>
        <input type="range" min={-20} max={20} step={5} value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{ width: "100%", accentColor: color, height: 6, cursor: "pointer" }} />
      </div>
      <div className="num" style={{ textAlign: "right", fontSize: 17, fontWeight: 800,
        color: value === 0 ? "var(--text-dim)" : pos ? "var(--pos)" : "var(--neg)" }}>
        {pos ? "+" : ""}{value}%
      </div>
    </div>
  );
}

/* ---- Method Ladder (3층위 종합표: XGBoost 한계기여 / Pooled OLS / Panel FE) ---- */
export function MethodLadder({ methods, takeaway }) {
  if (!methods || methods.length === 0) return null;
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 14,
      padding: "16px 18px 14px",
    }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 11.5, fontWeight: 800, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--text-dim)" }}>
            과학소양 ↔ 혁신 — 같은 데이터, 다른 질문
          </div>
          <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)", marginTop: 4 }}>
            3층위 분석 종합 <span style={{ color: "var(--text-muted)", fontWeight: 500, fontSize: 13 }}>· 견고한 횡단 연관 / within 인과 미검출</span>
          </div>
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1.4fr 0.8fr 1fr 1.4fr", gap: 0, fontSize: 13 }}>
        <div style={{ padding: "8px 10px", fontWeight: 700, color: "var(--text-dim)", fontSize: 11.5, borderBottom: "1px solid var(--line)" }}>분석</div>
        <div style={{ padding: "8px 10px", fontWeight: 700, color: "var(--text-dim)", fontSize: 11.5, borderBottom: "1px solid var(--line)", textAlign: "right" }}>값</div>
        <div style={{ padding: "8px 10px", fontWeight: 700, color: "var(--text-dim)", fontSize: 11.5, borderBottom: "1px solid var(--line)" }}>해석</div>
        <div style={{ padding: "8px 10px", fontWeight: 700, color: "var(--text-dim)", fontSize: 11.5, borderBottom: "1px solid var(--line)" }}>세부</div>
        {methods.map((m, i) => (
          <div key={m.label} style={{ display: "contents" }}>
            <div style={{ padding: "10px", fontWeight: 700, color: "var(--text-soft)", background: m.ns ? "var(--inset)" : "transparent", opacity: m.ns ? 0.85 : 1 }}>{m.label}</div>
            <div className="num" style={{ padding: "10px", fontWeight: 800, color: m.ns ? "var(--text-dim)" : "var(--c-culture)", textAlign: "right", background: m.ns ? "var(--inset)" : "transparent" }}>{m.value}</div>
            <div style={{ padding: "10px", color: m.ns ? "var(--text-dim)" : "var(--text-soft)", background: m.ns ? "var(--inset)" : "transparent" }}>{m.verdict}</div>
            <div style={{ padding: "10px", color: "var(--text-dim)", fontSize: 12, background: m.ns ? "var(--inset)" : "transparent" }}>{m.detail}</div>
          </div>
        ))}
      </div>
      {takeaway && (
        <div style={{ marginTop: 12, padding: "10px 12px", fontSize: 12.5, color: "var(--text-muted)",
          background: "rgba(94,148,255,0.06)", border: "1px solid rgba(94,148,255,0.18)", borderRadius: 8, lineHeight: 1.5 }}>
          <b style={{ color: "var(--accent-bright)" }}>핵심:</b> {takeaway}
        </div>
      )}
    </div>
  );
}

/* ---- MediationCard (§7.5 Baron-Kenny + Sobel · 경로 도식 + 매개 비율 막대) ----
   주의: '인과' 단어 금지 — "경로(pathway)·매개·연관"만 사용. */
export function MediationCard({ data }) {
  if (!data || !data.pathways || data.pathways.length === 0) return null;
  const { x, y, pathways, total_effect, note } = data;
  const maxPct = Math.max(...pathways.map((p) => p.mediated_pct));

  /* 좌(X) — 매개변수 박스(stacked) — 우(Y) 도식 */
  return (
    <section style={{
      background: "linear-gradient(180deg, var(--surface), var(--surface-2))",
      border: "1px solid var(--line)",
      borderRadius: "var(--r-lg)",
      padding: "22px 24px 20px",
      boxShadow: "var(--shadow-card)",
      display: "flex", flexDirection: "column", gap: 16,
    }}>
      {/* head */}
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
        <div>
          <div className="eyebrow" style={{ marginBottom: 6 }}>§7.5 매개분석 · Baron-Kenny + Sobel</div>
          <h3 style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-0.02em", color: "var(--text)" }}>
            과학소양 → 매개변수 → 혁신 <span style={{ color: "var(--text-muted)", fontSize: 13.5, fontWeight: 600 }}>· 경로(pathway) 분해</span>
          </h3>
        </div>
        <div className="num" style={{
          fontSize: 12.5, fontWeight: 700, color: "var(--text-muted)",
          padding: "5px 11px", borderRadius: 999,
          background: "var(--inset)", border: "1px solid var(--line)",
          whiteSpace: "nowrap",
        }}>
          총효과 c = +{total_effect.toFixed(2)}***
        </div>
      </div>

      {/* X → M → Y 도식 */}
      <div style={{
        display: "grid", gridTemplateColumns: "minmax(140px, 1fr) minmax(280px, 2.2fr) minmax(140px, 1fr)",
        alignItems: "center", gap: 14,
      }} className="mediation-row">
        {/* X */}
        <div style={{
          padding: "14px 16px", borderRadius: 12,
          background: "linear-gradient(180deg, rgba(94,148,255,0.16), rgba(94,148,255,0.04))",
          border: "1px solid var(--line-accent)",
          textAlign: "center",
        }}>
          <div style={{ fontSize: 10.5, fontWeight: 800, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--c-culture)" }}>X (독립)</div>
          <div style={{ fontSize: 14.5, fontWeight: 800, color: "var(--text)", marginTop: 5, whiteSpace: "nowrap" }}>{x}</div>
        </div>

        {/* M stack — 매개변수 박스 3개 */}
        <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
          {pathways.map((p) => {
            const isH = !!p.highlight;
            const barPct = maxPct > 0 ? (p.mediated_pct / maxPct) * 100 : 0;
            return (
              <div key={p.m} style={{
                position: "relative",
                padding: "10px 12px 11px",
                borderRadius: 10,
                background: isH
                  ? "linear-gradient(180deg, rgba(94,148,255,0.14), rgba(94,148,255,0.03))"
                  : "var(--inset)",
                border: isH ? "1px solid var(--line-accent)" : "1px solid var(--line)",
              }}>
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "minmax(110px, 1.4fr) minmax(72px, 0.9fr) minmax(72px, 0.9fr) minmax(110px, 1.4fr)",
                  alignItems: "baseline", gap: 10, fontSize: 12.5,
                }}>
                  <div style={{ fontWeight: 800, color: isH ? "var(--accent-bright)" : "var(--text-soft)", whiteSpace: "nowrap" }}>
                    {isH && <span style={{ marginRight: 5 }}>★</span>}{p.m_label}
                  </div>
                  <div className="num" style={{ color: "var(--text-muted)" }}>
                    a = <b style={{ color: "var(--text)" }}>{p.a.toFixed(3)}</b>
                  </div>
                  <div className="num" style={{ color: "var(--text-muted)" }}>
                    b = <b style={{ color: "var(--text)" }}>{p.b.toFixed(3)}</b>
                  </div>
                  <div className="num" style={{ color: "var(--text-muted)", textAlign: "right", whiteSpace: "nowrap" }}>
                    Sobel p {p.sobel_p}
                  </div>
                </div>
                {/* 매개 비율 bar */}
                <div style={{ marginTop: 8, display: "grid", gridTemplateColumns: "1fr 64px", alignItems: "center", gap: 10 }}>
                  <div style={{ height: 8, borderRadius: 6, background: "var(--surface-3)", overflow: "hidden" }}>
                    <div style={{
                      height: "100%", width: `${barPct}%`,
                      background: isH ? "var(--c-culture)" : "var(--text-dim)",
                      borderRadius: 6,
                      animation: "barGrow .6s cubic-bezier(.22,1,.36,1)",
                      transformOrigin: "left",
                    }} />
                  </div>
                  <div className="num" style={{
                    textAlign: "right", fontSize: 13, fontWeight: 800,
                    color: isH ? "var(--c-culture)" : "var(--text-soft)",
                  }}>
                    {p.mediated_pct.toFixed(1)}%
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Y */}
        <div style={{
          padding: "14px 16px", borderRadius: 12,
          background: "linear-gradient(180deg, rgba(47,211,154,0.14), rgba(47,211,154,0.04))",
          border: "1px solid rgba(47,211,154,0.30)",
          textAlign: "center",
        }}>
          <div style={{ fontSize: 10.5, fontWeight: 800, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--pos)" }}>Y (종속)</div>
          <div style={{ fontSize: 14.5, fontWeight: 800, color: "var(--text)", marginTop: 5, whiteSpace: "nowrap" }}>{y}</div>
        </div>
      </div>

      {/* legend & note */}
      <div style={{
        display: "flex", flexWrap: "wrap", gap: "6px 16px", alignItems: "center",
        fontSize: 11, color: "var(--text-dim)",
      }}>
        <span><b style={{ color: "var(--text-muted)" }}>a</b>: X → M</span>
        <span><b style={{ color: "var(--text-muted)" }}>b</b>: M → Y (X 통제)</span>
        <span><b style={{ color: "var(--text-muted)" }}>매개 비율</b> = 간접효과 / 총효과 × 100</span>
        <span>★ = 최대 매개경로</span>
      </div>
      {note && (
        <div style={{
          padding: "10px 12px", fontSize: 12, color: "var(--text-muted)",
          background: "rgba(94,148,255,0.06)", border: "1px solid rgba(94,148,255,0.18)",
          borderRadius: 8, lineHeight: 1.55,
        }}>
          {note}
        </div>
      )}
    </section>
  );
}

/* ---- DecircleCard (§7.7·§7.8 탈순환 검증 · 비순환 타깃 3구도 표) ----
   주의: '인과' 단어 금지(MediationCard 규칙 동일) — "연관·경로·검증" 어휘만.
   v1 은 data.decircle=null → 렌더 안 함. */
export function DecircleCard({ data }) {
  if (!data || !data.rows || data.rows.length === 0) return null;
  const { intro, rows, robustness, conclusion, caveats } = data;
  const th = {
    padding: "8px 10px", fontWeight: 700, color: "var(--text-dim)",
    fontSize: 11.5, borderBottom: "1px solid var(--line)", whiteSpace: "nowrap",
  };
  const td = { padding: "10px", fontSize: 13 };
  return (
    <section style={{
      background: "linear-gradient(180deg, var(--surface), var(--surface-2))",
      border: "1px solid var(--line)",
      borderRadius: "var(--r-lg)",
      padding: "22px 24px 20px",
      boxShadow: "var(--shadow-card)",
      display: "flex", flexDirection: "column", gap: 14,
    }}>
      {/* head */}
      <div>
        <div className="eyebrow" style={{ marginBottom: 6 }}>§7.7·§7.8 탈순환 검증 · 비순환 타깃 3구도</div>
        <h3 style={{ fontSize: 20, fontWeight: 800, letterSpacing: "-0.02em", color: "var(--text)" }}>
          탈순환 검증 — 'GII 순환 덕' 반론 기각
        </h3>
        <p style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 6, lineHeight: 1.55, textWrap: "pretty" }}>
          {intro}
        </p>
      </div>

      {/* 3행 표 — 좁은 화면은 가로 스크롤 */}
      <div style={{ overflowX: "auto" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr 0.9fr 1.3fr 0.6fr", minWidth: 640 }}>
          <div style={th}>타깃</div>
          <div style={th}>종류</div>
          <div style={{ ...th, textAlign: "right" }}>횡단 β</div>
          <div style={{ ...th, textAlign: "right" }}>within FE</div>
          <div style={{ ...th, textAlign: "right" }}>매개</div>
          {rows.map((r) => (
            <div key={r.target} style={{ display: "contents" }}>
              <div style={{ ...td, fontWeight: 700, color: "var(--text-soft)" }}>
                {r.target}
                {r.cvCulture != null && (
                  <div className="num" style={{ fontSize: 11, fontWeight: 600, color: "var(--text-dim)", marginTop: 3, whiteSpace: "nowrap" }}>
                    CV 문화 {r.cvCulture.toFixed(3)} &gt; R&D투입 {r.cvRnd.toFixed(3)}
                  </div>
                )}
              </div>
              <div style={{ ...td, color: "var(--text-muted)", fontSize: 12.5 }}>{r.kind}</div>
              <div className="num" style={{ ...td, textAlign: "right", fontWeight: 700, color: "var(--text)" }}>{r.beta}</div>
              <div className="num" style={{
                ...td, textAlign: "right", whiteSpace: "nowrap",
                fontWeight: r.withinSig ? 800 : 600,
                color: r.withinSig ? "var(--accent-bright)" : "var(--text-dim)",
                background: r.withinSig ? "rgba(94,148,255,0.12)" : "transparent",
                borderRadius: r.withinSig ? 8 : 0,
              }}>
                {r.withinSig && <span style={{ marginRight: 4 }}>★</span>}{r.within}
              </div>
              <div className="num" style={{ ...td, textAlign: "right", fontWeight: 700, color: "var(--text-soft)" }}>{r.mediation}</div>
            </div>
          ))}
        </div>
      </div>

      {/* 결론 + robustness + caveats */}
      <div style={{
        padding: "10px 12px", fontSize: 12.5, color: "var(--text-muted)",
        background: "rgba(94,148,255,0.06)", border: "1px solid rgba(94,148,255,0.18)",
        borderRadius: 8, lineHeight: 1.55,
      }}>
        <b style={{ color: "var(--accent-bright)" }}>결론:</b> {conclusion}
      </div>
      <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
        <b style={{ color: "var(--pos)" }}>robustness</b> · {robustness}
      </div>
      <div style={{ fontSize: 11.5, color: "var(--text-dim)", lineHeight: 1.55 }}>
        ⚠ {caveats}
      </div>
    </section>
  );
}

/* ---- Caveats Panel (Data·Methods·한계 접이식) ---- */
export function CaveatsPanel({ caveats, open, onToggle }) {
  if (!caveats || caveats.length === 0) return null;
  return (
    <div style={{
      background: "var(--surface)", border: "1px solid var(--line)", borderRadius: 14,
      padding: open ? "14px 18px 16px" : "12px 18px",
    }}>
      <button onClick={onToggle} style={{
        cursor: "pointer", font: "inherit", border: "none", background: "transparent", padding: 0,
        width: "100%", display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <span style={{ fontSize: 13, fontWeight: 800, letterSpacing: "0.06em", color: "var(--text-soft)" }}>
          ⓘ Data · Methods · 한계 ({caveats.length})
        </span>
        <span style={{ fontSize: 12, fontWeight: 700, color: "var(--text-dim)" }}>{open ? "닫기 ▴" : "펼치기 ▾"}</span>
      </button>
      {open && (
        <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 12 }}>
          {caveats.map((cv) => (
            <div key={cv.title} style={{ padding: "12px 14px", background: "var(--inset)", borderRadius: 10, border: "1px solid var(--line)" }}>
              <div style={{ fontSize: 12.5, fontWeight: 800, color: "var(--accent-bright)", marginBottom: 6 }}>{cv.title}</div>
              <div style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.55 }}>{cv.body}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
