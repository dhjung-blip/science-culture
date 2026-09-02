/* ============================================================
   CHART PRIMITIVES (custom SVG/CSS, Recharts-equivalent shapes)
   ES module port of science_culture_frontend/charts.jsx
   ============================================================ */

/* resolve a CSS var to its computed color (for SVG fills/strokes) */
export function cssVar(name) {
  if (typeof window === "undefined") return "#fff";
  const v = name.startsWith("var(") ? name.slice(4, -1) : name;
  return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || name;
}

/* ---- Horizontal bars (category + feature) ---- */
export function HBar({ rows, max, unit = "", layout = "stacked", barH = 16, fmt }) {
  const m = max || Math.max(...rows.map((r) => Math.abs(r.value))) * 1.08;
  const format = fmt || ((v) => v);
  if (layout === "stacked") {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
        {rows.map((r, i) => (
          <div key={r.label}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 8 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ width: 11, height: 11, borderRadius: 3, background: r.color }} />
                <span style={{ fontSize: 16, fontWeight: 700, color: "var(--text)", whiteSpace: "nowrap" }}>{r.label}</span>
                {r.sub && <span style={{ fontSize: 12.5, color: "var(--text-dim)", whiteSpace: "nowrap" }}>{r.sub}</span>}
              </div>
              <span className="num" style={{ fontSize: 20, fontWeight: 800, color: r.color }}>{format(r.value)}{unit}</span>
            </div>
            <div style={{ height: barH, borderRadius: 8, background: "var(--inset)", overflow: "hidden" }}>
              <div style={{
                height: "100%", width: `${(r.value / m) * 100}%`,
                background: `linear-gradient(90deg, ${r.color}, color-mix(in srgb, ${r.color} 78%, #fff))`,
                borderRadius: 8, transformOrigin: "left",
                animation: `barGrow .8s ${i * 0.08}s cubic-bezier(.22,1,.36,1) both`,
                boxShadow: `0 0 22px -6px ${r.color}`,
              }} />
            </div>
          </div>
        ))}
      </div>
    );
  }
  // inline (compact, for top-10)
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
      {rows.map((r, i) => (
        <div key={r.label} style={{ display: "grid", gridTemplateColumns: "232px 1fr 52px", alignItems: "center", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: r.color, flex: "none" }} />
            <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-soft)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.label}</span>
          </div>
          <div style={{ height: barH, borderRadius: 6, background: "var(--inset)", overflow: "hidden" }}>
            <div style={{
              height: "100%", width: `${(r.value / m) * 100}%`,
              background: r.color, borderRadius: 6, transformOrigin: "left",
              animation: `barGrow .7s ${i * 0.05}s cubic-bezier(.22,1,.36,1) both`,
            }} />
          </div>
          <span className="num" style={{ textAlign: "right", fontSize: 14.5, fontWeight: 700, color: "var(--text)" }}>{format(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

/* ---- Tornado (diverging horizontal) ---- */
export function Tornado({ rows, fmt }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value))) * 1.12;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
      {rows.map((r, i) => {
        const pos = r.value >= 0;
        const col = pos ? "var(--pos)" : "var(--neg)";
        const w = (Math.abs(r.value) / max) * 50; // % of half-track
        return (
          <div key={r.name} style={{ display: "grid", gridTemplateColumns: "150px 1fr", alignItems: "center", gap: 14 }}>
            <div style={{ textAlign: "right", fontSize: 13.5, fontWeight: 600, color: "var(--text-soft)" }}>{r.name}</div>
            <div style={{ position: "relative", height: 26 }}>
              {/* center line */}
              <div style={{ position: "absolute", left: "50%", top: -3, bottom: -3, width: 2, background: "var(--line-strong)", transform: "translateX(-1px)" }} />
              {/* bar */}
              <div style={{
                position: "absolute", top: 3, height: 20, borderRadius: 6,
                left: pos ? "50%" : `${50 - w}%`, width: `${w}%`,
                background: `linear-gradient(90deg, ${pos ? col : "color-mix(in srgb," + col + " 78%,#fff)"}, ${pos ? "color-mix(in srgb," + col + " 78%,#fff)" : col})`,
                transformOrigin: pos ? "left" : "right",
                animation: `barGrow .7s ${i * 0.05}s cubic-bezier(.22,1,.36,1) both`,
                boxShadow: `0 0 18px -6px ${col}`,
              }} />
              {/* value */}
              <span className="num" style={{
                position: "absolute", top: "50%", transform: "translateY(-50%)",
                [pos ? "left" : "right"]: `calc(${50 + w}% + 8px)`,
                ...(pos ? {} : { left: "auto", right: `calc(${50 + w}% + 8px)` }),
                fontSize: 14, fontWeight: 800, color: col, whiteSpace: "nowrap",
              }}>{pos ? "+" : "−"}{Math.abs(r.value).toFixed(2)}</span>
            </div>
          </div>
        );
      })}
      <div style={{ display: "grid", gridTemplateColumns: "150px 1fr", gap: 14, marginTop: 4 }}>
        <div />
        <div style={{ position: "relative", display: "flex", justifyContent: "space-between", fontSize: 11.5, fontWeight: 700, color: "var(--text-dim)" }}>
          <span style={{ color: "var(--neg)" }}>◀ 병목 (음의 기여)</span>
          <span style={{ color: "var(--pos)" }}>강점 (양의 기여) ▶</span>
        </div>
      </div>
    </div>
  );
}

/* ---- Line chart (time-lag correlation, peak 연출 제거 — 평탄선 그대로 노출) ---- */
export function LineChart({ lags, series, peak }) {
  const W = 760, H = 320, pl = 52, pr = 24, pt = 26, pb = 44;
  const iw = W - pl - pr, ih = H - pt - pb;
  // y축 0~1.0(전체 범위)로 펴 평탄함을 그대로 보이게 (가짜 봉우리 연출 방지).
  const yMax = 1.0, yMin = 0;
  const x = (i) => pl + (i / (lags.length - 1)) * iw;
  const y = (v) => pt + (1 - (v - yMin) / (yMax - yMin)) * ih;
  const yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];

  const pathFor = (vals) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");

  const C = {
    line: cssVar("--line"), dim: cssVar("--text-dim"), muted: cssVar("--text-muted"),
    bg: cssVar("--bg"),
  };

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto" }}>
      {/* grid */}
      {yticks.map((t) => (
        <g key={t}>
          <line x1={pl} x2={W - pr} y1={y(t)} y2={y(t)} stroke={C.line} strokeWidth="1" />
          <text x={pl - 12} y={y(t) + 4} textAnchor="end" fontSize="12" fill={C.dim} className="num">{t.toFixed(1)}</text>
        </g>
      ))}
      {/* x labels */}
      {lags.map((l, i) => (
        <text key={l} x={x(i)} y={H - pb + 24} textAnchor="middle" fontSize="12.5" fill={C.muted} className="num" fontWeight="600">{l}년</text>
      ))}
      <text x={(pl + W - pr) / 2} y={H - 6} textAnchor="middle" fontSize="12" fill={C.dim} fontWeight="700">시차 (Time-lag, years) · 횡단 Pearson r</text>

      {series.map((s, si) => {
        const col = cssVar(s.color);
        return (
          <g key={s.name}>
            {s.area && (
              <path d={`${pathFor(s.data)} L${x(s.data.length - 1)},${y(0)} L${x(0)},${y(0)} Z`} fill={col} opacity="0.08" />
            )}
            <path d={pathFor(s.data)} fill="none" stroke={col} strokeWidth={s.area ? 3 : 2.5}
              strokeLinecap="round" strokeLinejoin="round"
              strokeDasharray={s.dashed ? "6 6" : "0"}
              style={{ animation: `popIn .6s ${si * 0.15}s both` }} />
            {s.data.map((v, i) => (
              <circle key={i} cx={x(i)} cy={y(v)} r={4}
                fill={C.bg} stroke={col} strokeWidth="2.5"
                style={{ animation: `popIn .4s ${0.3 + i * 0.04}s both` }} />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

/* ---- Comparison bars (vertical, country gap) ---- */
export function ComparisonBars({ rows, max = 100 }) {
  return (
    <div>
      <div style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 14, height: 230, padding: "0 4px" }}>
        {rows.map((r, i) => {
          const h = (r.value / max) * 100;
          return (
            <div key={r.id} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-end", height: "100%", gap: 10 }}>
              <span className="num" style={{ fontSize: 18, fontWeight: 800, color: r.korea ? "var(--c-culture)" : "var(--text-soft)" }}>{r.value}</span>
              <div style={{
                width: "100%", maxWidth: 64, height: `${h}%`,
                background: r.korea
                  ? "linear-gradient(180deg, var(--accent-bright), var(--c-culture-d))"
                  : "linear-gradient(180deg, color-mix(in srgb, var(--text-dim) 24%, var(--surface)), color-mix(in srgb, var(--text-dim) 14%, var(--surface)))",
                border: r.korea ? "1px solid var(--line-accent)" : "1px solid var(--line)",
                borderRadius: "8px 8px 4px 4px",
                transformOrigin: "bottom",
                animation: `barGrow .8s ${i * 0.07}s cubic-bezier(.22,1,.36,1) both`,
                boxShadow: r.korea ? "0 0 28px -6px var(--accent-glow)" : "none",
              }} />
              <span style={{ fontSize: 14, fontWeight: r.korea ? 800 : 600, color: r.korea ? "var(--text)" : "var(--text-muted)" }}>{r.name}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
