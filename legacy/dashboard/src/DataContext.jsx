/* ============================================================
   DataContext — v1(초기 사양) vs v2(확장 분석) 토글
   ============================================================ */
import { createContext, useContext, useState, useMemo } from 'react'
import { DATASETS, DEFAULT_MODE } from './data.js'

const Ctx = createContext({ data: DATASETS[DEFAULT_MODE], mode: DEFAULT_MODE, setMode: () => {} });

export function DataProvider({ children }) {
  const available = Object.keys(DATASETS);
  const initial = available.includes(DEFAULT_MODE) ? DEFAULT_MODE : available[0];
  const [mode, setMode] = useState(() => {
    const saved = typeof localStorage !== 'undefined' && localStorage.getItem('xai-mode');
    return (saved && available.includes(saved)) ? saved : initial;
  });
  const handleSetMode = (m) => {
    if (available.includes(m)) {
      setMode(m);
      if (typeof localStorage !== 'undefined') localStorage.setItem('xai-mode', m);
    }
  };
  const value = useMemo(() => ({
    data: DATASETS[mode], mode, setMode: handleSetMode, available,
  }), [mode]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useData() {
  return useContext(Ctx);
}

/* ---- Mode selector pill (헤더에 배치) ---- */
export function ModeSelector() {
  const { mode, setMode, available } = useData();
  if (available.length < 2) return null;
  const LABEL = {
    v2: { short: 'v2', long: '확장 분석' },
    v1: { short: 'v1', long: '초기 사양' },
  };
  return (
    <div style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: 3, borderRadius: 999, background: 'var(--surface)', border: '1px solid var(--line)',
      whiteSpace: 'nowrap',
    }}>
      {available.map((m) => {
        const active = m === mode;
        const lab = LABEL[m] || { short: m, long: m };
        return (
          <button key={m} onClick={() => setMode(m)} title={m === 'v1' ? '초기 사양 (20국·합성)' : '확장 분석 (44국·실데이터)'}
            style={{
              cursor: 'pointer', font: 'inherit', fontSize: 12.5, fontWeight: 800,
              padding: '6px 14px', borderRadius: 999,
              border: active ? '1px solid var(--line-accent)' : '1px solid transparent',
              background: active
                ? 'linear-gradient(180deg, rgba(94,148,255,0.22), rgba(94,148,255,0.06))'
                : 'transparent',
              color: active ? 'var(--accent-bright)' : 'var(--text-muted)',
              transition: 'all .14s',
            }}>
            {lab.short} · {lab.long}
          </button>
        );
      })}
    </div>
  );
}
