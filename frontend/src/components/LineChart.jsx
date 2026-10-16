import { finiteExtent, linePath, niceScale, spreadLabels } from '../monitor'

// svg coordinates; the svg scales to its container's width, so these only fix the proportions
const W = 560
const H = 220
// the right margin holds the direct labels at the line ends
const M = { top: 16, right: 56, bottom: 32, left: 48 }
const PLOT_W = W - M.left - M.right
const PLOT_H = H - M.top - M.bottom

// one y-axis, epochs on x; hand-written svg, no charting library (section 14 decision)
// series: [{ name, color, values }], values aligned with epochs; markers: [{ epoch, label, at: 'top' | 'bottom' }]
export default function LineChart({ title, epochs, series, markers = [], yMin, yMax, format = (v) => v.toFixed(3) }) {
  const xs = niceScale(0, Math.max(epochs.at(-1) ?? 1, 2), 5)
  // epochs are whole numbers, so a short run's half-step ticks (0.5, 1.5) are dropped
  const xTicks = xs.ticks.filter(Number.isInteger)
  const extent = finiteExtent(series.flatMap((s) => s.values))
  // a fixed bound (loss from 0, accuracy up to 1) keeps the axis from rescaling around small changes
  const ys = niceScale(yMin ?? extent?.[0] ?? 0, yMax ?? extent?.[1] ?? 1)
  const sx = (v) => M.left + ((v - xs.lo) / (xs.hi - xs.lo)) * PLOT_W
  const sy = (v) => M.top + PLOT_H - ((v - ys.lo) / (ys.hi - ys.lo)) * PLOT_H

  // the last finite point of each series anchors its end dot and its direct label
  const ends = series.map((s) => {
    for (let i = s.values.length - 1; i >= 0; i--) {
      if (Number.isFinite(s.values[i])) return { x: epochs[i], y: s.values[i] }
    }
    return null
  })
  const labelYs = spreadLabels(ends.map((e) => (e ? sy(e.y) : -100)))
  const colW = PLOT_W / (xs.hi - xs.lo)

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      {/* a legend as well as direct labels, so a series is never identified by color alone */}
      <ul className="legend">
        {series.map((s) => (
          <li key={s.name}>
            <span className="swatch" style={{ background: s.color }} />
            {s.name}
          </li>
        ))}
      </ul>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title}, by epoch`}>
        <g className="axis">
          {ys.ticks.map((t) => (
            <g key={t}>
              <line x1={M.left} x2={M.left + PLOT_W} y1={sy(t)} y2={sy(t)} className="grid" />
              {/* String, not format: ticks are round numbers, so 0.5 needs no trailing zeros */}
              <text x={M.left - 6} y={sy(t)} textAnchor="end" dominantBaseline="middle">{String(t)}</text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} x={sx(t)} y={H - M.bottom + 16} textAnchor="middle">{t}</text>
          ))}
          <text x={M.left + PLOT_W / 2} y={H - 2} textAnchor="middle">epoch</text>
        </g>
        {markers.map((m) => (
          <g key={`${m.label}-${m.epoch}`} className="marker">
            <line x1={sx(m.epoch)} x2={sx(m.epoch)} y1={M.top} y2={M.top + PLOT_H} />
            {/* two label rows, so markers on the same epoch (a halving at the best epoch) don't overlap */}
            <text x={sx(m.epoch) + 3} y={m.at === 'bottom' ? M.top + PLOT_H - 4 : M.top - 4}>{m.label}</text>
          </g>
        ))}
        {series.map((s, i) => (
          <g key={s.name}>
            <path d={linePath(epochs, s.values, sx, sy)} fill="none" stroke={s.color} strokeWidth="2" strokeLinejoin="round" />
            {/* the end dot keeps a one-epoch run visible, since a path through one point draws nothing */}
            {ends[i] && <circle cx={sx(ends[i].x)} cy={sy(ends[i].y)} r="3.5" fill={s.color} />}
            {ends[i] && (
              <text x={M.left + PLOT_W + 6} y={labelYs[i]} dominantBaseline="middle" className="direct-label">{s.name}</text>
            )}
          </g>
        ))}
        {/* one invisible column per epoch, wider than the line, whose title is the browser's native hover tooltip */}
        {epochs.map((e, i) => (
          <rect key={e} x={sx(e) - colW / 2} y={M.top} width={colW} height={PLOT_H} className="hit">
            <title>{[`epoch ${e}`, ...series.map((s) => `${s.name}: ${Number.isFinite(s.values[i]) ? format(s.values[i]) : 'not finite'}`)].join('\n')}</title>
          </rect>
        ))}
      </svg>
    </figure>
  )
}
