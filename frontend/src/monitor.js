// pure helpers for the training monitor, kept out of the components so they can be tested without react

// round tick values covering [min, max]: steps of 1, 2 or 5 times a power of ten, with the axis ending on a tick
export function niceScale(min, max, count = 4) {
  if (!(max > min)) max = min + 1
  const raw = (max - min) / count
  const mag = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw)
  // toPrecision trims float noise like 0.7000000000000001 from multiplying by a step that isn't exact in binary
  const clean = (v) => Number(v.toPrecision(12))
  const lo = clean(Math.floor(min / step) * step)
  const hi = clean(Math.ceil(max / step) * step)
  const ticks = []
  for (let i = 0; lo + i * step <= hi + step / 2; i++) ticks.push(clean(lo + i * step))
  return { lo, hi, ticks }
}

// smallest and largest finite value; a diverged run's NaN (or the null it may arrive as) is left out
export function finiteExtent(values) {
  const f = values.filter(Number.isFinite)
  return f.length ? [Math.min(...f), Math.max(...f)] : null
}

// an svg path through the finite points; a non-finite value lifts the pen, so the line breaks there instead of dropping to 0
export function linePath(xs, ys, sx, sy) {
  let d = ''
  let pen = false
  xs.forEach((x, i) => {
    const y = ys[i]
    if (!Number.isFinite(y)) {
      pen = false
      return
    }
    d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`
    pen = true
  })
  return d
}

// the epochs that started at a lower learning rate than the epoch before (the halvings of --lr-step)
export function lrDrops(progress) {
  return progress.filter((p, i) => i > 0 && p.lr < progress[i - 1].lr).map((p) => p.epoch)
}

// the epoch fit() would restore from what's in so far: lowest val loss, the first one on a tie (fit uses strict <);
// isFinite, not just <, because null < Infinity is true in javascript
export function bestEpoch(progress) {
  let best = null
  let bestLoss = Infinity
  for (const p of progress) {
    if (Number.isFinite(p.val_loss) && p.val_loss < bestLoss) {
      bestLoss = p.val_loss
      best = p.epoch
    }
  }
  return best
}

// moves label positions apart (in pixels) so direct labels at the line ends don't overlap; keeps their order
export function spreadLabels(ys, gap = 12) {
  const order = ys.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0])
  for (let k = 1; k < order.length; k++) {
    if (order[k][0] - order[k - 1][0] < gap) order[k][0] = order[k - 1][0] + gap
  }
  const out = []
  for (const [y, i] of order) out[i] = y
  return out
}
