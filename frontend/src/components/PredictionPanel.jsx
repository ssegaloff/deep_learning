// display names for the backend's architecture keys; an unknown key falls back to the key itself
import { MODEL_LABELS } from '../models'

const pct = (p) => `${(p * 100).toFixed(1)}%`

export default function PredictionPanel({ selectedId, result, onClassify }) {
  if (!selectedId) return <p className="muted">Select an object to classify it.</p>
  const loading = result?.status === 'loading'
  return (
    <div className="prediction">
      <button type="button" className="classify" onClick={() => onClassify(selectedId)} disabled={loading}>
        {loading ? 'Classifying…' : 'Classify'}
      </button>
      {result?.status === 'error' && <p className="error">Prediction failed. {result.message}</p>}
      {result?.status === 'done' && <ProbabilityTable data={result.data} />}
    </div>
  )
}

function ProbabilityTable({ data }) {
  const archs = Object.keys(data.predictions)
  // rows follow the backend's class order (meta.json), the same order as the confusion matrices, so both columns line up
  const classes = Object.keys(data.predictions[archs[0]].probabilities)
  return (
    <>
      <p className="truth">
        True class: <strong>{data.true_class}</strong> <span className="muted">({data.split})</span>
      </p>
      {/* a real table: each model is identified by its column, not by color, and screen readers get row and column headers */}
      <table className="probs">
        <thead>
          <tr>
            <th scope="col">Class</th>
            {archs.map((a) => <th scope="col" key={a}>{MODEL_LABELS[a] ?? a}</th>)}
          </tr>
          <tr>
            <th scope="row">Predicted</th>
            {archs.map((a) => {
              const predicted = data.predictions[a].predicted
              const correct = predicted === data.true_class
              return (
                // icon plus class name carries the verdict; the color only reinforces it
                <td key={a} className={correct ? 'verdict good' : 'verdict bad'}>
                  {correct ? '✓' : '✗'} {predicted}
                </td>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {classes.map((c) => (
            <tr key={c} className={c === data.true_class ? 'is-true' : undefined}>
              <th scope="row">
                {c}
                {c === data.true_class && <span className="tag">true</span>}
              </th>
              {archs.map((a) => {
                const p = data.predictions[a].probabilities[c]
                return (
                  // exact value on hover; printed only from 1% up, so the few classes that matter aren't drowned out
                  <td key={a} title={`${MODEL_LABELS[a] ?? a}, ${c}: ${pct(p)}`}>
                    <span className="cell">
                      <span className="track"><span className="bar" style={{ width: pct(p) }} /></span>
                      <span className="value">{p >= 0.01 ? pct(p) : ''}</span>
                    </span>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}
