import { MODEL_LABELS } from '../models'
import { bestEpoch, lrDrops } from '../monitor'
import LineChart from './LineChart'

const TRAIN = 'var(--series-1)'
const VAL = 'var(--series-2)'
// a diverged epoch's loss is NaN in the file and may arrive as null; either way it shows as a dash
const fmt = (v) => (Number.isFinite(v) ? v.toFixed(3) : '–')
const SPLITS = [
  ['train', 'Train'],
  ['val', 'Val'],
  // the course asks for test numbers; choosing a run by them would make them optimistic (decided oct 16)
  ['test', 'Test (not for choosing)'],
]

function settings(request) {
  return Object.entries(request)
    .filter(([key]) => key !== 'model')
    .map(([key, v]) => `${key} ${v === null ? 'off' : Array.isArray(v) ? v.join('×') : v}`)
    .join(' · ')
}

// one run's curves and, once it has them, its final metrics, confusion matrix, or log
export default function RunMonitor({ record }) {
  const progress = record.progress ?? []
  const epochs = progress.map((p) => p.epoch)
  const col = (key) => progress.map((p) => p[key])
  // a finished run says which epoch fit() restored; while running, the same rule over the epochs so far
  const best = record.results?.config?.best_epoch ?? bestEpoch(progress)
  const markers = [
    ...lrDrops(progress).map((epoch) => ({ epoch, label: 'lr ½' })),
    ...(best ? [{ epoch: best, label: 'best', at: 'bottom' }] : []),
  ]

  return (
    <section className="monitor" aria-label="Training monitor">
      <h3>
        {MODEL_LABELS[record.model] ?? record.model} · {new Date(record.created_at).toLocaleString()} · {record.status}
      </h3>
      <p className="muted small">{settings(record.request)}</p>
      {progress.length === 0 ? (
        <p className="muted">No epochs finished yet.</p>
      ) : (
        <>
          <LineChart
            title="Loss (cross-entropy)"
            epochs={epochs}
            yMin={0}
            markers={markers}
            series={[
              { name: 'train', color: TRAIN, values: col('train_loss') },
              { name: 'val', color: VAL, values: col('val_loss') },
            ]}
          />
          <LineChart
            title="Accuracy"
            epochs={epochs}
            yMax={1}
            markers={markers}
            series={[
              { name: 'train', color: TRAIN, values: col('train_accuracy') },
              { name: 'val', color: VAL, values: col('val_accuracy') },
            ]}
          />
          <p className="muted small">
            Train values are averages over each epoch while the weights were changing and dropout was on, so they read a
            little worse than an eval-mode pass. “best” marks the lowest val loss, the epoch whose weights are kept.
          </p>
          <EpochTable progress={progress} />
        </>
      )}
      {record.status === 'done' && record.results && <Results results={record.results} />}
      {record.status === 'failed' && (
        <>
          <h4>Last lines of the log</h4>
          <pre className="log">{(record.log_tail ?? []).join('\n')}</pre>
        </>
      )}
    </section>
  )
}

// the exact numbers behind the curves, and the table view for anyone who can't read the chart
function EpochTable({ progress }) {
  return (
    <details>
      <summary>Per-epoch numbers</summary>
      <div className="table-scroll">
        <table className="numbers">
          <thead>
            <tr>
              <th scope="col">Epoch</th>
              <th scope="col">lr</th>
              <th scope="col">Train loss</th>
              <th scope="col">Train acc.</th>
              <th scope="col">Val loss</th>
              <th scope="col">Val acc.</th>
              <th scope="col">Val macro-F1</th>
              <th scope="col">Seconds</th>
            </tr>
          </thead>
          <tbody>
            {progress.map((p) => (
              <tr key={p.epoch}>
                <th scope="row">{p.epoch}</th>
                {/* runs from before the lr schedule have no lr in their progress */}
                <td>{p.lr === undefined ? '–' : p.lr.toExponential(2)}</td>
                <td>{fmt(p.train_loss)}</td>
                <td>{fmt(p.train_accuracy)}</td>
                <td>{fmt(p.val_loss)}</td>
                <td>{fmt(p.val_accuracy)}</td>
                <td>{fmt(p.val_macro_f1)}</td>
                <td>{fmt(p.seconds)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  )
}

function Results({ results }) {
  return (
    <>
      <h4>Final metrics</h4>
      <table className="numbers">
        <thead>
          <tr>
            <th scope="col">Split</th>
            <th scope="col">Loss</th>
            <th scope="col">Accuracy</th>
            <th scope="col">Macro-F1</th>
            <th scope="col">Balanced acc.</th>
          </tr>
        </thead>
        <tbody>
          {SPLITS.map(([key, label]) => (
            <tr key={key} className={key === 'test' ? 'test-row' : undefined}>
              <th scope="row">{label}</th>
              <td>{fmt(results[key].loss)}</td>
              <td>{fmt(results[key].accuracy)}</td>
              <td>{fmt(results[key].macro_f1)}</td>
              <td>{fmt(results[key].balanced_accuracy)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        Weights from epoch {results.config.best_epoch} of {results.history.length}; {results.n_parameters.toLocaleString()}{' '}
        parameters. All three rows are an eval-mode pass with those weights. Compare runs on val; the reported model is
        the level-1 one in the README.
      </p>
      <Confusion classes={results.classes} matrix={results.confusion_matrix} />
    </>
  )
}

// rows are true classes and columns predicted, the order train.py writes; recall is the diagonal over the row total
function Confusion({ classes, matrix }) {
  return (
    <div className="table-scroll">
      <table className="numbers confusion">
        <caption>Test predictions: rows are the true class, columns the predicted class</caption>
        <thead>
          <tr>
            <th scope="col">True \ predicted</th>
            {classes.map((c) => <th key={c} scope="col">{c}</th>)}
            <th scope="col">Recall</th>
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0)
            return (
              <tr key={classes[i]}>
                <th scope="row">{classes[i]}</th>
                {row.map((n, j) => (
                  <td key={classes[j]} className={i === j ? 'diag' : n === 0 ? 'zero' : undefined}>{n}</td>
                ))}
                <td>{total ? (row[i] / total).toFixed(2) : '–'}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
