import { useEffect, useState } from 'react'
import { addReview, getReviews } from '../api'

const VERDICTS = [
  { value: 'ok', label: 'Label is right' },
  { value: 'wrong', label: 'Label is wrong' },
  { value: 'ambiguous', label: 'Ambiguous' },
]
// the form after a save: the newest history row already shows what was saved
const EMPTY = { verdict: null, shouldBe: '', meshProblem: false, note: '' }

// mounted with key={id} by App, so every object starts with an empty form and its own history
export default function ReviewPanel({ id, trueClass, classes, sawPrediction }) {
  const [form, setForm] = useState(EMPTY)
  // { status: 'loading' | 'done' | 'error', rows?, message? }
  const [history, setHistory] = useState({ status: 'loading' })
  // { status: 'idle' | 'saving' | 'error', message? }
  const [save, setSave] = useState({ status: 'idle' })

  useEffect(() => {
    // one panel per object, so the flag only has to cover strict mode's double run and unmounting
    let ignore = false
    getReviews(id)
      .then((rows) => { if (!ignore) setHistory({ status: 'done', rows }) })
      .catch((err) => { if (!ignore) setHistory({ status: 'error', message: err.message }) })
    return () => { ignore = true }
  }, [id])

  const update = (patch) => setForm((f) => ({ ...f, ...patch }))
  // mirrors the backend's rule, so the button explains the 422 before it can happen
  const complete = form.verdict && (form.verdict !== 'wrong' || form.shouldBe)
  const saving = save.status === 'saving'

  function submit(e) {
    // a form submit would otherwise reload the page
    e.preventDefault()
    setSave({ status: 'saving' })
    addReview(id, {
      verdict: form.verdict,
      // a class picked under "wrong" stays in the form if the verdict changes, so it's sent only for "wrong"
      should_be: form.verdict === 'wrong' ? form.shouldBe : null,
      mesh_problem: form.meshProblem,
      // read at save time: what was on screen when the verdict was given, not when the panel opened
      saw_prediction: sawPrediction,
      note: form.note,
    })
      .then((row) => {
        // the server's row, not the form: it carries the review_id, timestamp, and stored note
        setHistory((h) => (h.status === 'done' ? { ...h, rows: [row, ...h.rows] } : h))
        setForm(EMPTY)
        setSave({ status: 'idle' })
      })
      .catch((err) => setSave({ status: 'error', message: err.message }))
  }

  return (
    <section className="review" aria-labelledby="review-heading">
      <h2 id="review-heading">Label review</h2>
      <form onSubmit={submit}>
        <fieldset>
          <legend>Is “{trueClass}” the right label?</legend>
          {VERDICTS.map((v) => (
            <label key={v.value} className="choice">
              <input
                type="radio"
                name="verdict"
                value={v.value}
                checked={form.verdict === v.value}
                onChange={() => update({ verdict: v.value })}
              />
              {v.label}
            </label>
          ))}
        </fieldset>
        {form.verdict === 'wrong' && (
          <label className="field">
            Should be
            <select value={form.shouldBe} onChange={(e) => update({ shouldBe: e.target.value })}>
              <option value="" disabled>Pick a class</option>
              {/* the current label is excluded: "wrong, should be table" on a table is a contradiction the backend rejects */}
              {classes.filter((c) => c !== trueClass).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        )}
        <label className="choice">
          <input
            type="checkbox"
            checked={form.meshProblem}
            onChange={(e) => update({ meshProblem: e.target.checked })}
          />
          Mesh problem (broken, incomplete, or odd geometry)
        </label>
        <label className="field">
          Note
          <textarea rows={2} value={form.note} onChange={(e) => update({ note: e.target.value })} />
        </label>
        <button type="submit" className="save" disabled={!complete || saving}>
          {saving ? 'Saving…' : 'Save review'}
        </button>
        {/* says what will be recorded, so the reviewer knows the flag exists */}
        <p className="muted small">
          {sawPrediction ? 'Recorded as given after seeing the prediction.' : 'Recorded as given without seeing the prediction.'}
        </p>
        {save.status === 'error' && <p className="error">Save failed. {save.message}</p>}
      </form>
      <History history={history} />
    </section>
  )
}

function History({ history }) {
  if (history.status === 'loading') return <p className="muted">Loading reviews…</p>
  if (history.status === 'error') return <p className="error">Could not load reviews. {history.message}</p>
  if (history.rows.length === 0) return <p className="muted">Not reviewed yet.</p>
  return (
    // newest first, as the backend sends it: the top entry is the current verdict
    <ol className="history">
      {history.rows.map((r, i) => (
        <li key={r.review_id} className={i === 0 ? 'current' : undefined}>
          <strong>{r.verdict === 'wrong' ? `wrong → ${r.should_be}` : r.verdict}</strong>
          {i === 0 && <span className="tag">current</span>}
          {r.mesh_problem && <span className="tag">mesh problem</span>}
          {r.saw_prediction && <span className="tag">saw prediction</span>}
          {/* stored in utc with an offset, shown in local time */}
          <time className="muted small" dateTime={r.created_at}>{new Date(r.created_at).toLocaleString()}</time>
          {r.note && <p className="note">{r.note}</p>}
        </li>
      ))}
    </ol>
  )
}
