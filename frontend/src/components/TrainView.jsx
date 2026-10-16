import { useEffect, useState } from 'react'
import { ApiError, cancelRun, getRun, getRunDefaults, listRuns, startRun } from '../api'
import { MODEL_LABELS } from '../models'

// layout only: which settings exist, their defaults, sources and menu options all come from GET /runs/defaults
const LABELS = {
  lr: 'Learning rate',
  lr_step: 'Halve the learning rate on a schedule',
  weight_decay: 'Weight decay',
  batch_size: 'Batch size',
  max_epochs: 'Max epochs',
  patience: 'Patience (epochs without a better val loss)',
  dropout: 'Dropout',
  hidden: 'Hidden layers (units per layer)',
  activation: 'Activation',
  seed: 'Seed',
}
// display order; a field the backend adds later still shows, after these, under its own name
const ORDER = Object.keys(LABELS)
// filled in when the schedule is switched on: the pointnet paper's step (appendix c)
const PAPER_LR_STEP = '20'
const POLL_MS = 1000
const NO_ERRORS = { fields: {}, general: [] }

// the form edits strings exactly as typed; they become numbers only on submit
function toDraft(values) {
  const draft = {}
  for (const [key, v] of Object.entries(values)) {
    if (key === 'model') continue
    if (key === 'hidden') draft.hidden = v.map(String)
    else if (key === 'lr_step') {
      draft.lr_step_on = v !== null
      draft.lr_step = v === null ? PAPER_LR_STEP : String(v)
    } else draft[key] = String(v)
  }
  return draft
}

function parseNumber(text) {
  const t = text.trim()
  // Number('') is 0, so an emptied field would silently become 0; empty is a form error instead
  if (t === '') return { error: 'enter a number' }
  const v = Number(t)
  if (!Number.isFinite(v)) return { error: `not a number: ${t}` }
  // a non-integer in an integer field is left to the backend, whose 422 says so by name
  return { value: v }
}

// returns { body } or { errors }; error keys match the 422 mapping below ('lr', 'hidden.0'), so both show in one place
function toBody(model, draft) {
  const body = { model }
  const errors = {}
  const number = (key, text) => {
    const p = parseNumber(text)
    if (p.error) errors[key] = p.error
    return p.value
  }
  for (const [key, text] of Object.entries(draft)) {
    if (key === 'lr_step_on') continue
    if (key === 'activation') body.activation = text
    else if (key === 'hidden') body.hidden = text.map((t, i) => number(`hidden.${i}`, t))
    // off is null, sent explicitly, so the request says "no schedule" rather than leaving it to a default
    else if (key === 'lr_step') body.lr_step = draft.lr_step_on ? number('lr_step', text) : null
    else body[key] = number(key, text)
  }
  return Object.keys(errors).length ? { errors } : { body }
}

// fastapi's 422 detail is a list of { loc: ['body', model, field, index?], msg }; an error with no field goes to the top
function fieldErrors(detail) {
  const fields = {}
  const general = []
  for (const e of Array.isArray(detail) ? detail : [{ msg: String(detail) }]) {
    const [where, , field, index] = e.loc ?? []
    if (where === 'body' && field !== undefined) {
      const key = index === undefined ? field : `${field}.${index}`
      fields[key] = fields[key] ? `${fields[key]}; ${e.msg}` : e.msg
    } else general.push(e.msg)
  }
  return { fields, general }
}

function showDefault(v) {
  if (v === null) return 'off'
  return Array.isArray(v) ? v.join(', ') : String(v)
}

const findRunning = (runs) => runs.find((r) => r.status === 'running') ?? null

export default function TrainView() {
  // { status: 'loading' | 'done' | 'error', data?, message? }
  const [defaults, setDefaults] = useState({ status: 'loading' })
  // pointnet first: it's the selected model
  const [model, setModel] = useState('pointnet')
  // one draft per model, so switching models and back keeps what was typed
  const [drafts, setDrafts] = useState(null)
  const [errors, setErrors] = useState(NO_ERRORS)
  const [starting, setStarting] = useState(false)
  // the latest record of the run this page started or found running: { id, model, status, request, progress?, ... }
  const [run, setRun] = useState(null)
  const [runError, setRunError] = useState(null)
  const [cancelling, setCancelling] = useState(false)

  useEffect(() => {
    let ignore = false
    getRunDefaults()
      .then((data) => {
        if (ignore) return
        setDefaults({ status: 'done', data })
        setDrafts(Object.fromEntries(Object.entries(data).map(([arch, d]) => [arch, toDraft(d.values)])))
      })
      .catch((err) => { if (!ignore) setDefaults({ status: 'error', message: err.message }) })
    // a run keeps going on the server through a reload, so look for one instead of showing an idle form
    listRuns()
      .then((runs) => { if (!ignore) setRun(findRunning(runs)) })
      .catch((err) => { if (!ignore) setRunError(err.message) })
    return () => { ignore = true }
  }, [])

  const runId = run?.id
  const running = run?.status === 'running'
  useEffect(() => {
    if (!running) return
    let stopped = false
    let timer
    function poll() {
      getRun(runId)
        .then((r) => { if (!stopped) { setRun(r); setRunError(null) } })
        // the run goes on without the page; keep polling, so the status comes back when the backend does
        .catch((err) => { if (!stopped) setRunError(err.message) })
        // the next poll is scheduled only after this one answers, so a slow backend can't stack requests up
        .finally(() => { if (!stopped) timer = setTimeout(poll, POLL_MS) })
    }
    timer = setTimeout(poll, POLL_MS)
    // stops when the run ends or the id changes; strict mode's extra mount/unmount also lands here
    return () => {
      stopped = true
      clearTimeout(timer)
    }
  }, [runId, running])

  if (defaults.status === 'loading' || (defaults.status === 'done' && !drafts)) {
    return <section className="train"><p>Loading training settings…</p></section>
  }
  if (defaults.status === 'error') {
    return <section className="train"><p className="error">Could not load training settings. {defaults.message}</p></section>
  }

  const { values, sources, choices } = defaults.data[model]
  const draft = drafts[model]
  const fieldError = (key) => errors.fields[key]

  const update = (patch) => setDrafts((d) => ({ ...d, [model]: { ...d[model], ...patch } }))
  function changeModel(arch) {
    setModel(arch)
    // errors name fields of the model they came from, so they'd sit under the wrong form
    setErrors(NO_ERRORS)
  }
  function reset() {
    update(toDraft(values))
    setErrors(NO_ERRORS)
  }

  function findActive() {
    listRuns().then((runs) => setRun(findRunning(runs))).catch((err) => setRunError(err.message))
  }

  function submit(e) {
    // a form submit would otherwise reload the page
    e.preventDefault()
    const parsed = toBody(model, draft)
    if (parsed.errors) {
      setErrors({ fields: parsed.errors, general: [] })
      return
    }
    setErrors(NO_ERRORS)
    setStarting(true)
    startRun(parsed.body)
      .then((record) => {
        setRun(record)
        setRunError(null)
      })
      .catch((err) => {
        const api = err instanceof ApiError
        if (api && err.status === 422) setErrors(fieldErrors(err.detail))
        else setErrors({ fields: {}, general: [api && typeof err.detail === 'string' ? err.detail : err.message] })
        // 409: a run is already going (perhaps started from another tab or curl); pick it up so its status shows
        if (api && err.status === 409) findActive()
      })
      .finally(() => setStarting(false))
  }

  function cancel() {
    setCancelling(true)
    cancelRun(run.id)
      .then((record) => setRun(record))
      // a 409 here means it ended on its own first; the next poll shows how
      .catch((err) => setRunError(err.message))
      .finally(() => setCancelling(false))
  }

  const keys = [...ORDER.filter((k) => k in draft), ...Object.keys(draft).filter((k) => !ORDER.includes(k) && k !== 'lr_step_on')]

  // every input gets a name: chrome's autofill and devtools' issues panel expect one, and the field key is unique per form
  function renderField(key) {
    const source = (
      <span className="muted small">
        Default {showDefault(values[key])}: {sources[key]}
      </span>
    )
    const err = (k) => fieldError(k) && <span className="error small">{fieldError(k)}</span>

    if (key === 'hidden') {
      return (
        <fieldset key={key} className="field">
          <legend>{LABELS.hidden}</legend>
          <div className="layers">
            {/* index keys are safe: layers are only added or removed at the end, never reordered */}
            {draft.hidden.map((text, i) => (
              <span key={i} className="layer">
                <input
                  type="text"
                  inputMode="numeric"
                  name={`hidden-${i}`}
                  aria-label={`Layer ${i + 1} units`}
                  aria-invalid={Boolean(fieldError(`hidden.${i}`))}
                  value={text}
                  onChange={(e) => update({ hidden: draft.hidden.map((t, j) => (j === i ? e.target.value : t)) })}
                />
                {err(`hidden.${i}`)}
              </span>
            ))}
            {/* a new layer starts at the last layer's size, the most likely next edit */}
            <button type="button" onClick={() => update({ hidden: [...draft.hidden, draft.hidden.at(-1)] })}>
              + layer
            </button>
            {/* no limit on adding (the backend's 422 says when there are too many), but at least one layer:
                with none, the mlp would be logistic regression, which hw2 rules out */}
            <button type="button" disabled={draft.hidden.length === 1} onClick={() => update({ hidden: draft.hidden.slice(0, -1) })}>
              − layer
            </button>
          </div>
          {err('hidden')}
          {source}
        </fieldset>
      )
    }
    if (key === 'activation') {
      return (
        <label key={key} className="field">
          {LABELS.activation}
          <select name="activation" value={draft.activation} onChange={(e) => update({ activation: e.target.value })}>
            {choices.activation.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
          {err('activation')}
          {source}
        </label>
      )
    }
    if (key === 'lr_step') {
      return (
        <div key={key} className="field">
          <label className="choice">
            <input type="checkbox" name="lr_step_on" checked={draft.lr_step_on} onChange={(e) => update({ lr_step_on: e.target.checked })} />
            {LABELS.lr_step}
          </label>
          {draft.lr_step_on && (
            <label className="inline">
              every
              <input
                type="text"
                inputMode="numeric"
                name="lr_step"
                aria-invalid={Boolean(fieldError('lr_step'))}
                value={draft.lr_step}
                onChange={(e) => update({ lr_step: e.target.value })}
              />
              epochs
            </label>
          )}
          {err('lr_step')}
          {source}
        </div>
      )
    }
    return (
      <label key={key} className="field">
        {LABELS[key] ?? key}
        <input
          type="text"
          inputMode="decimal"
          name={key}
          aria-invalid={Boolean(fieldError(key))}
          value={draft[key]}
          onChange={(e) => update({ [key]: e.target.value })}
        />
        {err(key)}
        {source}
      </label>
    )
  }

  return (
    <section className="train">
      <h2>Train a model</h2>
      <p className="muted">
        Each run trains a fresh model in its own folder under <code>runs/</code>; the level-1 weights the Explore view uses are never changed.
      </p>
      {run && <RunStatus run={run} error={runError} cancelling={cancelling} onCancel={cancel} />}
      {!run && runError && <p className="error">Could not check for a running job. {runError}</p>}
      <form className="train-form" onSubmit={submit}>
        <fieldset className="field">
          <legend>Model</legend>
          {Object.keys(defaults.data).map((arch) => (
            <label key={arch} className="choice">
              <input type="radio" name="train-model" value={arch} checked={model === arch} onChange={() => changeModel(arch)} />
              {MODEL_LABELS[arch] ?? arch}
            </label>
          ))}
        </fieldset>
        {errors.general.map((msg) => <p key={msg} className="error">{msg}</p>)}
        {keys.map(renderField)}
        <div className="actions">
          {/* one run at a time (the backend answers 409 otherwise), so start waits for the current one */}
          <button type="submit" className="save" disabled={starting || running}>
            {starting ? 'Starting…' : 'Start training'}
          </button>
          <button type="button" onClick={reset}>Reset to defaults</button>
          {running && <span className="muted small">A run is in progress; start another when it ends or is cancelled.</span>}
        </div>
      </form>
    </section>
  )
}

function RunStatus({ run, error, cancelling, onCancel }) {
  const epochs = run.progress?.length ?? 0
  let detail
  if (run.status === 'running') detail = epochs ? `epoch ${epochs} of at most ${run.request.max_epochs} finished` : 'starting'
  else detail = `after ${epochs} epoch${epochs === 1 ? '' : 's'}`
  return (
    <div className="run-status" role="status">
      <p>
        <strong>{MODEL_LABELS[run.model] ?? run.model}</strong> run <code>{run.id}</code>: {run.status}, {detail}
      </p>
      {run.status === 'running' && (
        <button type="button" onClick={onCancel} disabled={cancelling}>
          {cancelling ? 'Cancelling…' : 'Cancel run'}
        </button>
      )}
      {error && <p className="error small">Status unavailable, still retrying. {error}</p>}
    </div>
  )
}
