import { useEffect, useMemo, useState } from 'react'
import { getObject, getObjects, getPredictions, predict } from './api'
import ObjectInfo from './components/ObjectInfo'
import About from './components/About'
import { FILTERS } from './models'
import PredictionPanel from './components/PredictionPanel'
import ObjectPicker from './components/ObjectPicker'
import './App.css'
import MeshViewer from './components/MeshViewer'


export default function App() {
  const [objects, setObjects] = useState(null)
  const [error, setError] = useState(null)
  const [selectedClass, setSelectedClass] = useState(null)
  // test by default: those are the objects the models never trained on, so their predictions mean something
  const [split, setSplit] = useState('test')
  const [selectedId, setSelectedId] = useState(null)
  // { id, status: 'loading' | 'done' | 'error', data?, message? }: tagged with its object, so it can be matched to the selection
  const [result, setResult] = useState(null)
  const [filter, setFilter] = useState('all')
  // predicted labels per `${split}/${class}`; keyed, so a late response lands under its own class and split, never the current one
  const [labels, setLabels] = useState({})
  // { id, data } or { id, error }: tagged with its object, like result
  const [info, setInfo] = useState(null)

  useEffect(() => {
    // strict mode runs effects twice in dev; the flag drops the first run's response instead of setting state from it
    let ignore = false
    getObjects()
      .then((data) => { if (!ignore) setObjects(data) })
      .catch((err) => { if (!ignore) setError(err.message) })
    return () => { ignore = true }
  }, [])

  useEffect(() => {
    if (!selectedId) return
    // a fast click sequence can return out of order; the flag drops every response but the current selection's
    let ignore = false
    getObject(selectedId)
      .then((data) => { if (!ignore) setInfo({ id: selectedId, data }) })
      .catch((err) => { if (!ignore) setInfo({ id: selectedId, error: err.message }) })
    return () => { ignore = true }
  }, [selectedId])

  // display only: labels travel as class-name strings, so this order can't shift a prediction
  const classes = useMemo(
    () => (objects ? [...new Set(objects.map((o) => o.class_name))] : []),
    [objects],
  )
  // derived, not stored: no effect needed to pick a default once the objects arrive
  const activeClass = selectedClass ?? classes[0]
  const visible = useMemo(
    () => (objects ? objects.filter((o) => o.class_name === activeClass && o.split === split) : []),
    [objects, activeClass, split],
  )
  const entry = filter === 'all' ? null : labels[`${split}/${activeClass}`]
  const listed = useMemo(() => {
    if (filter === 'all') return visible
    if (entry?.status !== 'done') return []
    return visible.filter((o) => FILTERS[filter].keep(entry.byId[o.id], o.class_name))
  }, [filter, entry, visible])

  // cleared in the handler, not an effect, so no render ever shows a selection outside the list
  function loadLabels(s, c) {
    const key = `${s}/${c}`
    // fixed weights give fixed predictions, so each class and split is fetched once; a failed fetch is retried next time
    if (labels[key] && labels[key].status !== 'error') return
    setLabels((l) => ({ ...l, [key]: { status: 'loading' } }))
    getPredictions(s, c)
      .then((rows) => {
        const byId = Object.fromEntries(rows.map((r) => [r.id, r.predicted]))
        setLabels((l) => ({ ...l, [key]: { status: 'done', byId } }))
      })
      .catch((err) => setLabels((l) => ({ ...l, [key]: { status: 'error', message: err.message } })))
  }

  // cleared in the handler, not an effect, so no render ever shows a selection outside the list;
  // the fetch also starts here, because each of these user actions is what makes the labels needed
  function changeClass(c) {
    setSelectedClass(c)
    setSelectedId(null)
    if (filter !== 'all') loadLabels(split, c)
  }
  function changeSplit(s) {
    setSplit(s)
    setSelectedId(null)
    if (filter !== 'all') loadLabels(s, activeClass)
  }
  function changeFilter(f) {
    setFilter(f)
    setSelectedId(null)
    if (f !== 'all') loadLabels(split, activeClass)
  }
  function classify(id) {
    setResult({ id, status: 'loading' })
    predict(id)
      .then((data) => {
        // the backend echoes the id; a mismatch would show one object's prediction for another, so fail loudly
        if (data.id !== id) throw new Error(`asked for ${id}, got ${data.id}`)
        // functional update: a slow response for an earlier click only writes if its object is still the latest request
        setResult((r) => (r?.id === id ? { id, status: 'done', data } : r))
      })
      .catch((err) => setResult((r) => (r?.id === id ? { id, status: 'error', message: err.message } : r)))
  }

  // derived: a result is shown only under its own object, so a late response can never sit under a different mesh
  const shown = result?.id === selectedId ? result : null
  // derived: while a new object's info is in flight, the stored info belongs to the old id and isn't shown
  const shownInfo = info?.id === selectedId ? info : null
  const selected = selectedId ? objects?.find((o) => o.id === selectedId) : null

  if (error) return <p className="status error">Could not load objects. {error}</p>
  if (!objects) return <p className="status">Loading objects…</p>

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>ModelNet10 classifier</h1>
        <ObjectPicker
          classes={classes}
          selectedClass={activeClass}
          split={split}
          filter={filter}
          objects={listed}
          total={visible.length}
          loadState={entry}
          selectedId={selectedId}
          onClassChange={changeClass}
          onSplitChange={changeSplit}
          onFilterChange={changeFilter}
          onSelect={setSelectedId}
        />
      </aside>
      <main className="main">
        <ObjectInfo object={selected} info={shownInfo} />
        {/* always mounted: the canvas lives as long as the page, so changing class or split never tears down its webgl context */}
        <MeshViewer id={selectedId} />
      </main>
      <aside className="results">
        {/* about while nothing is selected; the classify panel takes over once there's an object to classify */}
        {selectedId ? <PredictionPanel selectedId={selectedId} result={shown} onClassify={classify} /> : <About />}
      </aside>
    </div>
  )
}
