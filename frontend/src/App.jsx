import { useEffect, useMemo, useState } from 'react'
import { getObjects } from './api'
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

  useEffect(() => {
    // strict mode runs effects twice in dev; the flag drops the first run's response instead of setting state from it
    let ignore = false
    getObjects()
      .then((data) => { if (!ignore) setObjects(data) })
      .catch((err) => { if (!ignore) setError(err.message) })
    return () => { ignore = true }
  }, [])

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

  // cleared in the handler, not an effect, so no render ever shows a selection outside the list
  function changeClass(c) {
    setSelectedClass(c)
    setSelectedId(null)
  }
  function changeSplit(s) {
    setSplit(s)
    setSelectedId(null)
  }

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
          objects={visible}
          selectedId={selectedId}
          onClassChange={changeClass}
          onSplitChange={changeSplit}
          onSelect={setSelectedId}
        />
      </aside>
      <main className="main">
        {/* always mounted: the canvas lives as long as the page, so changing class or split never tears down its webgl context */}
        <MeshViewer id={selectedId} />
      </main>
    </div>
  )
}
