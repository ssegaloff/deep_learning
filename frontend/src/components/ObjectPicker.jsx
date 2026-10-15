import { FILTERS } from '../models'

const SPLITS = ['test', 'val', 'train']

function Count({ filter, shown, total, loadState }) {
  if (loadState?.status === 'loading') return <p className="count">Loading predictions…</p>
  if (loadState?.status === 'error') return <p className="count error">Could not load predictions. {loadState.message}</p>
  return <p className="count">{filter === 'all' ? `${shown} objects` : `${shown} of ${total} objects`}</p>
}

export default function ObjectPicker({
  classes, selectedClass, split, filter, objects, total, loadState, selectedId,
  onClassChange, onSplitChange, onFilterChange, onSelect,
}) {
  return (
    <div className="picker">
      {/* wrapping label: the select gets its accessible name without an id/htmlFor pair */}
      <label>
        Class
        <select value={selectedClass} onChange={(e) => onClassChange(e.target.value)}>
          {classes.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </label>
      <label>
        Split
        <select value={split} onChange={(e) => onSplitChange(e.target.value)}>
          {SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </label>
      <label>
        Show
        <select value={filter} onChange={(e) => onFilterChange(e.target.value)}>
          {Object.entries(FILTERS).map(([key, f]) => <option key={key} value={key}>{f.label}</option>)}
        </select>
      </label>
      <Count filter={filter} shown={objects.length} total={total} loadState={loadState} />
      <ul className="object-list">
        {objects.map((o) => (
          // ids are unique and stable, so react keeps each row's identity when the filter changes
          <li key={o.id}>
            {/* aria-pressed carries the selection for screen readers, and the css styles off the same attribute */}
            <button type="button" aria-pressed={o.id === selectedId} onClick={() => onSelect(o.id)}>
              {o.id}
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
