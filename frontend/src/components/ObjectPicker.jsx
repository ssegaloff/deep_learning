const SPLITS = ['test', 'val', 'train']

export default function ObjectPicker({ classes, selectedClass, split, objects, selectedId, onClassChange, onSplitChange, onSelect }) {
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
      <p className="count">{objects.length} objects</p>
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
