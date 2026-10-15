// facts about the selected object: list fields show at once, mesh facts fill in when the fetch lands
export default function ObjectInfo({ object, info }) {
  // rendered even when empty, so the canvas below doesn't jump when the first object is picked
  if (!object) return <p className="info muted">No object selected.</p>
  const d = info?.data
  return (
    <dl className="info">
      <Item term="Object">{object.id}</Item>
      <Item term="Class">{object.class_name}</Item>
      <Item term="Split">
        {object.split}
        {d && d.source_split !== object.split && <span className="muted"> (ModelNet {d.source_split})</span>}
      </Item>
      {info?.error ? (
        <Item term="Mesh"><span className="error">{info.error}</span></Item>
      ) : (
        <>
          <Item term="File">{d ? d.relpath : '…'}</Item>
          <Item term="Vertices">{d ? d.n_vertices.toLocaleString() : '…'}</Item>
          <Item term="Faces">{d ? d.n_faces.toLocaleString() : '…'}</Item>
        </>
      )}
    </dl>
  )
}

function Item({ term, children }) {
  // a div around each dt/dd pair is valid inside a dl, and lets css lay the pairs out in a row
  return (
    <div>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </div>
  )
}
