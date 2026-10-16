import { VIEWS } from '../route'

export default function NavBar({ view }) {
  return (
    <nav className="nav">
      <h1>ModelNet10 classifier</h1>
      {Object.entries(VIEWS).map(([name, label]) => (
        // aria-current tells a screen reader which link is the open view; the css styles off the same attribute
        <a key={name} href={`#/${name}`} aria-current={view === name ? 'page' : undefined}>
          {label}
        </a>
      ))}
    </nav>
  )
}
