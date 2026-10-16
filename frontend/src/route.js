import { useSyncExternalStore } from 'react'

// display names; the keys are what appears after #/
export const VIEWS = { explore: 'Explore', train: 'Train' }
const DEFAULT_VIEW = 'explore'

// read from the url on every render, so the url is the only copy of which view is open
function currentView() {
  const name = window.location.hash.replace(/^#\/?/, '')
  // hasOwn, not `in`: `'toString' in VIEWS` is true through the prototype, and would render no view at all
  return Object.hasOwn(VIEWS, name) ? name : DEFAULT_VIEW
}

function subscribe(onChange) {
  window.addEventListener('hashchange', onChange)
  return () => window.removeEventListener('hashchange', onChange)
}

export function useView() {
  return useSyncExternalStore(subscribe, currentView)
}
