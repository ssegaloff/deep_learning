// every backend call goes through here, so the /api prefix and the status check live in one place
const BASE = '/api'

// keeps the status and fastapi's parsed "detail" next to the usual message, so a caller can put a 422's
// messages beside their fields; callers that only read .message see the same text as before
export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options)
  // fetch only rejects on network failure; a 404 or 500 resolves normally, so an unchecked error page would be parsed as data
  if (!res.ok) {
    const text = await res.text()
    let detail = null
    // fastapi answers with json ({"detail": ...}), but the vite proxy's 502 is plain text, so parsing is tried, not assumed
    try {
      detail = JSON.parse(text).detail
    } catch {
      // not json: the message below still carries the text
    }
    throw new ApiError(`${options?.method ?? 'GET'} ${path}: ${res.status} ${text}`, res.status, detail)
  }
  return res
}

function postJson(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function getObjects() {
  return (await request('/objects')).json()
}

// a url, not the data: the viewer hands it to PLYLoader, which fetches and caches by url
export function meshUrl(id) {
  return `${BASE}/objects/${encodeURIComponent(id)}/mesh`
}

export async function predict(id) {
  const res = await request('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id }),
  })
  return res.json()
}

export async function getPredictions(split, className) {
  // URLSearchParams escapes the values, so no class name can break the query string
  const query = new URLSearchParams({ split, class_name: className })
  return (await request(`/predictions?${query}`)).json()
}

export async function getObject(id) {
  return (await request(`/objects/${encodeURIComponent(id)}`)).json()
}

export async function getReviews(id) {
  return (await request(`/objects/${encodeURIComponent(id)}/reviews`)).json()
}

// body: { verdict, should_be, mesh_problem, saw_prediction, note }; returns the stored row
export async function addReview(id, body) {
  const res = await request(`/objects/${encodeURIComponent(id)}/reviews`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return res.json()
}
// { [arch]: { values, sources, choices } }: the form's prefill, where each default comes from, and menu options
export async function getRunDefaults() {
  return (await request('/runs/defaults')).json()
}

export async function startRun(body) {
  return (await postJson('/runs', body)).json()
}

export async function listRuns() {
  return (await request('/runs')).json()
}

export async function getRun(id) {
  return (await request(`/runs/${encodeURIComponent(id)}`)).json()
}

export async function cancelRun(id) {
  return (await postJson(`/runs/${encodeURIComponent(id)}/cancel`, {})).json()
}
