// every backend call goes through here, so the /api prefix and the status check live in one place
const BASE = '/api'

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options)
  // fetch only rejects on network failure; a 404 or 500 resolves normally, so an unchecked error page would be parsed as data
  if (!res.ok) {
    throw new Error(`${options?.method ?? 'GET'} ${path}: ${res.status} ${await res.text()}`)
  }
  return res
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