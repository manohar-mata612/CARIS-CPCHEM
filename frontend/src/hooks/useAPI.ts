// hooks/useAPI.ts
// Handles all FastAPI communication with JWT token

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export async function apiLogin(username: string, password: string) {
  const res = await fetch(`${API_BASE}/api/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) throw new Error('Login failed')
  return res.json()
}

export async function apiFetch(path: string, token: string, options: RequestInit = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  })
  if (!res.ok) throw new Error(`API error: ${res.status}`)
  return res.json()
}

export async function sendSensorReading(token: string, reading: object) {
  return apiFetch('/api/sensor-reading', token, {
    method: 'POST',
    body: JSON.stringify(reading),
  })
}

export async function acknowledgeWorkOrder(token: string, woNumber: string) {
  return apiFetch(`/api/acknowledge/${woNumber}`, token, { method: 'POST' })
}

// Sample fault readings for demo button
export const SAMPLE_READINGS = {
  normal: {
    equipment_id: 'CB-CGC-001', vibration_rms: 1.2,
    peak: 4.1, kurtosis: 3.1, crest_factor: 3.4, std: 0.7,
  },
  inner_race: {
    equipment_id: 'CB-CGC-001', vibration_rms: 4.8,
    peak: 16.2, kurtosis: 11.3, crest_factor: 6.8, std: 2.1,
  },
  outer_race: {
    equipment_id: 'CB-CGC-002', vibration_rms: 3.9,
    peak: 12.4, kurtosis: 8.7, crest_factor: 5.9, std: 1.8,
  },
  ball_fault: {
    equipment_id: 'CB-QWP-001', vibration_rms: 3.1,
    peak: 10.2, kurtosis: 6.4, crest_factor: 5.2, std: 1.4,
  },
}
