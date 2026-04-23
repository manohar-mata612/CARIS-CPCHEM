'use client'
import { useState, useEffect, useCallback } from 'react'
import {
  Activity, AlertTriangle, CheckCircle, Clock,
  Zap, Settings, RefreshCw, LogIn, ChevronRight,
  Wrench, FileText, BarChart2, Shield
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { apiLogin, apiFetch, sendSensorReading, acknowledgeWorkOrder, SAMPLE_READINGS } from '@/hooks/useAPI'

// ── Types ───────────────────────────────────────────────────
interface Alert {
  equipment_id: string; severity: string; fault_type: string
  priority: string; diagnosis: string; timestamp: string
  vibration_rms?: number; kurtosis?: number; status?: string
  saved_at?: string
}
interface WorkOrder {
  work_order_number: string; equipment_id: string; priority: string
  failure_code: string; short_description: string; parts_required?: string[]
  estimated_labor_hrs?: number; required_start?: string; status?: string
  saved_at?: string
}
interface LogEntry {
  agent: string; action: string; timestamp: string
  diagnosis?: string; severity?: string; error?: string
}

// ── Color helpers ────────────────────────────────────────────
const priorityColor = (p: string) => ({
  P1: '#EF4444', P2: '#F59E0B', P3: '#3B82F6', normal: '#22C55E'
}[p] || '#9896B8')

const severityBadge = (s: string) => {
  const map: Record<string, string> = {
    critical: 'rgba(239,68,68,0.15)', warning: 'rgba(245,158,11,0.15)',
    normal: 'rgba(34,197,94,0.15)', error: 'rgba(239,68,68,0.15)'
  }
  return map[s] || 'rgba(152,150,184,0.1)'
}

const fmt = (ts: string) => {
  if (!ts) return '—'
  try { return new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' }) }
  catch { return ts.slice(11, 19) }
}

// ── Login Screen ─────────────────────────────────────────────
function LoginScreen({ onLogin }: { onLogin: (t: string, u: string) => void }) {
  const [user, setUser] = useState('engineer')
  const [pass, setPass] = useState('cpchem2025')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (e: React.FormEvent) => {
    e.preventDefault(); setLoading(true); setError('')
    try {
      const data = await apiLogin(user, pass)
      onLogin(data.access_token, data.username)
    } catch { setError('Invalid credentials. Try engineer / cpchem2025') }
    finally { setLoading(false) }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-base)' }}>
      <div style={{ width: 420, animation: 'fade-in 0.4s ease' }}>
        {/* Frame logo area */}
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <div style={{ width: 36, height: 36, background: 'var(--frame-teal)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Activity size={20} color="#0F0E1A" />
            </div>
            <span style={{ fontSize: 20, fontWeight: 600, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>CARIS</span>
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>Cedar Bayou Reliability Intelligence System</p>
          <p style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 4 }}>Powered by Frame Data AI</p>
        </div>

        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 16, padding: 32 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 6 }}>Sign in</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: 12, marginBottom: 24 }}>CPChem Cedar Bayou — authorized personnel only</p>

          <form onSubmit={submit}>
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>Username</label>
              <input value={user} onChange={e => setUser(e.target.value)}
                style={{ width: '100%', background: 'var(--bg-base)', border: '1px solid var(--bg-border)', borderRadius: 8, padding: '10px 14px', color: 'var(--text-primary)', fontSize: 14, outline: 'none' }} />
            </div>
            <div style={{ marginBottom: 24 }}>
              <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>Password</label>
              <input type="password" value={pass} onChange={e => setPass(e.target.value)}
                style={{ width: '100%', background: 'var(--bg-base)', border: '1px solid var(--bg-border)', borderRadius: 8, padding: '10px 14px', color: 'var(--text-primary)', fontSize: 14, outline: 'none' }} />
            </div>
            {error && <p style={{ color: '#EF4444', fontSize: 12, marginBottom: 16 }}>{error}</p>}
            <button type="submit" disabled={loading}
              style={{ width: '100%', background: 'var(--frame-indigo)', border: 'none', borderRadius: 8, padding: '12px', color: '#fff', fontSize: 14, fontWeight: 500, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <LogIn size={16} />{loading ? 'Signing in...' : 'Sign in'}
            </button>
          </form>
          <p style={{ marginTop: 16, fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>Demo: engineer / cpchem2025</p>
        </div>
      </div>
    </div>
  )
}

// ── Metric Card ──────────────────────────────────────────────
function MetricCard({ label, value, sub, accent }: { label: string; value: string | number; sub?: string; accent?: string }) {
  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 12, padding: '18px 20px' }}>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>{label}</p>
      <p style={{ fontSize: 28, fontWeight: 600, color: accent || 'var(--text-primary)', letterSpacing: '-0.02em' }}>{value}</p>
      {sub && <p style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>{sub}</p>}
    </div>
  )
}

// ── Sensor Heatmap ───────────────────────────────────────────
function SensorHeatmap({ readings }: { readings: any[] }) {
  const equipment = ['CB-CGC-001', 'CB-CGC-002', 'CB-QWP-001', 'CB-FDF-001']
  const latest: Record<string, any> = {}
  readings.forEach(r => { latest[r.equipment_id] = r })

  const getColor = (eq: string) => {
    const r = latest[eq]
    if (!r) return 'var(--bg-card-2)'
    if (r.kurtosis > 8) return 'rgba(239,68,68,0.25)'
    if (r.kurtosis > 5) return 'rgba(245,158,11,0.25)'
    return 'rgba(34,197,94,0.15)'
  }
  const getDot = (eq: string) => {
    const r = latest[eq]
    if (!r) return '#5C5A80'
    if (r.kurtosis > 8) return '#EF4444'
    if (r.kurtosis > 5) return '#F59E0B'
    return '#22C55E'
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
      {equipment.map(eq => (
        <div key={eq} style={{ background: getColor(eq), border: `1px solid ${getDot(eq)}33`, borderRadius: 10, padding: '14px 16px', transition: 'all 0.3s' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 500, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{eq}</span>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: getDot(eq) }} />
          </div>
          {latest[eq] ? (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
              <div><p style={{ fontSize: 10, color: 'var(--text-muted)' }}>RMS</p><p style={{ fontSize: 13, fontWeight: 500 }}>{latest[eq].vibration_rms?.toFixed(2)}</p></div>
              <div><p style={{ fontSize: 10, color: 'var(--text-muted)' }}>Kurtosis</p><p style={{ fontSize: 13, fontWeight: 500 }}>{latest[eq].kurtosis?.toFixed(1)}</p></div>
            </div>
          ) : (
            <p style={{ fontSize: 11, color: 'var(--text-muted)' }}>No data yet</p>
          )}
        </div>
      ))}
    </div>
  )
}

// ── Alert Row ────────────────────────────────────────────────
function AlertRow({ alert }: { alert: Alert }) {
  return (
    <div style={{ padding: '14px 0', borderBottom: '1px solid var(--bg-border)', animation: 'fade-in 0.3s ease' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: '50%', background: priorityColor(alert.priority), flexShrink: 0,
            boxShadow: alert.priority === 'P1' ? `0 0 8px ${priorityColor(alert.priority)}` : 'none' }} />
          <span style={{ fontSize: 13, fontWeight: 500, fontFamily: 'var(--font-mono)' }}>{alert.equipment_id}</span>
          <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 10, background: severityBadge(alert.severity), color: priorityColor(alert.priority) }}>
            {alert.priority}
          </span>
        </div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmt(alert.saved_at || alert.timestamp)}</span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', paddingLeft: 16 }}>{alert.diagnosis?.slice(0, 100) || 'Anomaly detected'}{alert.diagnosis?.length > 100 ? '...' : ''}</p>
      <div style={{ paddingLeft: 16, marginTop: 4, display: 'flex', gap: 12 }}>
        {alert.kurtosis && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Kurt: <span style={{ color: 'var(--text-secondary)' }}>{Number(alert.kurtosis).toFixed(1)}</span></span>}
        {alert.vibration_rms && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>RMS: <span style={{ color: 'var(--text-secondary)' }}>{Number(alert.vibration_rms).toFixed(2)}</span></span>}
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>fault: <span style={{ color: 'var(--frame-teal)' }}>{alert.fault_type}</span></span>
      </div>
    </div>
  )
}

// ── Work Order Row ───────────────────────────────────────────
function WorkOrderRow({ wo, token, onRefresh }: { wo: WorkOrder; token: string; onRefresh: () => void }) {
  const [acking, setAcking] = useState(false)
  const ack = async () => {
    setAcking(true)
    try { await acknowledgeWorkOrder(token, wo.work_order_number); onRefresh() }
    catch (e) { console.error(e) }
    finally { setAcking(false) }
  }

  return (
    <div style={{ padding: '14px 0', borderBottom: '1px solid var(--bg-border)', animation: 'fade-in 0.3s ease' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--frame-teal)' }}>{wo.work_order_number}</span>
          <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 10, background: severityBadge(wo.priority === 'P1' ? 'critical' : wo.priority === 'P2' ? 'warning' : 'normal'), color: priorityColor(wo.priority) }}>{wo.priority}</span>
          {wo.status === 'acknowledged' && <span style={{ fontSize: 10, color: '#22C55E' }}>✓ acknowledged</span>}
        </div>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{fmt(wo.saved_at || '')}</span>
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6 }}>{wo.short_description}</p>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: 12 }}>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Failure: <span style={{ color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{wo.failure_code}</span></span>
          {wo.estimated_labor_hrs && <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>Labor: <span style={{ color: 'var(--text-secondary)' }}>{wo.estimated_labor_hrs}h</span></span>}
        </div>
        {wo.status !== 'acknowledged' && (
          <button onClick={ack} disabled={acking}
            style={{ fontSize: 11, padding: '4px 12px', borderRadius: 6, background: 'var(--frame-indigo)', border: 'none', color: '#fff', cursor: 'pointer' }}>
            {acking ? '...' : 'Acknowledge'}
          </button>
        )}
      </div>
    </div>
  )
}

// ── Simulate Panel ───────────────────────────────────────────
function SimulatePanel({ token, onSent }: { token: string; onSent: () => void }) {
  const [loading, setLoading] = useState<string | null>(null)
  const [msg, setMsg] = useState('')

  const send = async (type: keyof typeof SAMPLE_READINGS) => {
    setLoading(type); setMsg('')
    try {
      await sendSensorReading(token, SAMPLE_READINGS[type])
      setMsg(`Sent ${type} reading — agents running. Refresh in 10s.`)
    } catch (e) { setMsg('Error sending reading') }
    finally { setLoading(null) }
  }

  const buttons = [
    { key: 'normal', label: 'Normal', color: '#22C55E' },
    { key: 'inner_race', label: 'Inner Race Fault', color: '#EF4444' },
    { key: 'outer_race', label: 'Outer Race Fault', color: '#F59E0B' },
    { key: 'ball_fault', label: 'Ball Fault', color: '#3B82F6' },
  ] as const

  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 12, padding: 20 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <Zap size={16} color="var(--frame-teal)" />
        <span style={{ fontSize: 13, fontWeight: 500 }}>Simulate Sensor Reading</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
        {buttons.map(b => (
          <button key={b.key} onClick={() => send(b.key)} disabled={loading !== null}
            style={{ padding: '10px', borderRadius: 8, border: `1px solid ${b.color}33`, background: `${b.color}11`, color: b.color, fontSize: 12, fontWeight: 500, cursor: 'pointer', opacity: loading !== null ? 0.6 : 1 }}>
            {loading === b.key ? 'Sending...' : b.label}
          </button>
        ))}
      </div>
      {msg && <p style={{ fontSize: 11, color: 'var(--frame-teal)', background: 'var(--frame-teal-dim)', padding: '8px 12px', borderRadius: 6 }}>{msg}</p>}
    </div>
  )
}

// ── Main Dashboard ───────────────────────────────────────────
export default function Dashboard() {
  const [token, setToken] = useState<string | null>(null)
  const [username, setUsername] = useState('')
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [workOrders, setWorkOrders] = useState<WorkOrder[]>([])
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [readings, setReadings] = useState<any[]>([])
  const [activeTab, setActiveTab] = useState<'alerts' | 'workorders' | 'logs'>('alerts')
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const refresh = useCallback(async () => {
    if (!token) return
    try {
      const [a, w, l, r] = await Promise.all([
        apiFetch('/api/alerts?limit=20', token),
        apiFetch('/api/work-orders?limit=20', token),
        apiFetch('/api/agent-log?limit=30', token),
        apiFetch('/api/sensor-readings?limit=50', token),
      ])
      setAlerts(a.alerts || [])
      setWorkOrders(w.work_orders || [])
      setLogs(l.logs || [])
      setReadings(r.readings || [])
      setLastRefresh(new Date())
    } catch (e) { console.error('Refresh error:', e) }
  }, [token])

  useEffect(() => { if (token) refresh() }, [token, refresh])
  useEffect(() => {
    if (!token || !autoRefresh) return
    const interval = setInterval(refresh, 5000)
    return () => clearInterval(interval)
  }, [token, autoRefresh, refresh])

  if (!token) return <LoginScreen onLogin={(t, u) => { setToken(t); setUsername(u) }} />

  const p1Count = alerts.filter(a => a.priority === 'P1').length
  const p2Count = alerts.filter(a => a.priority === 'P2').length
  const openWOs = workOrders.filter(w => w.status !== 'acknowledged').length

  // chart data from readings
  const chartData = readings.slice(0, 20).reverse().map((r, i) => ({
    i, rms: Number(r.vibration_rms?.toFixed(2) || 0),
    kurt: Number(r.kurtosis?.toFixed(1) || 0),
  }))

  const tabs = [
    { key: 'alerts', label: 'Alerts', icon: <AlertTriangle size={14} />, count: alerts.length },
    { key: 'workorders', label: 'Work Orders', icon: <Wrench size={14} />, count: workOrders.length },
    { key: 'logs', label: 'Agent Log', icon: <FileText size={14} />, count: logs.length },
  ] as const

  return (
    <div style={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>

      {/* Nav */}
      <nav style={{ background: 'var(--frame-indigo-dark)', borderBottom: '1px solid var(--bg-border)', padding: '0 24px', height: 56, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ width: 28, height: 28, background: 'var(--frame-teal)', borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Activity size={16} color="#0F0E1A" />
          </div>
          <div>
            <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>CARIS</span>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>Cedar Bayou · CPChem</span>
          </div>
          {p1Count > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(239,68,68,0.15)', border: '1px solid #EF444433', borderRadius: 20, padding: '3px 10px' }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#EF4444', animation: 'pulse 1s infinite' }} />
              <span style={{ fontSize: 11, color: '#EF4444', fontWeight: 500 }}>{p1Count} P1 ACTIVE</span>
            </div>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {lastRefresh && <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Updated {fmt(lastRefresh.toISOString())}</span>}
          <button onClick={refresh} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 4, fontSize: 12 }}>
            <RefreshCw size={13} /> Refresh
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 28, height: 28, borderRadius: '50%', background: 'var(--frame-indigo)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 600, color: 'var(--frame-teal)' }}>
              {username[0]?.toUpperCase()}
            </div>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{username}</span>
          </div>
        </div>
      </nav>

      {/* Body */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '300px 1fr 340px', gap: 0, height: 'calc(100vh - 56px)', overflow: 'hidden' }}>

        {/* Left sidebar */}
        <div style={{ borderRight: '1px solid var(--bg-border)', padding: 20, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div>
            <p style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 12 }}>Equipment health</p>
            <SensorHeatmap readings={readings} />
          </div>
          <SimulatePanel token={token} onSent={refresh} />
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 12, padding: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <Shield size={14} color="var(--frame-teal)" />
              <span style={{ fontSize: 12, fontWeight: 500 }}>System status</span>
            </div>
            {[
              { label: 'Monitor Agent', ok: true },
              { label: 'RAG Retriever', ok: true },
              { label: 'Work Order Gen', ok: true },
              { label: 'API Backend', ok: true },
            ].map(item => (
              <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderBottom: '1px solid var(--bg-border)' }}>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{item.label}</span>
                <span style={{ fontSize: 11, color: item.ok ? '#22C55E' : '#EF4444' }}>{item.ok ? '● Online' : '● Offline'}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Center main */}
        <div style={{ padding: 24, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
            <MetricCard label="Total Alerts" value={alerts.length} sub="last 20 events" />
            <MetricCard label="P1 Critical" value={p1Count} accent={p1Count > 0 ? '#EF4444' : undefined} sub="immediate action" />
            <MetricCard label="P2 Warning" value={p2Count} accent={p2Count > 0 ? '#F59E0B' : undefined} sub="within 24 hours" />
            <MetricCard label="Open Work Orders" value={openWOs} accent={openWOs > 0 ? 'var(--frame-teal)' : undefined} sub="pending acknowledgment" />
          </div>

          {/* Chart */}
          {chartData.length > 0 && (
            <div style={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 12, padding: 20 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                <BarChart2 size={15} color="var(--frame-teal)" />
                <span style={{ fontSize: 13, fontWeight: 500 }}>Sensor trend — last {chartData.length} readings</span>
              </div>
              <ResponsiveContainer width="100%" height={160}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(45,43,107,0.3)" />
                  <XAxis dataKey="i" hide />
                  <YAxis yAxisId="rms" orientation="left" tick={{ fontSize: 10, fill: '#9896B8' }} />
                  <YAxis yAxisId="kurt" orientation="right" tick={{ fontSize: 10, fill: '#9896B8' }} />
                  <Tooltip contentStyle={{ background: 'var(--bg-card-2)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 12 }} />
                  <Line yAxisId="rms" type="monotone" dataKey="rms" stroke="#00C2CB" strokeWidth={2} dot={false} name="RMS (mm/s)" />
                  <Line yAxisId="kurt" type="monotone" dataKey="kurt" stroke="#F59E0B" strokeWidth={2} dot={false} name="Kurtosis" />
                </LineChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', gap: 20, marginTop: 8 }}>
                <span style={{ fontSize: 11, color: '#9896B8' }}><span style={{ color: '#00C2CB' }}>─</span> RMS (mm/s)</span>
                <span style={{ fontSize: 11, color: '#9896B8' }}><span style={{ color: '#F59E0B' }}>─</span> Kurtosis</span>
              </div>
            </div>
          )}

          {/* Tabs */}
          <div style={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 12, overflow: 'hidden', flex: 1 }}>
            <div style={{ display: 'flex', borderBottom: '1px solid var(--bg-border)' }}>
              {tabs.map(t => (
                <button key={t.key} onClick={() => setActiveTab(t.key as any)}
                  style={{ flex: 1, padding: '14px', background: 'none', border: 'none', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, fontSize: 12, fontWeight: 500, color: activeTab === t.key ? 'var(--frame-teal)' : 'var(--text-muted)', borderBottom: activeTab === t.key ? '2px solid var(--frame-teal)' : '2px solid transparent' }}>
                  {t.icon}{t.label}
                  <span style={{ fontSize: 10, background: 'var(--bg-base)', borderRadius: 10, padding: '1px 6px' }}>{t.count}</span>
                </button>
              ))}
            </div>
            <div style={{ padding: '0 20px', maxHeight: 400, overflowY: 'auto' }}>
              {activeTab === 'alerts' && (
                alerts.length === 0
                  ? <p style={{ color: 'var(--text-muted)', fontSize: 13, padding: '24px 0', textAlign: 'center' }}>No alerts. Send a sensor reading to trigger agents.</p>
                  : alerts.map((a, i) => <AlertRow key={i} alert={a} />)
              )}
              {activeTab === 'workorders' && (
                workOrders.length === 0
                  ? <p style={{ color: 'var(--text-muted)', fontSize: 13, padding: '24px 0', textAlign: 'center' }}>No work orders generated yet.</p>
                  : workOrders.map((w, i) => <WorkOrderRow key={i} wo={w} token={token} onRefresh={refresh} />)
              )}
              {activeTab === 'logs' && (
                logs.length === 0
                  ? <p style={{ color: 'var(--text-muted)', fontSize: 13, padding: '24px 0', textAlign: 'center' }}>No agent activity yet.</p>
                  : logs.map((l, i) => (
                    <div key={i} style={{ padding: '10px 0', borderBottom: '1px solid var(--bg-border)', animation: 'fade-in 0.3s ease' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--frame-teal)' }}>{l.agent}</span>
                          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{l.action}</span>
                        </div>
                        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{fmt(l.timestamp)}</span>
                      </div>
                      {l.diagnosis && <p style={{ fontSize: 11, color: 'var(--text-secondary)', paddingLeft: 0 }}>{l.diagnosis.slice(0, 80)}</p>}
                      {l.error && <p style={{ fontSize: 11, color: '#EF4444' }}>Error: {l.error}</p>}
                    </div>
                  ))
              )}
            </div>
          </div>
        </div>

        {/* Right sidebar — latest work order detail */}
        <div style={{ borderLeft: '1px solid var(--bg-border)', padding: 20, overflowY: 'auto' }}>
          <p style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)', marginBottom: 16 }}>Latest work order</p>
          {workOrders[0] ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ background: 'var(--bg-card-2)', borderRadius: 10, padding: 16, border: `1px solid ${priorityColor(workOrders[0].priority)}33` }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                  <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--frame-teal)' }}>{workOrders[0].work_order_number}</span>
                  <span style={{ fontSize: 10, padding: '2px 8px', borderRadius: 10, background: `${priorityColor(workOrders[0].priority)}22`, color: priorityColor(workOrders[0].priority), border: `1px solid ${priorityColor(workOrders[0].priority)}33` }}>
                    {workOrders[0].priority}
                  </span>
                </div>
                <p style={{ fontSize: 13, fontWeight: 500, marginBottom: 8 }}>{workOrders[0].short_description}</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {[
                    { label: 'Equipment', value: workOrders[0].equipment_id },
                    { label: 'Failure code', value: workOrders[0].failure_code },
                    { label: 'Labor estimate', value: workOrders[0].estimated_labor_hrs ? `${workOrders[0].estimated_labor_hrs} hours` : '—' },
                    { label: 'Status', value: workOrders[0].status || 'open' },
                  ].map(item => (
                    <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{item.label}</span>
                      <span style={{ fontSize: 11, color: 'var(--text-secondary)', fontFamily: item.label === 'Equipment' || item.label === 'Failure code' ? 'var(--font-mono)' : 'inherit' }}>{item.value}</span>
                    </div>
                  ))}
                </div>
              </div>
              {workOrders[0].parts_required && workOrders[0].parts_required.length > 0 && (
                <div style={{ background: 'var(--bg-card-2)', borderRadius: 10, padding: 16 }}>
                  <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>Parts required</p>
                  {(workOrders[0].parts_required || []).map((p, i) => (
                    <div key={i} style={{ fontSize: 12, color: 'var(--text-secondary)', padding: '4px 0', borderBottom: '1px solid var(--bg-border)' }}>
                      <ChevronRight size={10} style={{ marginRight: 6, color: 'var(--frame-teal)' }} />{p}
                    </div>
                  ))}
                </div>
              )}
              <div style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 10, padding: 14 }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                  <Shield size={13} color="#EF4444" />
                  <span style={{ fontSize: 11, fontWeight: 500, color: '#EF4444' }}>Safety requirements</span>
                </div>
                <p style={{ fontSize: 11, color: 'var(--text-secondary)' }}>LOTO required per SOP-CGC-004. Complete PTW before any maintenance.</p>
              </div>
            </div>
          ) : (
            <div style={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 10, padding: 24, textAlign: 'center' }}>
              <CheckCircle size={32} color="var(--text-muted)" style={{ margin: '0 auto 12px' }} />
              <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>No work orders yet</p>
              <p style={{ color: 'var(--text-muted)', fontSize: 11, marginTop: 6 }}>Send a fault reading to generate one</p>
            </div>
          )}
          <div style={{ marginTop: 20, padding: 16, background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 12 }}>
            <p style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', marginBottom: 10 }}>Powered by Frame Data AI</p>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              CARIS automates bearing fault detection, RAG-based diagnosis, and SAP work order generation for CPChem Cedar Bayou.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
