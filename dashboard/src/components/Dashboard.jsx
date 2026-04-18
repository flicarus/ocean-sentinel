import { useState, useCallback } from 'react'
import { MapContainer, TileLayer, Marker, Circle, Tooltip, useMap } from 'react-leaflet'
import L from 'leaflet'
import { EVENTS as MOCK_EVENTS, THREAT_COLORS } from '../data/events'
import Spectrogram from './Spectrogram'
import './Dashboard.css'

const API = 'http://localhost:8000'

// Hydrophone locations we listen to. MBARI (Monterey Canyon) + 8 Orcasound
// nodes across the Salish Sea / Puget Sound. Each circle on the map is one.
const HYDROPHONES = [
  { id: 'mbari',              name: 'MBARI — Monterey Canyon (CA)', lat: 36.7128,   lon: -122.186 },
  { id: 'orcasound_lab',      name: 'Orcasound Lab — Haro Strait',  lat: 48.5583362, lon: -123.1735774 },
  { id: 'port_townsend',      name: 'Port Townsend — Admiralty Inlet', lat: 48.135743, lon: -122.760614 },
  { id: 'bush_point',         name: 'Bush Point — Whidbey Island',  lat: 48.0336664, lon: -122.6040035 },
  { id: 'sunset_bay',         name: 'Sunset Bay — Edmonds',         lat: 47.864973, lon: -122.333936 },
  { id: 'mast_center',        name: 'MaST Center — Puget Sound',    lat: 47.34922,   lon: -122.32512 },
  { id: 'point_robinson',     name: 'Point Robinson — Vashon-Maury',lat: 47.388383, lon: -122.37267 },
  { id: 'andrews_bay',        name: 'Andrews Bay — San Juan Island',lat: 48.546653, lon: -123.166408 },
  { id: 'north_sjc',          name: 'North San Juan Channel',       lat: 48.591294, lon: -123.058779 },
]

function FlyTo({ center }) {
  const map = useMap()
  if (center) map.flyTo(center, 13, { duration: 1 })
  return null
}

function EventMarker({ evt }) {
  const color = THREAT_COLORS[evt.threat] || '#475569'
  const size = evt.threat === 'CRITICAL' ? 16 : evt.threat === 'HIGH' ? 13 : 10
  const icon = L.divIcon({
    className: '',
    html: `<div style="width:${size}px;height:${size}px;background:${color};border-radius:50%;border:2px solid rgba(255,255,255,0.3);box-shadow:0 0 ${size}px ${color}40,0 0 ${size*3}px ${color}15;"></div>`,
    iconSize: [size, size],
    iconAnchor: [size/2, size/2],
  })
  return <Marker position={[evt.lat, evt.lon]} icon={icon} />
}

const THREAT_CLASS = { CRITICAL:'threat-high', HIGH:'threat-high', MEDIUM:'threat-medium', LOW:'threat-low' }

export default function Dashboard({ selectedEvent, onSelectEvent }) {
  const [events, setEvents] = useState(MOCK_EVENTS)
  const [scanning, setScanning] = useState(false)
  const [scanDate, setScanDate] = useState('2024-02-01')
  const [scanStep, setScanStep] = useState(300)
  const [scanSource, setScanSource] = useState('mbari')
  const [progress, setProgress] = useState('')
  const [scanResult, setScanResult] = useState(null)

  const runScan = useCallback(async () => {
    setScanning(true)
    setProgress('Connecting to pipeline...')
    setScanResult(null)

    try {
      const resp = await fetch(`${API}/pipeline/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source: scanSource, date: scanDate, step: scanStep, sample_every: 5 }),
      })

      if (!resp.ok) throw new Error(`API error: ${resp.status}`)

      const data = await resp.json()
      setScanResult(data)
      setProgress(`Done — ${data.chunks_scanned} scanned, ${data.chunks_classified} classified, ${data.classifications.length} detections`)

      // Convert classifications to dashboard events
      if (data.classifications.length > 0) {
        const newEvents = data.classifications.map((c, i) => ({
          id: `scan-${i}`,
          time: c.time,
          ts: parseInt(c.time.split(':')[0]) + parseInt(c.time.split(':')[1]) / 60,
          lat: (c.lat ?? 36.7128) + (Math.random() - 0.5) * 0.02,
          lon: (c.lon ?? -122.186) + (Math.random() - 0.5) * 0.02,
          threat: c.threat_level,
          conf: c.confidence,
          vessel: c.vessel_type || 'Unknown vessel',
          flag: 'Unknown',
          reasoning: c.reasoning,
          energy: -35,
          peak: 125,
        }))
        setEvents(newEvents)
      }
    } catch (err) {
      setProgress(`Error: ${err.message}. Using mock data.`)
    } finally {
      setScanning(false)
    }
  }, [scanDate, scanStep, scanSource])

  const flyCenter = selectedEvent ? [selectedEvent.lat, selectedEvent.lon] : null
  const sorted = [...events].sort((a, b) => b.conf - a.conf)

  return (
    <section className="dashboard" id="dashboard">
      <div className="section-header">
        <h2>Detection Map</h2>
        <span className="section-tag">Real-time</span>
      </div>

      {/* Scan controls */}
      <div className="scan-controls">
        <div className="scan-row">
          <label className="scan-label">
            Source
            <select value={scanSource} onChange={e => setScanSource(e.target.value)} className="scan-input">
              <option value="mbari">MBARI — Monterey (CA)</option>
              <option value="orcasound_lab">Orcasound Lab — Haro Strait</option>
              <option value="port_townsend">Port Townsend — Admiralty Inlet</option>
              <option value="bush_point">Bush Point — Whidbey Island</option>
              <option value="sunset_bay">Sunset Bay — Edmonds</option>
            </select>
          </label>
          <label className="scan-label">
            Date
            <input type="date" value={scanDate} onChange={e => setScanDate(e.target.value)} className="scan-input" />
          </label>
          <label className="scan-label">
            Step (seconds)
            <select value={scanStep} onChange={e => setScanStep(Number(e.target.value))} className="scan-input">
              <option value={60}>60s — full scan (~25min)</option>
              <option value={300}>300s — quick scan (~5min)</option>
              <option value={900}>900s — fast scan (~2min)</option>
            </select>
          </label>
          <button className="scan-btn" onClick={runScan} disabled={scanning}>
            {scanning ? 'Scanning...' : 'Run Pipeline Scan'}
          </button>
        </div>
        {progress && <div className="scan-progress">{progress}</div>}
        {scanResult && (
          <div className="scan-stats">
            <span>Scanned: {scanResult.chunks_scanned}</span>
            <span>Classified: {scanResult.chunks_classified}</span>
            <span>Failed: {scanResult.chunks_failed}</span>
            <span>Detections: {scanResult.classifications.length}</span>
          </div>
        )}
      </div>

      <div className="dash-grid">
        <div className="map-container">
          <MapContainer center={[43, -122.5]} zoom={5} zoomControl={true} attributionControl={false} style={{ height:'100%', width:'100%' }}>
            <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" maxZoom={19} />
            {HYDROPHONES.map(h => (
              <Circle
                key={h.id}
                center={[h.lat, h.lon]}
                radius={8000}
                pathOptions={{ color:'rgba(34,211,238,0.35)', fillColor:'rgba(34,211,238,0.06)', fillOpacity:1, weight:1, dashArray:'8,6' }}
              >
                <Tooltip>{h.name}</Tooltip>
              </Circle>
            ))}
            {events.map(evt => <EventMarker key={evt.id} evt={evt} />)}
            <FlyTo center={flyCenter} />
          </MapContainer>
        </div>

        <div className="feed">
          <div className="feed-header">
            Detection Feed
            <span className="feed-count">{events.length} events</span>
          </div>
          <div className="feed-list">
            {sorted.map(evt => (
              <div key={evt.id} className={`event-card ${selectedEvent?.id === evt.id ? 'active' : ''}`} onClick={() => onSelectEvent(evt)}>
                <div className="event-card-top">
                  <span className={`threat ${THREAT_CLASS[evt.threat] || ''}`}>{evt.threat}</span>
                  <span className="event-time">{evt.time}</span>
                </div>
                <div className="event-title">{evt.vessel}</div>
                <div className="event-detail">{evt.reasoning.slice(0,100)}...</div>
                <div className="confidence-bar">
                  <div className="confidence-fill" style={{ width: `${evt.conf*100}%`, background: THREAT_COLORS[evt.threat] || '#475569' }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div className="timeline-section">
        <div className="timeline-container">
          <div className="timeline-header">24-Hour Activity — <span style={{ color:'var(--text-muted)', fontWeight:400 }}>{scanDate}</span></div>
          <div className="timeline">
            {Array.from({ length: 24 }, (_, h) => {
              const evt = events.find(e => Math.floor(e.ts) === h)
              const bg = evt ? (THREAT_COLORS[evt.threat] || '#475569') : `rgba(34,211,238,${0.1 + Math.random()*0.15})`
              return <div key={h} className="timeline-block" style={{ background: bg, opacity: evt ? 0.8 : 1 }} title={evt ? `${evt.time} — ${evt.threat}` : `${String(h).padStart(2,'0')}:00`} />
            })}
          </div>
          <div className="timeline-labels">
            <span>00:00</span><span>04:00</span><span>08:00</span><span>12:00</span><span>16:00</span><span>20:00</span><span>24:00</span>
          </div>
        </div>
      </div>

      <Spectrogram event={selectedEvent || events[0]} />
    </section>
  )
}
