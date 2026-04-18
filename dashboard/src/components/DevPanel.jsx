import { useState, useEffect, useRef, useCallback } from 'react'
import './DevPanel.css'

const API = 'http://localhost:8000'

const LEVEL_COLORS = {
  info: '#22d3ee',
  warning: '#f59e0b',
  error: '#ef4444',
  debug: '#64748b',
}

// Events worth highlighting in the log stream
const HIGHLIGHT_EVENTS = [
  'rag_context_injected',
  'acoustic_entry_stored',
  'acoustic_query_complete',
  'gemma_response',
  'classification_stored_to_memory',
  'training_example_logged',
]

export default function DevPanel() {
  const [open, setOpen] = useState(false)
  const [logs, setLogs] = useState([])
  const [filter, setFilter] = useState('')
  const [connected, setConnected] = useState(false)
  const [autoScroll, setAutoScroll] = useState(true)
  const bottomRef = useRef(null)
  const eventSourceRef = useRef(null)

  const connect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    const es = new EventSource(`${API}/logs/stream`)
    eventSourceRef.current = es

    es.onopen = () => setConnected(true)

    es.onmessage = (msg) => {
      try {
        const entry = JSON.parse(msg.data)
        setLogs(prev => {
          const next = [...prev, entry]
          // Keep last 500 entries to avoid memory issues
          return next.length > 500 ? next.slice(-500) : next
        })
      } catch {
        // ignore malformed messages
      }
    }

    es.onerror = () => {
      setConnected(false)
      es.close()
      // Reconnect after 3 seconds
      setTimeout(connect, 3000)
    }
  }, [])

  useEffect(() => {
    if (open && !eventSourceRef.current) {
      connect()
    }
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
    }
  }, [open, connect])

  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  const filtered = filter
    ? logs.filter(l =>
        l.event.toLowerCase().includes(filter.toLowerCase()) ||
        JSON.stringify(l.details).toLowerCase().includes(filter.toLowerCase())
      )
    : logs

  const formatDetails = (details) => {
    const entries = Object.entries(details).filter(([k]) => k !== '_record')
    if (entries.length === 0) return ''
    return entries.map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ')
  }

  return (
    <>
      <button
        className={`dev-toggle ${connected ? 'connected' : ''}`}
        onClick={() => setOpen(!open)}
      >
        {open ? '▼' : '▲'} Dev
        {connected && <span className="dev-dot" />}
        {logs.length > 0 && <span className="dev-count">{logs.length}</span>}
      </button>

      {open && (
        <div className="dev-panel">
          <div className="dev-toolbar">
            <input
              className="dev-filter"
              type="text"
              placeholder="Filter logs..."
              value={filter}
              onChange={e => setFilter(e.target.value)}
            />
            <label className="dev-auto">
              <input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} />
              Auto-scroll
            </label>
            <span className="dev-status">
              {connected ? '● Connected' : '○ Disconnected'}
            </span>
            <span className="dev-entries">{filtered.length} entries</span>
            <button className="dev-clear" onClick={() => setLogs([])}>Clear</button>
          </div>

          <div className="dev-logs">
            {filtered.map((entry, i) => {
              const isHighlight = HIGHLIGHT_EVENTS.includes(entry.event)
              return (
                <div
                  key={i}
                  className={`dev-line ${isHighlight ? 'highlight' : ''}`}
                >
                  <span className="dev-time">
                    {entry.timestamp?.split('T')[1]?.slice(0, 12) || ''}
                  </span>
                  <span
                    className="dev-level"
                    style={{ color: LEVEL_COLORS[entry.level] || '#94a3b8' }}
                  >
                    {entry.level?.toUpperCase().padEnd(5)}
                  </span>
                  <span className="dev-event">{entry.event}</span>
                  <span className="dev-details">{formatDetails(entry.details)}</span>
                </div>
              )
            })}
            <div ref={bottomRef} />
          </div>
        </div>
      )}
    </>
  )
}
