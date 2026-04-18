import { EVENTS } from '../data/events'
import './Ticker.css'

const TCLASS = { CRITICAL:'threat-high', HIGH:'threat-high', MEDIUM:'threat-medium', LOW:'threat-low' }

function TickerItem({ evt }) {
  return (
    <div className="ticker-item">
      <span className={`threat ${TCLASS[evt.threat] || ''}`}>{evt.threat}</span>
      <span>{evt.vessel}</span>
      <span className="ticker-dim">{evt.lat.toFixed(2)}°N {Math.abs(evt.lon).toFixed(2)}°W</span>
      <span className="ticker-dim">conf {Math.round(evt.conf * 100)}%</span>
    </div>
  )
}

export default function Ticker() {
  const items = [...EVENTS, ...EVENTS, ...EVENTS, ...EVENTS]

  return (
    <div className="ticker">
      <div className="ticker-label">
        <span className="ticker-dot" />
        LIVE SIGNAL
      </div>
      <div className="ticker-track">
        <div className="ticker-scroll">
          {items.map((evt, i) => <TickerItem key={i} evt={evt} />)}
        </div>
      </div>
    </div>
  )
}
