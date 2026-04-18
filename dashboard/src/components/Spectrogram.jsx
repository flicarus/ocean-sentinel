import { useEffect, useRef } from 'react'
import { THREAT_COLORS } from '../data/events'
import './Spectrogram.css'

export default function Spectrogram({ event }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!event || !canvasRef.current) return
    draw(canvasRef.current, event)
  }, [event])

  return (
    <div className="spectrogram-section">
      <div className="spectrogram-container">
        <div className="spec-header">
          Spectrogram — <span style={{ color:'var(--text-muted)', fontWeight:400 }}>
            {event.time} — {event.vessel} ({event.peak}Hz peak)
          </span>
        </div>
        <canvas ref={canvasRef} className="spectrogram-canvas" />
      </div>
    </div>
  )
}

function draw(canvas, event) {
  const ctx = canvas.getContext('2d')
  const dpr = 2
  canvas.width = canvas.offsetWidth * dpr
  canvas.height = 400
  ctx.scale(dpr, dpr)

  const totalW = canvas.offsetWidth
  const totalH = 200
  const axisW = 55
  const panelW = 180
  const specW = totalW - axisW - panelW - 20
  const specH = totalH - 30
  const specX = axisW
  const specY = 5
  const maxFreq = 1000
  const cols = 300, rows = 100
  const colW = specW / cols, rowH = specH / rows

  ctx.fillStyle = '#0a0e1a'
  ctx.fillRect(0, 0, totalW, totalH)

  // Spectrogram
  for (let x = 0; x < cols; x++) {
    for (let y = 0; y < rows; y++) {
      const freq = (rows - y) / rows
      const time = x / cols
      let energy = Math.random() * 0.08 + (1 - freq) * 0.06
      const ef = event.peak / maxFreq
      const bw = 0.06
      energy += Math.exp(-(Math.pow((freq - ef) / bw, 2))) * (0.4 + 0.4 * Math.sin(time * 30 + Math.random() * 0.5)) * event.conf
      for (let h = 2; h <= 4; h++) {
        const hf = ef * h
        if (hf <= 1) energy += Math.exp(-(Math.pow((freq - hf) / (bw * 0.7), 2))) * (0.2 / h) * (0.4 + 0.3 * Math.sin(time * 30 + h)) * event.conf
      }
      energy = Math.min(1, Math.max(0, energy))
      const [r, g, b] = magma(energy)
      ctx.fillStyle = `rgb(${r},${g},${b})`
      ctx.fillRect(specX + x * colW, specY + y * rowH, colW + 0.5, rowH + 0.5)
    }
  }

  // Y axis
  ctx.textAlign = 'right'
  for (const f of [0,100,200,300,400,500,600,700,800,900,1000]) {
    const fy = specY + specH - (f / maxFreq) * specH
    ctx.fillStyle = '#475569'; ctx.fillRect(specX - 4, fy, 4, 1)
    ctx.fillStyle = '#94a3b8'; ctx.font = '9px JetBrains Mono'; ctx.fillText(f + ' Hz', specX - 8, fy + 3)
  }
  ctx.save(); ctx.translate(12, specY + specH / 2); ctx.rotate(-Math.PI / 2)
  ctx.fillStyle = '#64748b'; ctx.font = '10px Outfit'; ctx.textAlign = 'center'; ctx.fillText('Frequency', 0, 0)
  ctx.restore()

  // X axis
  ctx.textAlign = 'center'; ctx.font = '9px JetBrains Mono'
  for (let s = 0; s <= 60; s += 10) {
    const sx = specX + (s / 60) * specW
    ctx.fillStyle = '#475569'; ctx.fillRect(sx, specY + specH, 1, 4)
    ctx.fillStyle = '#94a3b8'; ctx.fillText(s + 's', sx, specY + specH + 14)
  }

  // Engine band
  const bandTop = specY + specH - (500 / maxFreq) * specH
  const bandBot = specY + specH - (50 / maxFreq) * specH
  ctx.strokeStyle = 'rgba(34,211,238,0.5)'; ctx.lineWidth = 1; ctx.setLineDash([6,4])
  ctx.strokeRect(specX, bandTop, specW, bandBot - bandTop); ctx.setLineDash([])
  ctx.fillStyle = 'rgba(34,211,238,0.7)'; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'left'
  ctx.fillText('ENGINE BAND', specX + 4, bandTop - 3)

  // Peak line
  const peakY = specY + specH - (event.peak / maxFreq) * specH
  ctx.strokeStyle = '#ef4444'; ctx.setLineDash([3,3])
  ctx.beginPath(); ctx.moveTo(specX, peakY); ctx.lineTo(specX + specW, peakY); ctx.stroke(); ctx.setLineDash([])
  ctx.fillStyle = '#ef4444'; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'right'
  ctx.fillText('PEAK ' + event.peak + 'Hz', specX + specW - 4, peakY - 4)

  // Harmonics
  ctx.fillStyle = 'rgba(249,115,22,0.6)'; ctx.font = '7px JetBrains Mono'
  for (let h = 2; h <= 4; h++) {
    const hf = event.peak * h
    if (hf <= maxFreq) {
      const hy = specY + specH - (hf / maxFreq) * specH
      ctx.setLineDash([2,4]); ctx.strokeStyle = 'rgba(249,115,22,0.3)'
      ctx.beginPath(); ctx.moveTo(specX, hy); ctx.lineTo(specX + specW, hy); ctx.stroke(); ctx.setLineDash([])
      ctx.textAlign = 'right'; ctx.fillText('H' + h + ' ' + hf + 'Hz', specX + specW - 4, hy - 3)
    }
  }

  // Color legend
  const legX = specX + specW + 10, legW = 12, legH = specH, legY = specY
  for (let ly = 0; ly < legH; ly++) {
    const [r, g, b] = magma(1 - ly / legH)
    ctx.fillStyle = `rgb(${r},${g},${b})`; ctx.fillRect(legX, legY + ly, legW, 1)
  }
  ctx.strokeStyle = '#1e293b'; ctx.strokeRect(legX, legY, legW, legH)
  ctx.fillStyle = '#94a3b8'; ctx.font = '8px JetBrains Mono'; ctx.textAlign = 'left'
  ctx.fillText('0 dB', legX + legW + 4, legY + 6)
  ctx.fillText('-30', legX + legW + 4, legY + legH * 0.33)
  ctx.fillText('-60', legX + legW + 4, legY + legH * 0.66)
  ctx.fillText('-90', legX + legW + 4, legY + legH - 2)

  // Info panel
  const px = legX + legW + 35, py = specY + 5
  const tc = THREAT_COLORS[event.threat] || '#475569'
  ctx.textAlign = 'left'; ctx.fillStyle = '#f1f5f9'; ctx.font = '600 11px Outfit'
  ctx.fillText('Analysis', px, py + 10)
  ctx.fillStyle = '#475569'; ctx.fillRect(px, py + 16, 80, 1)
  ctx.font = '9px JetBrains Mono'
  const lines = [
    ['Threat', event.threat, tc],
    ['Confidence', Math.round(event.conf * 100) + '%', '#f1f5f9'],
    ['Peak freq', event.peak + ' Hz', '#f1f5f9'],
    ['Energy', event.energy + ' dB', '#f1f5f9'],
    ['Vessel', event.vessel.length > 12 ? event.vessel.slice(0,12)+'..' : event.vessel, '#f1f5f9'],
    ['Flag', event.flag, '#f1f5f9'],
  ]
  lines.forEach(([label, val, col], i) => {
    const ry = py + 30 + i * 18
    ctx.fillStyle = '#64748b'; ctx.fillText(label, px, ry)
    ctx.fillStyle = col; ctx.fillText(val, px + 70, ry)
  })
  const barY = py + 30 + lines.length * 18 + 8
  ctx.fillStyle = '#1e293b'; ctx.fillRect(px, barY, 100, 4)
  ctx.fillStyle = tc; ctx.fillRect(px, barY, event.conf * 100, 4)
}

function magma(e) {
  if (e < 0.25) return [e*4*60, 0, e*4*80+10]
  if (e < 0.5) { const t=(e-0.25)*4; return [60+t*140, t*20, 80-t*30] }
  if (e < 0.75) { const t=(e-0.5)*4; return [200+t*55, 20+t*80, 50-t*30] }
  const t = (e-0.75)*4; return [255, 100+t*155, 20+t*100]
}
