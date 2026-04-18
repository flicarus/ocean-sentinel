import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './Problem.css'

const TABS = [
  {
    label: 'Illegal Fishing',
    stat: '$23B',
    title: 'Illegal fishing per year',
    img: '/problem-img.jpg',
    text: 'IUU fishing accounts for up to 26 million tonnes of catch annually. It devastates fish populations, destroys marine ecosystems, and threatens food security for hundreds of millions of people in coastal communities.',
    points: [
      'Coastal communities lose their primary food and income source',
      'Legal fishermen can\'t compete with industrial poachers',
      'Marine protected areas exist on paper with zero enforcement',
    ],
  },
  {
    label: 'Blind Spots',
    stat: '70%',
    title: 'Of the ocean is unmonitored',
    img: '/problem-blindspots.jpg',
    text: 'The vast majority of the ocean has zero real-time surveillance. Existing systems rely on satellite AIS — which vessels simply turn off. What you can\'t see, you can\'t protect.',
    points: [
      'AIS tracking is voluntary and trivially disabled',
      'Patrol boats can\'t cover millions of square kilometers',
      'Satellite imagery is expensive and low-frequency',
    ],
  },
  {
    label: 'Data Overload',
    stat: '6x',
    title: 'More data than analysts can process',
    img: '/problem-dataoverload.jpg',
    text: 'Ocean sensor data is growing 6x faster than the number of people who can analyze it. Hydrophones, satellites, and buoys generate terabytes — most of it is never looked at.',
    points: [
      'Hydrophone archives contain years of unanalyzed audio',
      'Manual review catches less than 1% of incidents',
      'By the time data is reviewed, poachers are long gone',
    ],
  },
]

export default function Problem() {
  const [active, setActive] = useState(0)
  const tab = TABS[active]

  return (
    <section className="problem" id="problem">
      <div className="problem-inner">
        <div className="problem-badge">The problem</div>
        <h2 className="problem-title">
          Billions lost, oceans <em>unprotected.</em>
        </h2>

        <div className="problem-tabs">
          {TABS.map((t, i) => (
            <button
              key={i}
              className={`problem-tab ${i === active ? 'active' : ''}`}
              onClick={() => setActive(i)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="problem-content">
          <AnimatePresence mode="wait">
            <motion.div
              key={active}
              className="problem-text"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
            >
              <div className="problem-stat">{tab.stat}</div>
              <h3>{tab.title}</h3>
              <p>{tab.text}</p>
              <ul className="problem-list">
                {tab.points.map((pt, i) => (
                  <li key={i}>{pt}</li>
                ))}
              </ul>
            </motion.div>
          </AnimatePresence>

          <AnimatePresence mode="wait">
            <motion.div
              key={active}
              className="problem-image"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              <img src={tab.img} alt={tab.label} />
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </section>
  )
}
