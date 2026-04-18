import { motion } from 'framer-motion'
import './About.css'

export default function About() {
  return (
    <section className="about" id="about">
      <h2>About <em>Sofar.AI</em></h2>
      <p className="about-subtitle">
        We build intelligence for the part of Earth everyone else ignores. Our system
        listens to the ocean 24/7, correlating underwater acoustics with satellite
        tracking and environmental data to detect illegal fishing in real time.
      </p>

      <div className="about-grid">
        <Card img="/hydrophone.jpg" title="Hydrophone Audio" text="We tap into underwater microphones on the ocean floor. Ship engines have a unique acoustic fingerprint at 50-500Hz — we generate spectrograms and extract features in real time." />
        <Card img="/vessel.jpg" title="Vessel Tracking" text="We monitor AIS gaps through Global Fishing Watch — when a vessel turns off its transponder near a marine sanctuary, that's a red flag we correlate with our acoustic data." />
        <Card img="/gemma.png" title="Gemma 4 Fusion" text="Google's multimodal AI analyzes spectrograms alongside vessel and ocean data. One model, three data streams, one verdict — with confidence scores and full reasoning." />
      </div>

      <div className="about-stats">
        <Stat value="2TB" label="Audio processed / month" />
        <Stat value="3" label="Real-time data streams" />
        <Stat value="<1s" label="Alert latency" />
        <Stat value="24/7" label="Continuous monitoring" />
      </div>

      <div className="about-team">
        <h3>Team</h3>
        <div className="team-grid">
          <div className="team-member">
            <div className="team-name">Jakub Koscielny</div>
            <div className="team-role">Audio pipeline, ML, Gemma integration</div>
          </div>
          <div className="team-member">
            <div className="team-name">Maciej Rychlewski</div>
            <div className="team-role">Data APIs, backend, alerts</div>
          </div>
        </div>
      </div>
    </section>
  )
}

function Card({ img, title, text }) {
  return (
    <motion.div
      className="about-card-img"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      style={{ backgroundImage: `url(${img})` }}
    >
      <div className="about-card-overlay">
        <h3>{title}</h3>
        <p>{text}</p>
      </div>
    </motion.div>
  )
}

function Stat({ value, label }) {
  return (
    <div>
      <div className="about-stat-value">{value}</div>
      <div className="about-stat-label">{label}</div>
    </div>
  )
}
