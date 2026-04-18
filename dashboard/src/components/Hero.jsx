import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import './Hero.css'

export default function Hero() {
  return (
    <section className="hero">
      <video
        className="hero-video"
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
        poster="/hydrophone.jpg"
        onError={(e) => console.error('hero video failed:', e)}
        ref={el => { if (el) el.playbackRate = 0.85 }}
      >
        <source src="/20349820-uhd_3840_2160_60fps.mp4" type="video/mp4" />
        <source src="/ocean-bg.mp4" type="video/mp4" />
      </video>

      <div className="hero-content">
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          The ocean finally has<br /><em>a voice.</em>
        </motion.h1>

        <motion.p
          className="hero-sub"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          Ocean Sentinel fuses hydrophone audio, vessel tracking, and
          oceanographic data — then lets Gemma&nbsp;4 find what human analysts miss.
        </motion.p>

        <motion.div
          className="cta-row"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
        >
          <Link to="/dashboard" className="btn btn-primary">See it in action</Link>
          <a href="https://github.com/flicarus/ocean-sentinel" className="btn btn-secondary" target="_blank" rel="noreferrer">View on GitHub</a>
        </motion.div>
      </div>
    </section>
  )
}
