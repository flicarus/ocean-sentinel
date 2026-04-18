import { Link } from 'react-router-dom'
import './Navbar.css'

export default function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/" className="nav-brand">
        Sofar<span>.AI</span>
      </Link>
      <ul className="nav-links">
        <li><Link to="/problem">Problem</Link></li>
        <li><Link to="/dashboard">Dashboard</Link></li>
        <li><Link to="/about">About</Link></li>
        <li><a href="https://github.com/flicarus/ocean-sentinel" target="_blank" rel="noreferrer">GitHub</a></li>
      </ul>
    </nav>
  )
}
