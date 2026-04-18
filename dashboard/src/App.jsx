import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Problem from './components/Problem'
import Dashboard from './components/Dashboard'
import About from './components/About'
import Ticker from './components/Ticker'
import DevPanel from './components/DevPanel'

export default function App() {
  const [selectedEvent, setSelectedEvent] = useState(null)

  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Hero />} />
        <Route path="/problem" element={<Problem />} />
        <Route path="/dashboard" element={<Dashboard selectedEvent={selectedEvent} onSelectEvent={setSelectedEvent} />} />
        <Route path="/about" element={<About />} />
      </Routes>
      <Ticker />
      <DevPanel />
    </BrowserRouter>
  )
}
