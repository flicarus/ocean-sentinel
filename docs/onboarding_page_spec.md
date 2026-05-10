# `/onboarding` — Next.js page spec

For Maciej. Target stack: existing Next.js app at `~/oceansentinelfrontend`.
Goal: a polished landing page that ships our CLI demo to the world,
embedded asciinema, install instructions, GitHub release link.

## Why this page exists

The hackathon submission rules ask for a "public demo or demo files."
Our actual product is a CLI (`os onboard`). This page is the
market-facing onramp:

  1. A juror lands on it from the Kaggle write-up.
  2. They see the SOFAR AI brand, what Ocean Sentinel does, and a
     recording of someone running `os onboard` end-to-end.
  3. They can copy a 3-line install snippet to try it themselves.
  4. They can click through to the GitHub release.

The page needs to feel like a Vercel / Stripe / Linear landing —
restrained, dark theme aware, fast. Not a marketing brochure.

## Page sections (top to bottom)

### 1. Hero
- Brand mark: existing `≡SOFARAI` lockup (already in design system)
- Eyebrow: `Ocean Sentinel`
- Headline (h1): `Acoustic monitoring for marine protected areas — set up in 5 minutes.`
- Sub: `A pre-trained CNN, per-site conformal calibration, and a Gemma agent that walks you through setup. Runs on your laptop. No GPU, no ML team, no labelled data.`
- Two CTAs side-by-side:
  - **Primary**: `Install now ↓` (smooth-scrolls to install section)
  - **Secondary**: `View on GitHub` (link to release page; placeholder URL below)

### 2. Demo (the asciinema embed)
- Heading: `See it in action`
- One-line caption: `5-minute walkthrough — onboard a new MBARI hydrophone site.`
- Embed `docs/demo/onboard.cast` from this repo via the official
  asciinema-player web component.
- Sized at `cols=80, rows=28`, centered, max-width ~960px.
- Below the player, three small "highlight chips" with timestamps so a
  juror can jump to the moments that matter:
  - `0:00 · Site discovery`
  - `1:05 · Acoustic fingerprint`
  - `3:12 · Conformal calibration`
- A muted line under the chips: `Powered by Gemma 4 native function calling. 14 tools, 8 onboarding steps, all running locally via Ollama.`

### 3. Install in 60 seconds
Three monospace blocks side by side or stacked. Use a 3-step layout
with numbered headings so the order is obvious.

```bash
# 1. Install Ollama and pull the Gemma 4 model
brew install ollama && ollama pull gemma4:e4b
```

```bash
# 2. Install Ocean Sentinel
pipx install git+https://github.com/flicarus/ocean-sentinel
```

```bash
# 3. Onboard your first site
os onboard
```

Below the blocks, a single dim line: `Requires Python 3.11+, ~10 GB disk for the Gemma model. Tested on Apple Silicon and Linux.`

### 4. What runs locally
A small 3-column grid showing the architecture:

| | |
|---|---|
| **Gemma 4 (4 B params)** | Native function calling. Orchestrates the 14-tool onboarding flow. Runs on Ollama, on-device. |
| **CNN v7.4 (2.3 M params)** | Trained on 200k+ minutes of MBARI / Orcasound / SanctSound. 90 % raw accuracy on held-out sites, 97.2 % end-to-end. |
| **Per-site adaptation** | Split-conformal calibration on your site's ambient gives a provable false-alarm budget — no labels required. |

### 5. Built for transparency
A small section on credibility — for jurors who care about ML rigor:

- **Empirically validated** — every threshold has a reproducible derivation script
- **Audit log** — every detection writes to `data/sites/{id}.events.jsonl`
- **Open evaluation** — `data/eval/per_site_v7_4.json` is the source of every accuracy claim

Link out to `docs/empirical_findings.md` ("Read our limitations, not just our wins").

### 6. Footer
- GitHub release link (latest)
- Authors / contact
- License (MIT — to be confirmed)
- Built for **Gemma 4 Good Hackathon** badge

## Asciinema embed reference

```jsx
'use client'
import 'asciinema-player/dist/bundle/asciinema-player.css'
import { useEffect, useRef } from 'react'

export function OnboardCast() {
  const containerRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let player: any
    ;(async () => {
      const AsciinemaPlayer = await import('asciinema-player')
      if (!containerRef.current) return
      player = AsciinemaPlayer.create(
        '/onboard.cast',
        containerRef.current,
        { cols: 80, rows: 28, autoPlay: false, fit: 'width', theme: 'asciinema' },
      )
    })()
    return () => player?.dispose()
  }, [])
  return <div ref={containerRef} className="rounded-lg overflow-hidden" />
}
```

The cast file at `docs/demo/onboard.cast` (in this repo) needs to be
copied into the Next.js app's `public/` folder so it's served at
`/onboard.cast`.

## Visual guidance

- Color palette: same teal `#00D4C8` as the CLI banner (matches the
  brand wordmark already in design system).
- Typography: keep the existing site-wide pairing (likely Inter / SF
  Pro for prose, JetBrains Mono / IBM Plex Mono for code).
- Spacing: generous. This is a marketing page; whitespace > density.
- Dark mode: page should be dark by default, with light mode supported
  via existing theme provider.
- Code blocks: use the existing `<pre><code>` pattern, no syntax
  highlighting required for the install snippets (clarity > prettiness).

## Out of scope for this page

- Live `/api/events` feed of detections — that lives on the existing
  `/dashboard` page. Linked from this page, not embedded.
- Per-site config viewer — also dashboard.
- A "try it in your browser" widget — the CLI runs locally; we don't
  shell out to a remote sandbox.

## Effort estimate

- Hero + section structure: 1 h
- Asciinema integration + cast loading: 30 min (might need to add the
  `asciinema-player` dep)
- Install / runs-locally / transparency sections: 1.5 h
- Polish + responsive + dark-mode review: 1 h

Total: **~4 h** focused work.

## Handoff checklist

- [ ] Cast file `docs/demo/onboard.cast` copied to `public/onboard.cast`
- [ ] `pnpm add asciinema-player` (or compatible alternative)
- [ ] GitHub release URL plugged in (currently placeholder)
- [ ] Hero copy reviewed by Jakub before merge
- [ ] Mobile breakpoint tested (asciinema's `fit: width` should handle it)
