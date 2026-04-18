export const EVENTS = [
  { id:"evt-001", time:"02:14 UTC", ts:2.23, lat:36.72, lon:-122.19, threat:"HIGH", conf:0.87, vessel:"Unknown trawler", flag:"Unknown", reasoning:"Low-frequency tonal consistent with diesel engine at 120Hz fundamental. AIS gap detected for vessel in MPA zone.", energy:-32.4, peak:125 },
  { id:"evt-002", time:"05:41 UTC", ts:5.68, lat:36.68, lon:-122.22, threat:"MEDIUM", conf:0.64, vessel:"Fishing vessel", flag:"Panama", reasoning:"Intermittent engine signature in 80-200Hz band. No AIS gap but vessel operating outside declared fishing zone.", energy:-38.1, peak:95 },
  { id:"evt-003", time:"08:22 UTC", ts:8.37, lat:36.75, lon:-122.15, threat:"LOW", conf:0.42, vessel:"Cargo ship", flag:"Liberia", reasoning:"Broadband noise consistent with large vessel transit. AIS active, within shipping lane. Low concern.", energy:-41.5, peak:200 },
  { id:"evt-004", time:"14:07 UTC", ts:14.12, lat:36.71, lon:-122.25, threat:"CRITICAL", conf:0.93, vessel:"Dark vessel", flag:"Unknown", reasoning:"Strong trawler engine signature with net winch harmonics at 60Hz. AIS disabled 18+ hours. Inside Monterey Bay NMS.", energy:-28.7, peak:62 },
  { id:"evt-005", time:"19:33 UTC", ts:19.55, lat:36.69, lon:-122.18, threat:"HIGH", conf:0.78, vessel:"Unknown vessel", flag:"China", reasoning:"Dual-engine signature typical of large fishing vessel. AIS gap coincides with entry into marine sanctuary.", energy:-34.2, peak:110 },
];

export const THREAT_COLORS = {
  CRITICAL: '#ef4444',
  HIGH: '#f97316',
  MEDIUM: '#f59e0b',
  LOW: '#10b981',
  NONE: '#475569',
};
