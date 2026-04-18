"""
One-time script to insert a sample DetectionEvent into the local SQLite DB.
Useful when you don't have the real pipeline running but need test data
(e.g. to test the feedback endpoints).

Run with:
    PYTHONPATH=src python scripts/seed_db.py
"""

import asyncio
from datetime import datetime, timezone
from ocean_sentinel.adapters.persistence import SQLiteEventStore
from ocean_sentinel.domain.models import DetectionEvent, GeoPoint
from ocean_sentinel.domain.enums import ThreatLevel

SAMPLE_EVENTS = [
    DetectionEvent(
        id="orcasound_orcasound_lab_2023-12-01_10800",
        timestamp=datetime(2026, 4, 16, 21, 17, 1, tzinfo=timezone.utc),
        location=GeoPoint(lat=47.9, lon=-122.7),  # Orcasound Lab, Salish Sea
        threat_level=ThreatLevel.NONE,
        confidence=0.98,
        classification_reasoning=(
            "The mel spectrogram shows unstructured, broadband noise without "
            "periodic or rhythmic patterns characteristic of mechanical propulsion. "
            "Acoustic features confirm low engine band energy (-16.61 dB) and the "
            "engine band is not dominant. There are no AIS gaps detected in the area. "
            "This signature is highly similar to prior observations (similarity 1.00) "
            "previously classified as ambient ocean noise or biological activity."
        ),
        ais_gaps=[],
        ocean_conditions=None,
        raw_model_output={
            "threat_level": "NONE",
            "confidence": 0.98,
            "vessel_type": "none",
            "recommended_action": "none",
            "features": {
                "engine_band_ratio": 0.675,
                "peak_frequency_hz": 47.3,
                "spectral_flatness": 0.1866,
                "rms_energy": 0.021479,
                "engine_band_energy_db": -16.61,
            },
        },
        audio_segment=None,
    ),
]


async def main():
    store = SQLiteEventStore()
    await store.init()

    for event in SAMPLE_EVENTS:
        await store.save_event(event)
        print(f"Inserted event: {event.id}  threat_level={event.threat_level.value}")

    await store.close()
    print("Done. Run: curl http://localhost:8000/events/ to verify.")


asyncio.run(main())
