from __future__ import annotations

from datetime import datetime

import chromadb
import structlog

from ocean_sentinel.domain.enums import ThreatLevel
from ocean_sentinel.domain.models import (
    AcousticEntry,
    AcousticFeatures,
    GeoPoint,
    SimilarMatch,
)

log = structlog.get_logger()


class ChromaDBAcousticMemory:
    """AcousticMemory implementation backed by ChromaDB.

    Stores acoustic feature embeddings alongside classification metadata,
    enabling RAG retrieval of similar past detections to enrich Gemma's
    context on future scans.
    """

    def __init__(self, persist_dir: str = "data/chromadb") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="spectrograms",
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "acoustic_memory_initialized",
            persist_dir=persist_dir,
            entries=self._collection.count(),
        )

    async def store(self, entry: AcousticEntry) -> None:
        vector = entry.features.to_vector()

        metadata = {
            # Classification verdict
            "threat_level": entry.threat_level.value,
            "confidence": entry.confidence,
            "reasoning": entry.reasoning,
            "vessel_type": entry.vessel_type or "",
            "recommended_action": entry.recommended_action or "",
            # Spatial / temporal
            "timestamp": entry.timestamp.isoformat(),
            "lat": entry.location.lat,
            "lon": entry.location.lon,
            # Raw features — needed to reconstruct AcousticFeatures on retrieval
            "engine_band_ratio": entry.features.engine_band_ratio,
            "peak_frequency_hz": entry.features.peak_frequency_hz,
            "spectral_flatness": entry.features.spectral_flatness,
            "rms_energy": entry.features.rms_energy,
            "engine_band_energy_db": entry.features.engine_band_energy_db,
        }

        self._collection.upsert(
            ids=[entry.event_id],
            embeddings=[vector],
            documents=[entry.context_text],
            metadatas=[metadata],
        )

        log.info(
            "acoustic_entry_stored",
            event_id=entry.event_id,
            threat_level=entry.threat_level.value,
            total_entries=self._collection.count(),
        )

    async def query_similar(
        self, features: AcousticFeatures, n: int = 3,
    ) -> list[SimilarMatch]:
        total = self._collection.count()
        if total == 0:
            return []

        vector = features.to_vector()

        results = self._collection.query(
            query_embeddings=[vector],
            n_results=min(n, total),
        )

        # ChromaDB returns nested lists — one per query. We only send one query.
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        matches: list[SimilarMatch] = []
        for event_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
            entry = AcousticEntry(
                event_id=event_id,
                timestamp=datetime.fromisoformat(meta["timestamp"]),
                location=GeoPoint(lat=meta["lat"], lon=meta["lon"]),
                features=AcousticFeatures(
                    engine_band_ratio=meta["engine_band_ratio"],
                    peak_frequency_hz=meta["peak_frequency_hz"],
                    spectral_flatness=meta["spectral_flatness"],
                    rms_energy=meta["rms_energy"],
                    engine_band_energy_db=meta["engine_band_energy_db"],
                ),
                context_text=doc,
                threat_level=ThreatLevel(meta["threat_level"]),
                confidence=meta["confidence"],
                reasoning=meta["reasoning"],
                vessel_type=meta["vessel_type"] or None,
                recommended_action=meta["recommended_action"] or None,
            )
            matches.append(SimilarMatch(entry=entry, score=distance))

        log.info(
            "acoustic_query_complete",
            n_requested=n,
            n_returned=len(matches),
            closest_score=matches[0].score if matches else None,
        )

        return matches

    async def count(self) -> int:
        return self._collection.count()

    async def close(self) -> None:
        # PersistentClient auto-flushes to disk — nothing to release.
        log.info("acoustic_memory_closed", total_entries=self._collection.count())
