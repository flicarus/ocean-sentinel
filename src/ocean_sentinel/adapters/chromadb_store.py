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


# ChromaDB collection name. Bumped from "spectrograms" (5-dim hand-crafted
# features) to "acoustic_memory_cnn64" when we switched the RAG key to the
# CNN's 64-dim shared backbone embedding. The two collections coexist on
# disk; we never read from the legacy one.
_COLLECTION_NAME = "acoustic_memory_cnn64"
_EMBEDDING_DIM = 64


class ChromaDBAcousticMemory:
    """AcousticMemory backed by ChromaDB, keyed on the CNN 64-dim embedding.

    Cosine distance — closer = more acoustically similar. Each entry carries
    enough metadata to rebuild the full AcousticEntry on retrieval.
    """

    def __init__(self, persist_dir: str = "data/chromadb") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "acoustic_memory_initialized",
            persist_dir=persist_dir,
            collection=_COLLECTION_NAME,
            entries=self._collection.count(),
        )

    async def store(self, entry: AcousticEntry) -> None:
        if entry.embedding is None:
            log.warning(
                "acoustic_entry_skipped_no_embedding",
                event_id=entry.event_id,
                reason=(
                    f"{_COLLECTION_NAME} requires {_EMBEDDING_DIM}-dim CNN "
                    "embeddings; entry has none"
                ),
            )
            return

        if len(entry.embedding) != _EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim mismatch: got {len(entry.embedding)}, "
                f"expected {_EMBEDDING_DIM}"
            )

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
            # Hand-crafted features kept as metadata for reconstruction +
            # debugging — they're cheap and let us inspect entries by feature.
            "engine_band_ratio": entry.features.engine_band_ratio,
            "peak_frequency_hz": entry.features.peak_frequency_hz,
            "spectral_flatness": entry.features.spectral_flatness,
            "rms_energy": entry.features.rms_energy,
            "engine_band_energy_db": entry.features.engine_band_energy_db,
        }

        self._collection.upsert(
            ids=[entry.event_id],
            embeddings=[list(entry.embedding)],
            documents=[entry.context_text],
            metadatas=[metadata],
        )

        log.info(
            "acoustic_entry_stored",
            event_id=entry.event_id,
            threat_level=entry.threat_level.value,
            total_entries=self._collection.count(),
        )

    async def query_by_embedding(
        self, embedding: list[float], n: int = 3,
    ) -> list[SimilarMatch]:
        """Cosine-similarity retrieval by CNN backbone embedding."""
        if len(embedding) != _EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dim mismatch: got {len(embedding)}, "
                f"expected {_EMBEDDING_DIM}"
            )
        return await self._query(list(embedding), n)

    async def query_similar(
        self, features: AcousticFeatures, n: int = 3,
    ) -> list[SimilarMatch]:
        """Legacy 5-dim path. Returns nothing useful in the new collection
        because all stored vectors are 64-dim. Kept for backward compat.
        """
        log.debug(
            "acoustic_query_similar_called",
            note=(
                "5-dim query against 64-dim collection — falling back, "
                "but caller should prefer query_by_embedding"
            ),
        )
        return []

    async def _query(
        self, vector: list[float], n: int,
    ) -> list[SimilarMatch]:
        total = self._collection.count()
        if total == 0:
            return []

        results = self._collection.query(
            query_embeddings=[vector],
            n_results=min(n, total),
        )

        # ChromaDB returns nested lists — one per query. We send one query.
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
                embedding=None,  # not round-tripped from chroma; only metadata
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
