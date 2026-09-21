from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .settings import RuntimeSettings


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def tokenize(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return TOKEN_PATTERN.findall(normalized)


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    rank: int
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    dense_similarity: float | None
    chunk: dict[str, Any]


class HybridRetriever:
    """In-memory BM25 plus optional dense retrieval, fused with weighted RRF."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.chunks = list(iter_jsonl(settings.path("chunks")))
        if not self.chunks:
            raise RuntimeError("No chunks were loaded.")
        self.chunk_by_id = {str(item["id"]): item for item in self.chunks}
        self._tokens = [tokenize(str(item.get("embedding_text") or item["content"])) for item in self.chunks]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = np.asarray([len(tokens) for tokens in self._tokens], dtype=np.float32)
        self._average_length = float(self._lengths.mean())
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))
        total = len(self.chunks)
        self._idf = {
            term: max(0.0, math.log((total - frequency + 0.5) / (frequency + 0.5)))
            for term, frequency in document_frequency.items()
        }

        index = json.loads(settings.path("dense_index").read_text(encoding="utf-8"))
        indexed_ids = [str(item["chunk_id"]) for item in index["chunks"]]
        chunk_ids = [str(item["id"]) for item in self.chunks]
        if indexed_ids != chunk_ids:
            raise RuntimeError("Dense vector index is not aligned with the retained chunk file.")
        self.vectors = np.load(settings.path("dense_vectors"), mmap_mode="r")
        if self.vectors.shape[0] != len(self.chunks):
            raise RuntimeError("Dense vectors and chunks have different row counts.")

    def _domain_mask(self, domain: str | None) -> np.ndarray:
        if not domain:
            return np.ones(len(self.chunks), dtype=bool)
        return np.asarray([item.get("domain") == domain for item in self.chunks], dtype=bool)

    def bm25_scores(self, query: str, domain: str | None = None) -> np.ndarray:
        config = self.settings.retrieval
        k1 = float(config["bm25_k1"])
        b = float(config["bm25_b"])
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for term in dict.fromkeys(tokenize(query)):
            idf = self._idf.get(term, 0.0)
            if idf <= 0:
                continue
            frequencies = np.asarray(
                [counter.get(term, 0) for counter in self._term_frequencies], dtype=np.float32
            )
            denominator = frequencies + k1 * (
                1.0 - b + b * self._lengths / self._average_length
            )
            scores += idf * ((frequencies * (k1 + 1.0)) / np.maximum(denominator, 1e-9))
        scores[~self._domain_mask(domain)] = -np.inf
        return scores

    @staticmethod
    def _top_indices(scores: np.ndarray, depth: int) -> list[int]:
        valid = np.flatnonzero(np.isfinite(scores))
        if not len(valid):
            return []
        depth = min(depth, len(valid))
        selected = valid[np.argpartition(scores[valid], -depth)[-depth:]]
        return [int(index) for index in selected[np.argsort(scores[selected])[::-1]]]

    def search(
        self,
        query: str,
        *,
        domain: str | None = None,
        query_vector: np.ndarray | None = None,
    ) -> tuple[list[RetrievalHit], dict[str, Any]]:
        config = self.settings.retrieval
        depth = int(config["candidate_depth"])
        bm25_scores = self.bm25_scores(query, domain)
        bm25_indices = self._top_indices(bm25_scores, depth)
        bm25_ranks = {index: rank for rank, index in enumerate(bm25_indices, start=1)}

        dense_indices: list[int] = []
        dense_scores: np.ndarray | None = None
        if query_vector is not None:
            vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            if vector.shape[0] != self.vectors.shape[1]:
                raise ValueError(
                    f"Query vector dimension {vector.shape[0]} does not match {self.vectors.shape[1]}."
                )
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("Query vector has zero norm.")
            vector = vector / norm
            dense_scores = np.asarray(self.vectors @ vector, dtype=np.float32)
            dense_scores[~self._domain_mask(domain)] = -np.inf
            dense_indices = self._top_indices(dense_scores, depth)
        dense_ranks = {index: rank for rank, index in enumerate(dense_indices, start=1)}

        rank_constant = float(config["rank_constant"])
        bm25_weight = float(config["bm25_weight"])
        dense_weight = float(config["dense_weight"])
        fused: defaultdict[int, float] = defaultdict(float)
        for index, rank in bm25_ranks.items():
            fused[index] += bm25_weight / (rank_constant + rank)
        for index, rank in dense_ranks.items():
            fused[index] += dense_weight / (rank_constant + rank)
        ordered = sorted(fused, key=lambda index: (-fused[index], str(self.chunks[index]["id"])))
        ordered = ordered[: int(config["returned_chunks"])]
        hits = [
            RetrievalHit(
                chunk_id=str(self.chunks[index]["id"]),
                rank=rank,
                score=float(fused[index]),
                bm25_rank=bm25_ranks.get(index),
                dense_rank=dense_ranks.get(index),
                dense_similarity=(float(dense_scores[index]) if dense_scores is not None else None),
                chunk=self.chunks[index],
            )
            for rank, index in enumerate(ordered, start=1)
        ]
        diagnostics = {
            "mode": "hybrid_rrf" if query_vector is not None else "bm25",
            "bm25_top_score": float(bm25_scores[bm25_indices[0]]) if bm25_indices else 0.0,
            "dense_top_similarity": (
                float(dense_scores[dense_indices[0]])
                if dense_scores is not None and dense_indices
                else None
            ),
            "candidate_depth": depth,
        }
        return hits, diagnostics


class EvidenceIndex:
    def __init__(self, documents_path: Path):
        self.spans: dict[str, dict[str, Any]] = {}
        self.section_spans: dict[tuple[str, str], list[str]] = {}
        for document in iter_jsonl(documents_path):
            document_id = str(document["id"])
            for section in document["sections"]:
                section_id = str(section["id"])
                ordered: list[str] = []
                for position, span in enumerate(section["spans"]):
                    span_id = str(span["id"])
                    ordered.append(span_id)
                    self.spans[span_id] = {
                        "span_id": span_id,
                        "document_id": document_id,
                        "domain": document["domain"],
                        "document_title": document["title"],
                        "source_document_id": document.get("provenance", {}).get("source_document_id"),
                        "section_id": section_id,
                        "section_title": section.get("title", ""),
                        "parent_titles": [item.get("text", "") for item in section.get("parent_titles", [])],
                        "section_position": position,
                        "text": str(span["text"]).strip(),
                    }
                self.section_spans[(document_id, section_id)] = ordered

    def select(
        self, query: str, hits: list[RetrievalHit], maximum: int
    ) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query))
        candidates: dict[str, tuple[float, dict[str, Any]]] = {}
        for hit in hits:
            for span_id in hit.chunk.get("source_span_ids", []):
                span = self.spans.get(str(span_id))
                if not span or not span["text"]:
                    continue
                span_terms = set(tokenize(span["text"]))
                overlap = len(query_terms.intersection(span_terms)) / max(1, len(query_terms))
                score = hit.score + 0.25 * overlap + 0.01 / hit.rank
                existing = candidates.get(str(span_id))
                if existing is None or score > existing[0]:
                    candidates[str(span_id)] = (score, span)
        ranked = sorted(candidates.values(), key=lambda item: (-item[0], item[1]["span_id"]))
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(span: dict[str, Any], score: float) -> None:
            if span["span_id"] in selected_ids or len(selected) >= maximum:
                return
            selected_ids.add(span["span_id"])
            selected.append({**span, "selection_score": float(score)})

        for score, span in ranked:
            if len(selected) >= max(1, maximum - 4):
                break
            add(span, score)
        for anchor in list(selected[:4]):
            ordered = self.section_spans.get((anchor["document_id"], anchor["section_id"]), [])
            position = int(anchor["section_position"])
            for neighbor_position in (position - 1, position + 1):
                if 0 <= neighbor_position < len(ordered):
                    neighbor = self.spans[ordered[neighbor_position]]
                    add(neighbor, anchor["selection_score"] - 0.001)
        for score, span in ranked:
            add(span, score)
            if len(selected) >= maximum:
                break
        selected.sort(key=lambda item: (-item["selection_score"], item["span_id"]))
        for rank, item in enumerate(selected, start=1):
            item["evidence_id"] = f"E{rank}"
            item["priority_rank"] = rank
        return selected
