"""Deterministic promotion of high-information query anchors in ranked evidence."""

from __future__ import annotations

import re
from typing import Any


def numeric_anchors(*texts: str) -> list[str]:
    anchors: list[str] = []
    for text in texts:
        normalized = " ".join(str(text).casefold().split())
        matches = re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:days?|months?|years?|percent|%|dollars?|hours?|weeks?)\b",
            normalized,
        )
        anchors.extend(" ".join(match.split()) for match in matches)
    return list(dict.fromkeys(anchors))


def promote_numeric_anchor_sources(
    sources: list[dict[str, Any]], anchors: list[str], protected_prefix: int = 1
) -> tuple[list[dict[str, Any]], list[str]]:
    """Move exact numeric-anchor matches after the protected primary evidence."""
    if not anchors or len(sources) <= protected_prefix:
        return [dict(source) for source in sources], []
    prefix = [dict(source) for source in sources[:protected_prefix]]
    remaining = [dict(source) for source in sources[protected_prefix:]]
    promoted_matches = [
        source
        for source in remaining
        if any(anchor in " ".join(source.get("text", "").casefold().split()) for anchor in anchors)
    ]
    governors: list[dict[str, Any]] = []
    for match in promoted_matches:
        try:
            match_position = int(match.get("source_span_id", -1))
        except (TypeError, ValueError):
            continue
        candidates: list[tuple[int, dict[str, Any]]] = []
        for source in remaining:
            if source in promoted_matches or source.get("section_id") != match.get("section_id"):
                continue
            try:
                position = int(source.get("source_span_id", -1))
            except (TypeError, ValueError):
                continue
            governing_text = source.get("text", "").casefold()
            if position < match_position and re.search(
                r"\b(?:not eligible|eligible|following|requirements?|criteria)\b",
                governing_text,
            ):
                candidates.append((position, source))
        if candidates:
            governor = dict(max(candidates, key=lambda item: item[0])[1])
            if governor not in governors:
                governors.append(governor)
    promoted = [*governors, *promoted_matches]
    rest = [source for source in remaining if source not in promoted]
    output = [*prefix, *promoted, *rest]
    for rank, source in enumerate(output, start=1):
        source["rank"] = rank
    return output, [source["span_id"] for source in promoted]
