"""Deterministic semantic-shape checks for generated claim plans."""

from __future__ import annotations

import re
from typing import Any


ATTRIBUTION_PATTERN = re.compile(
    r"\b((?:[A-Z][A-Za-z.&'-]*)(?:\s+[A-Z][A-Za-z.&'-]*){1,5})\s+"
    r"(?:has|have|provides?|offers?|says?|requires?|allows?)\b"
)


def unsupported_organization_attributions(
    claim: dict[str, Any], evidence_catalog: list[dict[str, Any]]
) -> list[str]:
    """Find named subjects attributed facts absent from authorized evidence text."""
    evidence_by_id = {
        str(item.get("evidence_id")): str(item.get("text", ""))
        for item in evidence_catalog
    }
    corpus = " ".join(
        evidence_by_id.get(str(evidence_id), "")
        for evidence_id in claim.get("evidence_ids", [])
    ).casefold()
    unsupported: list[str] = []
    for match in ATTRIBUTION_PATTERN.finditer(str(claim.get("claim_text", ""))):
        entity = " ".join(match.group(1).split())
        if entity.casefold() not in corpus:
            unsupported.append(entity)
    return list(dict.fromkeys(unsupported))


def conflicting_years_with_conversation(claim_text: str, conversation_text: str) -> list[str]:
    """Detect a likely date-range conflict sharing one endpoint with the dialogue."""
    claim_years = set(re.findall(r"\b(?:19|20)\d{2}\b", claim_text))
    conversation_years = set(re.findall(r"\b(?:19|20)\d{2}\b", conversation_text))
    if not claim_years or not conversation_years or not claim_years.intersection(conversation_years):
        return []
    return sorted(claim_years - conversation_years)


def unsupported_acronym_expansions(
    claim: dict[str, Any], evidence_catalog: list[dict[str, Any]]
) -> list[str]:
    """Detect acronym definitions that the claim's cited evidence never states."""
    evidence_by_id = {
        str(item.get("evidence_id")): str(item.get("text", ""))
        for item in evidence_catalog
    }
    corpus = " ".join(
        evidence_by_id.get(str(evidence_id), "")
        for evidence_id in claim.get("evidence_ids", [])
    ).casefold()
    claim_text = str(claim.get("claim_text", ""))
    matches = [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r"\b([A-Z][A-Z0-9]{1,7})\s+(?:stands for|means)\s+([^.;:]+)",
            claim_text,
        )
    ]
    matches.extend(
        (match.group(1), match.group(2))
        for match in re.finditer(r"\b([A-Z][A-Z0-9]{1,7})\s*\(([^)]+)\)", claim_text)
    )
    unsupported = []
    for acronym, expansion in matches:
        normalized_expansion = " ".join(expansion.casefold().split())
        if acronym.casefold() not in corpus or normalized_expansion not in " ".join(corpus.split()):
            unsupported.append(acronym)
    return list(dict.fromkeys(unsupported))
