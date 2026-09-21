from scripts.plan_contract import (
    conflicting_years_with_conversation,
    unsupported_acronym_expansions,
    unsupported_organization_attributions,
)


def test_flags_organization_absent_from_authorized_evidence() -> None:
    claim = {
        "claim_text": "Federal Student Aid has resources for parents.",
        "evidence_ids": ["E1"],
    }
    evidence = [{"evidence_id": "E1", "text": "Resources for parents include saving early."}]
    assert unsupported_organization_attributions(claim, evidence) == ["Federal Student Aid"]


def test_allows_attribution_present_in_authorized_evidence() -> None:
    claim = {
        "claim_text": "Social Security provides online services.",
        "evidence_ids": ["E1"],
    }
    evidence = [{"evidence_id": "E1", "text": "Social Security provides online services."}]
    assert unsupported_organization_attributions(claim, evidence) == []


def test_detects_conflicting_year_range_endpoint() -> None:
    assert conflicting_years_with_conversation(
        "The program ran between 1942 and 1975.",
        "You said the period was 1942 through 1972.",
    ) == ["1975"]


def test_flags_acronym_expansion_absent_from_cited_evidence() -> None:
    claim = {
        "claim_text": "COA stands for cost of attendance.",
        "evidence_ids": ["E1"],
    }
    evidence = [{"evidence_id": "E1", "text": "Your COA is the estimate of"}]
    assert unsupported_acronym_expansions(claim, evidence) == ["COA"]


def test_allows_acronym_expansion_present_in_cited_evidence() -> None:
    claim = {
        "claim_text": "COA stands for cost of attendance.",
        "evidence_ids": ["E1"],
    }
    evidence = [{"evidence_id": "E1", "text": "COA means cost of attendance."}]
    assert unsupported_acronym_expansions(claim, evidence) == []
