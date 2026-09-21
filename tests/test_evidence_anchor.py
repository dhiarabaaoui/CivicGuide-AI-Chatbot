from scripts.evidence_anchor import numeric_anchors, promote_numeric_anchor_sources


def test_extracts_numeric_unit_anchor() -> None:
    assert numeric_anchors("What if it closed within 120 days?") == ["120 days"]


def test_promotes_matching_source_without_displacing_primary() -> None:
    sources = [
        {"span_id": "primary", "text": "General eligibility"},
        {"span_id": "other", "text": "Another condition"},
        {"span_id": "match", "text": "The school closed within 120 days after withdrawal."},
    ]
    promoted, ids = promote_numeric_anchor_sources(sources, ["120 days"])
    assert [row["span_id"] for row in promoted] == ["primary", "match", "other"]
    assert ids == ["match"]


def test_promotes_governing_eligibility_clause_with_numeric_child() -> None:
    sources = [
        {"span_id": "primary", "section_id": "a", "source_span_id": "1", "text": "Overview"},
        {"span_id": "governor", "section_id": "b", "source_span_id": "12", "text": "You are not eligible if any of the following is true:"},
        {"span_id": "child", "section_id": "b", "source_span_id": "14", "text": "you withdrew more than 120 days before closure."},
    ]
    promoted, ids = promote_numeric_anchor_sources(sources, ["120 days"])
    assert [row["span_id"] for row in promoted] == ["primary", "governor", "child"]
    assert ids == ["governor", "child"]
