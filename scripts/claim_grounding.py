"""Runtime-observable personal-state grounding utilities for generated claims."""

from __future__ import annotations

import re
from typing import Any


PERSONAL_SUBJECT = r"(?:you|your|you['’]re|you['’]ve)"
FACTUAL_VERBS = (
    r"(?:are|aren['’]t|are not|were|weren['’]t|were not|have|haven['’]t|have not|"
    r"had|did|didn['’]t|did not|do not|don['’]t|served|received|enrolled|withdrew|"
    r"applied|worked|lived|closed|lost|became|completed|failed|passed|meet|meets)"
)


def normalize_space(text: str) -> str:
    return " ".join(text.strip().split())


def parse_conversation(input_text: str) -> list[dict[str, str]]:
    match = re.search(
        r"<CONVERSATION>\s*(.*?)\s*</CONVERSATION>", input_text, flags=re.DOTALL
    )
    if not match:
        return []
    turns: list[dict[str, str]] = []
    for line in match.group(1).splitlines():
        role, separator, text = line.partition(":")
        role = role.strip().lower()
        if separator and role in {"user", "agent"}:
            turns.append({"role": role, "text": normalize_space(text)})
    return turns


def infer_polarity(text: str) -> str:
    normalized = normalize_space(text).casefold()
    negative = bool(
        re.search(
            r"^(?:yes[, ]+)?(?:no\b|not\b|never\b|(?:i\s+)?(?:do not|don['’]t|"
            r"did not|didn['’]t|am not|haven['’]t|have not|cannot|can['’]t))",
            normalized,
        )
    )
    positive = bool(
        re.match(r"^(?:yes\b|yeah\b|yep\b|correct\b|i (?:do|am|have)\b)", normalized)
    )
    if negative:
        return "negative"
    if positive:
        return "positive"
    return "unknown"


def _clean_question_body(text: str) -> str:
    normalized = normalize_space(text)
    fragments = re.findall(
        r"(?i)(?:^|[.!]\s+)((?:are|were|do|have|did|is|was|does|can|will)\s+you\b[^?]*\?)",
        normalized,
    )
    if fragments:
        normalized = fragments[-1]
    return normalized.rstrip(" ?.!")


def question_to_user_fact(question: str, polarity: str) -> str | None:
    """Convert common yes/no assistant questions into explicit user-state propositions."""
    text = _clean_question_body(question)
    case_match = re.search(
        r"(?i)\bcalled\s+(?:a|an)\s+([^,?]+),?\s+is that your case$", text
    )
    if case_match and polarity in {"positive", "negative"}:
        body = normalize_space(case_match.group(1))
        prefix = "You have" if polarity == "positive" else "You do not have"
        return f"{prefix} {body}."
    patterns = [
        (r"(?i)^you['’]ve\s+(.+)$", "You have {body}", "You have not {body}"),
        (r"(?i)^are you\s+(.+)$", "You are {body}", "You are not {body}"),
        (r"(?i)^were you\s+(.+)$", "You were {body}", "You were not {body}"),
        (r"(?i)^do you have\s+(.+)$", "You have {body}", "You do not have {body}"),
        (r"(?i)^have you\s+(.+)$", "You have {body}", "You have not {body}"),
        (r"(?i)^did you\s+(.+)$", "You did {body}", "You did not {body}"),
        (r"(?i)^is your\s+(.+)$", "Your {body} is true", "Your {body} is not true"),
        (r"(?i)^was your\s+(.+)$", "Your {body} was true", "Your {body} was not true"),
        (r"(?i)^did your\s+(.+)$", "Your {body} did happen", "Your {body} did not happen"),
        (r"(?i)^does your\s+(.+)$", "Your {body} does apply", "Your {body} does not apply"),
        (r"(?i)^can you\s+(.+)$", "You can {body}", "You cannot {body}"),
        (r"(?i)^will you\s+(.+)$", "You will {body}", "You will not {body}"),
        (r"(?i)^do you\s+(.+)$", "You {body}", "You do not {body}"),
    ]
    if polarity not in {"positive", "negative"}:
        return None
    for pattern, positive_template, negative_template in patterns:
        match = re.match(pattern, text)
        if match:
            template = positive_template if polarity == "positive" else negative_template
            return normalize_space(template.format(body=match.group(1))) + "."
    return None


def build_user_state(turns: list[dict[str, str]]) -> dict[str, Any]:
    explicit_user_utterances = [turn["text"] for turn in turns if turn["role"] == "user"]
    inferred_facts: list[dict[str, str]] = []
    for previous, current in zip(turns, turns[1:]):
        if previous["role"] != "agent" or current["role"] != "user":
            continue
        polarity = infer_polarity(current["text"])
        fact = question_to_user_fact(previous["text"], polarity)
        if fact:
            inferred_facts.append(
                {
                    "question": previous["text"],
                    "response": current["text"],
                    "polarity": polarity,
                    "fact": fact,
                }
            )
    premise_lines = [
        "The following statements are the only known facts about the user.",
        *[f"USER SAID: {text}" for text in explicit_user_utterances],
        *[f"RESOLVED FACT: {item['fact']}" for item in inferred_facts],
    ]
    return {
        "explicit_user_utterances": explicit_user_utterances,
        "inferred_facts": inferred_facts,
        "evidence_atoms": [
            *[
                {
                    "source": "explicit_user_utterance",
                    "text": text,
                    "assertive": not text.rstrip().endswith("?")
                    and not re.match(r"(?i)^(?:if|unless|whether)\b", text),
                }
                for text in explicit_user_utterances
            ],
            *[
                {
                    "source": "resolved_question_answer",
                    "text": item["fact"],
                    "assertive": True,
                }
                for item in inferred_facts
            ],
            *[
                {
                    "source": "resolved_question_answer_context",
                    "text": f"Question: {item['question']} Answer: {item['response']}",
                    "assertive": True,
                }
                for item in inferred_facts
            ],
        ],
        "premise": "\n".join(premise_lines),
    }


def fact_to_first_person(fact: str) -> str:
    """Put an assistant assertion in the same grammatical voice as a user utterance."""
    text = normalize_space(fact)
    replacements = (
        (r"(?i)^you are\b", "I am"),
        (r"(?i)^you were\b", "I was"),
        (r"(?i)^you have\b", "I have"),
        (r"(?i)^you did\b", "I did"),
        (r"(?i)^you do\b", "I do"),
        (r"(?i)^you can\b", "I can"),
        (r"(?i)^you cannot\b", "I cannot"),
        (r"(?i)^you will\b", "I will"),
        (r"(?i)^you\b", "I"),
        (r"(?i)^your\b", "My"),
    )
    for pattern, replacement in replacements:
        updated = re.sub(pattern, replacement, text, count=1)
        if updated != text:
            return updated
    return text


def surface_fact_supported(fact: str, user_utterances: list[str]) -> bool:
    """Conservative lexical support for short statements that NLI often marks neutral."""
    stopwords = {
        "a", "an", "the", "i", "you", "your", "my", "am", "are", "is", "was",
        "were", "do", "did", "have", "has", "now", "already", "yes", "yeah",
    }

    def tokens(value: str) -> set[str]:
        output: set[str] = set()
        for token in re.findall(r"[a-z0-9]+", value.casefold()):
            if token in stopwords or len(token) <= 1:
                continue
            output.add(token[:-1] if token.endswith("s") and len(token) > 3 else token)
        return output

    fact_tokens = tokens(fact_to_first_person(fact))
    if len(fact_tokens) < 2:
        return False
    for utterance in user_utterances:
        lowered = normalize_space(utterance).casefold()
        if utterance.rstrip().endswith("?") or re.match(
            r"^(?:tell me|what|when|where|why|how|can you|could you|would you|"
            r"i (?:need|want|would like|['’]d like) (?:help|information|info|to know))\b",
            lowered,
        ):
            continue
        if fact_tokens.issubset(tokens(utterance)):
            return True
    return False


def extract_personal_facts(sentence: str) -> list[str]:
    """Extract asserted personal premises, excluding general rules and instructions."""
    text = normalize_space(sentence)
    if not text or text.endswith("?"):
        return []
    facts: list[str] = []
    causal_pattern = re.compile(
        rf"(?i)\b(?:because|since|given that)\s+({PERSONAL_SUBJECT}.+?)(?=;|\.|$)"
    )
    for match in causal_pattern.finditer(text):
        sentence_start = text.rfind(".", 0, match.start()) + 1
        clause_prefix = text[sentence_start : match.start()]
        if re.search(r"(?i)\b(?:if|unless|whether)\b", clause_prefix):
            continue
        candidate = match.group(1).strip()
        candidate = re.split(
            r"(?i),\s*(?!(?:and\s+)?(?:you|your)\b)", candidate, maxsplit=1
        )[0]
        candidate = re.split(
            r"(?i),?\s+and\s+(?=(?:you|your)\s+(?:must|should|need|can|may|will)\b)",
            candidate,
            maxsplit=1,
        )[0]
        facts.extend(
            part.strip()
            for part in re.split(r"(?i),?\s+and\s+(?=(?:you|your)\b)", candidate)
        )

    leading = re.sub(r"(?i)^(?:yes|no)\s*[—–,:-]*\s*", "", text)
    leading_pattern = re.compile(
        rf"(?i)^({PERSONAL_SUBJECT}\s+{FACTUAL_VERBS}\b.+?)(?=,|;|\.|$)"
    )
    match = leading_pattern.search(leading)
    if match:
        facts.append(match.group(1).strip())

    unique: list[str] = []
    for fact in facts:
        cleaned = normalize_space(fact).rstrip(".,;:")
        lowered = cleaned.casefold()
        if re.fullmatch(r"you are (?:neither|either|not either)", lowered):
            continue
        if re.match(r"(?i)^you are (?:encouraged|required|allowed|advised|eligible)\b", cleaned):
            continue
        if cleaned and lowered not in {item.casefold() for item in unique}:
            unique.append(cleaned)
    return unique


def grounding_decision(
    personal_fact_scores: list[dict[str, Any]],
    minimum_entailment: float,
    maximum_contradiction: float,
) -> dict[str, Any]:
    unsafe = [
        row
        for row in personal_fact_scores
        if (
            not row.get("surface_supported")
            and row["entailment"] < minimum_entailment
        )
        or row["contradiction"] > maximum_contradiction
    ]
    return {
        "decision": "block" if unsafe else "allow",
        "safe": not unsafe,
        "unsupported_personal_facts": unsafe,
        "recommended_fallback": "ask_followup_or_conditionalize" if unsafe else "none",
    }
