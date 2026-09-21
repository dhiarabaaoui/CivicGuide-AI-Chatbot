"""Streamlit interface for the independent external clean-validation workflow."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
PACK_ROOT = ROOT / "data" / "processed" / "multidoc2dial_v1" / "clean_validation"
PACKS = {
    "Questions existantes de la base (simple)": (
        PACK_ROOT / "existing_database_annotation_pack_v1" / "annotation_tasks.jsonl"
    ),
    "Questions externes indépendantes (production)": (
        PACK_ROOT / "external_annotation_pack_v1" / "annotation_tasks.jsonl"
    ),
}
ROLES = ("case_author", "annotator_a", "annotator_b", "adjudicator")
ACTIONS = ("answer", "ask_followup", "abstain")


def load_tasks(task_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in task_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save_tasks(task_path: Path, tasks: list[dict[str, Any]]) -> None:
    task_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix="human_annotations_", suffix=".jsonl", dir=task_path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            for task in tasks:
                stream.write(json.dumps(task, ensure_ascii=False) + "\n")
        os.replace(temporary_name, task_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reference_complete(value: dict[str, Any]) -> bool:
    return bool(
        value.get("annotator_id")
        and value.get("completed_at_utc")
        and value.get("authored_conversation_accepted") is True
        and value.get("expected_action") in ACTIONS
        and value.get("reference_answer")
    )


def score_complete(value: dict[str, Any]) -> bool:
    keys = (
        "routing_correct",
        "contract_valid",
        "citation_valid",
        "faithfulness_pass",
        "completeness_pass",
        "overall_pass",
    )
    return reference_complete(value) and all(
        isinstance(value.get(key), bool) for key in keys
    ) and isinstance(value.get("correctness_0_to_4"), int)


def render_progress(tasks: list[dict[str, Any]]) -> None:
    progress = {
        "Conversations": sum(bool(task.get("conversation")) for task in tasks),
        "Références A": sum(reference_complete(task.get("annotator_a") or {}) for task in tasks),
        "Références B": sum(reference_complete(task.get("annotator_b") or {}) for task in tasks),
        "Réponses candidat": sum(task.get("candidate_response") is not None for task in tasks),
        "Scores A": sum(score_complete(task.get("annotator_a") or {}) for task in tasks),
        "Scores B": sum(score_complete(task.get("annotator_b") or {}) for task in tasks),
        "Adjudications": sum(
            (task.get("adjudication") or {}).get("status") == "completed"
            for task in tasks
        ),
    }
    for label, count in progress.items():
        st.sidebar.progress(count / 80, text=f"{label}: {count}/80")


def parse_conversation(value: str) -> list[dict[str, str]]:
    conversation = json.loads(value)
    if not isinstance(conversation, list) or not conversation:
        raise ValueError("La conversation doit être une liste JSON non vide.")
    for turn in conversation:
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant"}:
            raise ValueError("Chaque tour doit avoir un rôle user ou assistant.")
        if not isinstance(turn.get("text"), str) or not turn["text"].strip():
            raise ValueError("Chaque tour doit contenir un texte non vide.")
    if conversation[-1]["role"] != "user":
        raise ValueError("Le dernier tour doit appartenir à l’utilisateur.")
    return conversation


def render_case_author(task: dict[str, Any], actor_id: str) -> bool:
    if not actor_id:
        st.warning("Saisissez votre identifiant indépendant.")
        return False
    existing_author = task.get("case_author_id")
    if existing_author and existing_author != actor_id:
        st.error(f"Ce cas est verrouillé par l’auteur {existing_author}.")
        return False
    initial = task.get("conversation") or task.get("conversation_draft") or []
    value = st.text_area(
        "Conversation JSON",
        value=json.dumps(initial, ensure_ascii=False, indent=2),
        height=220,
    )
    if st.button("Valider et verrouiller la conversation", type="primary"):
        try:
            conversation = parse_conversation(value)
        except (ValueError, json.JSONDecodeError) as error:
            st.error(str(error))
            return False
        task["conversation"] = conversation
        task["case_author_id"] = actor_id
        task["case_authored_at_utc"] = utc_now()
        task["authoring_status"] = "authored_pending_dual_reference_annotation"
        return True
    return False


def render_annotation(task: dict[str, Any], role: str, actor_id: str) -> bool:
    if not task.get("conversation"):
        st.error("La conversation doit d’abord être verrouillée par l’auteur du cas.")
        return False
    if not actor_id:
        st.warning("Saisissez votre identifiant indépendant.")
        return False
    if actor_id == task.get("case_author_id"):
        st.error("L’auteur du cas ne peut pas annoter ce même cas.")
        return False
    other_role = "annotator_b" if role == "annotator_a" else "annotator_a"
    if actor_id == (task.get(other_role) or {}).get("annotator_id"):
        st.error("Les annotateurs A et B doivent être deux personnes différentes.")
        return False
    current = task.get(role) or {}
    if current.get("annotator_id") and current["annotator_id"] != actor_id:
        st.error(f"Cette annotation est verrouillée par {current['annotator_id']}.")
        return False
    st.json(task["conversation"])
    existing_reference = task.get("dataset_reference") or {}
    if existing_reference:
        st.success("Cette question possède déjà une réponse officielle dans la base.")
        st.write(f"**Action attendue :** {existing_reference['expected_action']}")
        st.write(f"**Réponse de référence :** {existing_reference['reference_answer']}")
        st.write(
            "**Preuves de référence :** "
            + ", ".join(existing_reference.get("supporting_span_ids", []))
        )
        if task.get("candidate_response") is None:
            st.info(
                "Aucune annotation n’est nécessaire avant l’exécution du candidat. "
                "Revenez ici après la génération pour noter sa réponse."
            )
            return False
    action_value = existing_reference.get("expected_action") or current.get("expected_action")
    action_index = ACTIONS.index(action_value) if action_value in ACTIONS else 0
    expected_action = st.selectbox(
        "Action attendue", ACTIONS, index=action_index, disabled=bool(existing_reference)
    )
    reference_answer = st.text_area(
        "Réponse de référence",
        value=existing_reference.get("reference_answer") or current.get("reference_answer") or "",
        disabled=bool(existing_reference),
    )
    available_ids = task["source_section_ids"]
    supporting_ids = st.multiselect(
        "Sections justificatives",
        available_ids,
        default=[
            value
            for value in (
                existing_reference.get("supporting_span_ids")
                or current.get("supporting_span_ids", [])
            )
            if value in available_ids
        ],
        disabled=bool(existing_reference),
    )
    accepts = st.checkbox(
        "J’accepte la conversation comme naturelle et conforme au scénario.",
        value=bool(existing_reference) or current.get("authored_conversation_accepted") is True,
        disabled=bool(existing_reference),
    )
    notes = st.text_area("Notes", value=current.get("notes") or "")
    score_values: dict[str, Any] = {}
    if task.get("candidate_response") is not None:
        st.subheader("Évaluation aveugle du candidat")
        response = task["candidate_response"]
        payload = response.get("payload") or {}
        st.write(f"**Action produite :** {response.get('runtime_action', 'inconnue')}")
        st.write(f"**Réponse :** {payload.get('answer', '')}")
        citations = payload.get("citations") or []
        if citations:
            st.write("**Citations produites :**")
            for citation in citations:
                st.write(
                    f"- {citation.get('evidence_id', '?')} : "
                    f"{citation.get('evidence_quote', '')}"
                )
        else:
            st.caption("Aucune citation produite.")
        st.caption(
            "Le jugement automatique est volontairement masqué ici pour ne pas influencer votre note."
        )
        for key, label in (
            ("routing_correct", "Routage correct"),
            ("contract_valid", "Contrat valide"),
            ("citation_valid", "Citations valides"),
            ("faithfulness_pass", "Fidèle"),
            ("completeness_pass", "Complet"),
            ("overall_pass", "Réussite globale"),
        ):
            score_values[key] = st.checkbox(label, value=current.get(key) is True, key=f"{role}_{key}")
        score_values["correctness_0_to_4"] = st.slider(
            "Exactitude", 0, 4, int(current.get("correctness_0_to_4") or 0), key=f"{role}_correctness"
        )
    if st.button("Enregistrer mon annotation", type="primary"):
        if not accepts or not reference_answer.strip() or not supporting_ids:
            st.error("Acceptation, réponse de référence et au moins une section sont obligatoires.")
            return False
        task[role] = {
            **current,
            "annotator_id": actor_id,
            "completed_at_utc": utc_now(),
            "authored_conversation_accepted": True,
            "expected_action": expected_action,
            "reference_answer": reference_answer.strip(),
            "supporting_span_ids": supporting_ids,
            "notes": notes.strip() or None,
            **score_values,
        }
        return True
    return False


def render_adjudication(task: dict[str, Any], actor_id: str) -> bool:
    a, b = task.get("annotator_a") or {}, task.get("annotator_b") or {}
    if not (score_complete(a) and score_complete(b)):
        st.error("Les deux annotations et leurs scores doivent être complets.")
        return False
    if not actor_id or actor_id in {task.get("case_author_id"), a.get("annotator_id"), b.get("annotator_id")}:
        st.error("L’adjudicateur doit être une troisième personne indépendante.")
        return False
    st.subheader("Annotateur A")
    st.json(a)
    st.subheader("Annotateur B")
    st.json(b)
    current = task.get("adjudication") or {}
    action = st.selectbox("Action finale", ACTIONS)
    reference = st.text_area("Réponse de référence finale", value=current.get("reference_answer") or "")
    supporting = st.multiselect("Sections finales", task["source_section_ids"])
    correctness = st.slider("Exactitude finale", 0, 4, int(current.get("correctness_0_to_4") or 0))
    values = {
        key: st.checkbox(label, value=current.get(key) is True, key=f"adj_{key}")
        for key, label in (
            ("routing_correct", "Routage correct"),
            ("contract_valid", "Contrat valide"),
            ("citation_valid", "Citations valides"),
            ("faithfulness_pass", "Fidèle"),
            ("completeness_pass", "Complet"),
            ("overall_pass", "Réussite globale"),
        )
    }
    notes = st.text_area("Décision et justification")
    if st.button("Finaliser l’adjudication", type="primary"):
        if not reference.strip() or not supporting or not notes.strip():
            st.error("Réponse, sections et justification sont obligatoires.")
            return False
        task["adjudication"] = {
            "status": "completed",
            "adjudicator_id": actor_id,
            "annotator_id": actor_id,
            "completed_at_utc": utc_now(),
            "authored_conversation_accepted": True,
            "expected_action": action,
            "reference_answer": reference.strip(),
            "supporting_span_ids": supporting,
            "correctness_0_to_4": correctness,
            "notes": notes.strip(),
            **values,
        }
        return True
    return False


def main() -> None:
    st.set_page_config(page_title="Annotation qualité du RAG", layout="wide")
    st.title("Annotation qualité du RAG")
    pack_label = st.sidebar.selectbox("Jeu de questions", tuple(PACKS))
    task_path = PACKS[pack_label]
    tasks = load_tasks(task_path)
    render_progress(tasks)
    internal_pack = "Questions existantes" in pack_label
    if internal_pack:
        st.info(
            "Les questions et leurs réponses officielles viennent déjà de la base. "
            "Aucune annotation n’est nécessaire avant le test du candidat. Les rôles A et B "
            "serviront ensuite à noter ses réponses."
        )
    else:
        st.info(
            "Ce parcours externe commence par la validation des questions par l’auteur du cas, "
            "puis par deux annotations indépendantes."
        )
    actor_id = st.sidebar.text_input(
        "Votre nom ou identifiant", placeholder="Exemple : annotateur_amine"
    ).strip()
    role_choices = ("annotator_a", "annotator_b", "adjudicator") if internal_pack else ROLES
    role_labels = {
        "case_author": "Auteur des questions",
        "annotator_a": "Annotateur A",
        "annotator_b": "Annotateur B",
        "adjudicator": "Adjudicateur",
    }
    role = st.sidebar.selectbox("Votre rôle", role_choices, format_func=role_labels.get)
    domains = sorted({task["domain"] for task in tasks})
    domain = st.sidebar.selectbox("Domaine", domains)
    visible = [task for task in tasks if task["domain"] == domain]
    case_id = st.sidebar.selectbox("Cas", [task["case_id"] for task in visible])
    index = next(i for i, task in enumerate(tasks) if task["case_id"] == case_id)
    task = tasks[index]
    st.caption(f"{task['domain']} · {task['scenario']} · {task.get('source_url', 'source locale')}")
    st.subheader(task["task_instruction"])
    for evidence in task["source_evidence"]:
        with st.expander(f"{evidence['id']} — {evidence['title']}", expanded=True):
            st.write(evidence["text"])
    changed = (
        render_case_author(task, actor_id)
        if role == "case_author"
        else (
            render_annotation(task, role, actor_id)
            if role in {"annotator_a", "annotator_b"}
            else render_adjudication(task, actor_id)
        )
    )
    if changed:
        tasks[index] = task
        save_tasks(task_path, tasks)
        st.success("Enregistrement atomique effectué.")
        st.rerun()
    st.sidebar.caption("Répartition: " + str(dict(Counter(task["scenario"] for task in visible))))


if __name__ == "__main__":
    main()
