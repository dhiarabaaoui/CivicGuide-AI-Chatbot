from __future__ import annotations

import csv
import html
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "generated"
    / "multidoc2dial_v1"
    / "error_analysis_experiments"
)

REQUIRED_COLUMNS = [
    "example_id",
    "domain",
    "blind_label",
    "question",
    "conversation_history",
    "reference_answer",
    "gold_evidence",
    "candidate_status",
    "candidate_answer",
    "candidate_cited_sources",
    "adjudicated_faithfulness",
    "adjudication_notes",
]
VALID_LABELS = ("pass", "fail")


def find_adjudication_files() -> list[Path]:
    override = os.environ.get("FAITHFULNESS_ADJUDICATION_CSV", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        return [candidate] if candidate.exists() else []
    if not REPORTS_ROOT.exists():
        return []
    return sorted(
        REPORTS_ROOT.glob("answer_error_analysis_v1_*/faithfulness_adjudication_blind.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def detect_delimiter(path: Path) -> str:
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    return ";" if header.count(";") > header.count(",") else ","


def load_adjudication(path: Path) -> tuple[pd.DataFrame, str]:
    delimiter = detect_delimiter(path)
    dataframe = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        sep=delimiter,
    )
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Colonnes absentes : {', '.join(missing)}")
    if dataframe.duplicated(["example_id", "blind_label"]).any():
        raise ValueError("Le dossier contient des couples example_id/blind_label dupliqués.")
    invalid = dataframe[
        ~dataframe["adjudicated_faithfulness"].isin(["", *VALID_LABELS])
    ]
    if not invalid.empty:
        raise ValueError("Le dossier contient des labels différents de pass/fail.")
    return dataframe, delimiter


def make_backup_once(path: Path) -> Path:
    session_key = f"adjudication_backup::{path}"
    if session_key in st.session_state:
        return Path(st.session_state[session_key])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup)
    st.session_state[session_key] = str(backup)
    return backup


def atomic_save(dataframe: pd.DataFrame, path: Path, delimiter: str) -> Path:
    backup = make_backup_once(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        dataframe.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
            sep=delimiter,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        verification, _ = load_adjudication(temporary)
        if len(verification) != len(dataframe):
            raise ValueError("Le contrôle de sauvegarde a détecté un nombre de lignes différent.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def render_panel(title: str, content: str, css_class: str = "panel") -> None:
    safe_title = html.escape(title)
    safe_content = html.escape(content or "—").replace("\n", "<br>")
    st.markdown(
        f'<section class="{css_class}"><h3>{safe_title}</h3>'
        f'<div class="panel-content">{safe_content}</div></section>',
        unsafe_allow_html=True,
    )


def next_incomplete(dataframe: pd.DataFrame, current_index: int) -> int:
    for offset in range(1, len(dataframe) + 1):
        candidate = (current_index + offset) % len(dataframe)
        if dataframe.iloc[candidate]["adjudicated_faithfulness"] not in VALID_LABELS:
            return candidate
    return min(current_index + 1, len(dataframe) - 1)


def navigate(index: int, total: int) -> None:
    st.session_state["adjudication_index"] = max(0, min(index, total - 1))
    st.rerun()


st.set_page_config(
    page_title="Adjudication faithfulness",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1450px; padding-top: 1.3rem; padding-bottom: 4rem; }
      .panel, .answer-panel, .question-panel {
        background: white; border: 1px solid #dfe5ee; border-radius: 14px;
        padding: 1rem 1.1rem; margin: 0.35rem 0 0.8rem 0;
        box-shadow: 0 2px 8px rgba(23, 35, 55, 0.04);
      }
      .question-panel { border-left: 5px solid #7c3aed; background: #faf7ff; }
      .answer-panel { border-top: 4px solid #0f766e; }
      .panel h3, .answer-panel h3, .question-panel h3 {
        font-size: 0.95rem; margin: 0 0 0.65rem 0; color: #334155;
        text-transform: uppercase; letter-spacing: 0.04em;
      }
      .panel-content { color: #172033; line-height: 1.6; }
      .source-box {
        max-height: 520px; overflow-y: auto; white-space: pre-wrap;
        background: #0f172a; color: #e2e8f0; border-radius: 10px;
        padding: 0.9rem; font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        font-size: 0.84rem; line-height: 1.5;
      }
      div[data-testid="stForm"] { background: white; border-radius: 14px; padding: 0.5rem 1rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔎 Adjudication aveugle de la faithfulness")
st.caption(
    "Tu tranches uniquement les désaccords humain–juge. Les anciens labels et la variante 5k/8k sont masqués."
)

files = find_adjudication_files()
if not files:
    st.error("Aucun dossier d’adjudication trouvé. Exécute d’abord le notebook 12 avec Run All.")
    st.stop()

labels = {
    f"{path.parent.name} — {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}": path
    for path in files
}
selected = st.sidebar.selectbox("Expérience", list(labels), index=0)
path = labels[selected]

try:
    dataframe, delimiter = load_adjudication(path)
except Exception as error:
    st.error(f"Impossible de lire le dossier : {error}")
    st.stop()

total = len(dataframe)
complete_mask = dataframe["adjudicated_faithfulness"].isin(VALID_LABELS)
completed = int(complete_mask.sum())
if "adjudication_index" not in st.session_state:
    st.session_state["adjudication_index"] = next(
        (index for index, value in enumerate(complete_mask) if not value), 0
    )
index = min(int(st.session_state["adjudication_index"]), total - 1)
row = dataframe.iloc[index]

st.sidebar.metric("Progression", f"{completed} / {total}")
st.sidebar.progress(completed / total if total else 0.0)
st.sidebar.caption("Le juge GPT-5.6 et le premier label humain ne sont pas affichés.")

jump = st.sidebar.selectbox(
    "Aller à un cas",
    list(range(total)),
    index=index,
    format_func=lambda value: (
        f"{'✓' if complete_mask.iloc[value] else '○'} {value + 1:02d}/{total} · "
        f"{dataframe.iloc[value]['domain'].upper()} · candidat {dataframe.iloc[value]['blind_label']}"
    ),
)
if jump != index:
    navigate(jump, total)

previous_column, incomplete_column, next_column = st.columns([1, 2, 1])
with previous_column:
    if st.button("← Précédent", use_container_width=True, disabled=index == 0):
        navigate(index - 1, total)
with incomplete_column:
    if st.button("Prochain cas non terminé", use_container_width=True, disabled=completed == total):
        navigate(next_incomplete(dataframe, index), total)
with next_column:
    if st.button("Suivant →", use_container_width=True, disabled=index == total - 1):
        navigate(index + 1, total)

st.markdown(
    f"**Cas {index + 1}/{total}** · Domaine `{html.escape(row['domain'].upper())}` · "
    f"Candidat `{html.escape(row['blind_label'])}` · ID `{html.escape(row['example_id'])}`"
)

render_panel("Question actuelle", row["question"], "question-panel")
history_tab, reference_tab = st.tabs(["💬 Historique", "🎯 Référence et preuves gold"])
with history_tab:
    render_panel("Conversation avant la question", row["conversation_history"])
with reference_tab:
    render_panel("Réponse de référence", row["reference_answer"])
    with st.expander("Preuves gold du dataset", expanded=True):
        st.markdown(
            f'<div class="source-box">{html.escape(row["gold_evidence"])}</div>',
            unsafe_allow_html=True,
        )

render_panel(
    f"Réponse du candidat {row['blind_label']}",
    row["candidate_answer"],
    "answer-panel",
)
with st.expander("Sources réellement citées par cette réponse", expanded=True):
    sources = row["candidate_cited_sources"] or "Aucune source citée."
    st.markdown(f'<div class="source-box">{html.escape(sources)}</div>', unsafe_allow_html=True)

st.info(
    "Décide uniquement ceci : tous les faits importants de la réponse sont-ils soutenus par les sources "
    "réellement citées ? La réponse gold aide à comprendre le sujet, mais ne remplace pas les citations du candidat."
)

existing_index = (
    VALID_LABELS.index(row["adjudicated_faithfulness"])
    if row["adjudicated_faithfulness"] in VALID_LABELS
    else None
)
with st.form(f"adjudication_form::{row['example_id']}::{row['blind_label']}"):
    decision = st.radio(
        "Verdict final de faithfulness",
        VALID_LABELS,
        index=existing_index,
        horizontal=True,
        format_func=lambda value: (
            "PASS — tous les faits importants sont soutenus"
            if value == "pass"
            else "FAIL — au moins un fait important n’est pas soutenu"
        ),
    )
    notes = st.text_area(
        "Notes facultatives",
        value=row["adjudication_notes"],
        placeholder="Indique brièvement la phrase non soutenue si le verdict est fail.",
        height=90,
    )
    save_column, save_next_column = st.columns(2)
    with save_column:
        save = st.form_submit_button("Enregistrer", use_container_width=True)
    with save_next_column:
        save_next = st.form_submit_button("Enregistrer et suivant →", type="primary", use_container_width=True)

if save or save_next:
    if decision is None:
        st.error("Choisis pass ou fail avant d’enregistrer.")
    else:
        try:
            latest, latest_delimiter = load_adjudication(path)
            matches = latest.index[
                (latest["example_id"] == row["example_id"])
                & (latest["blind_label"] == row["blind_label"])
            ].tolist()
            if len(matches) != 1:
                raise ValueError("Le cas courant est introuvable ou dupliqué.")
            target = matches[0]
            latest.loc[target, "adjudicated_faithfulness"] = decision
            latest.loc[target, "adjudication_notes"] = str(notes).strip()
            atomic_save(latest, path, latest_delimiter)
        except PermissionError:
            st.error("Le fichier est verrouillé. Ferme-le dans Excel puis réessaie.")
        except Exception as error:
            st.error(f"Sauvegarde impossible : {error}")
        else:
            st.toast("Adjudication enregistrée", icon="✅")
            if save_next:
                st.session_state["adjudication_index"] = next_incomplete(latest, index)
            st.rerun()

if completed == total:
    st.success(
        "Les 22 désaccords sont adjudicés. Arrête l’application puis relance le notebook 12 avec Run All."
    )

