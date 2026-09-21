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
    / "quality_evaluation_experiments"
)

REQUIRED_COLUMNS = [
    "example_id",
    "domain",
    "conversation_history",
    "current_question",
    "reference_answer",
    "gold_evidence",
    "candidate_a_status",
    "candidate_a_answer",
    "candidate_a_cited_sources",
    "candidate_b_status",
    "candidate_b_answer",
    "candidate_b_cited_sources",
    "human_winner",
    "human_a_faithfulness",
    "human_b_faithfulness",
    "human_a_correctness_0_to_4",
    "human_b_correctness_0_to_4",
    "human_notes",
]

LABEL_COLUMNS = [
    "human_winner",
    "human_a_faithfulness",
    "human_b_faithfulness",
    "human_a_correctness_0_to_4",
    "human_b_correctness_0_to_4",
]

WINNER_VALUES = ("A", "B", "tie")
FAITHFULNESS_VALUES = ("pass", "fail")
CORRECTNESS_VALUES = ("0", "1", "2", "3", "4")


def find_calibration_files() -> list[Path]:
    override = os.environ.get("HUMAN_CALIBRATION_CSV", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        return [candidate] if candidate.exists() else []
    if not REPORTS_ROOT.exists():
        return []
    return sorted(
        REPORTS_ROOT.glob("answer_quality_evaluation_v1_*/human_calibration_blind.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def detect_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline()
    comma_count = header.count(",")
    semicolon_count = header.count(";")
    return ";" if semicolon_count > comma_count else ","


def load_annotations(path: Path) -> tuple[pd.DataFrame, str]:
    delimiter = detect_delimiter(path)
    dataframe = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        sep=delimiter,
    )

    # Excel/Power Query peut renommer les colonnes Column1...Column18 et
    # déplacer les vrais en-têtes dans la première ligne. La récupération ne
    # s'active que si cette ligne correspond exactement au schéma attendu.
    recovered_excel_header = False
    if len(dataframe) > 0:
        possible_header = [str(value).strip() for value in dataframe.iloc[0].tolist()]
        if possible_header == REQUIRED_COLUMNS:
            dataframe = dataframe.iloc[1:].copy()
            dataframe.columns = REQUIRED_COLUMNS
            dataframe.reset_index(drop=True, inplace=True)
            recovered_excel_header = True

    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Colonnes absentes : {', '.join(missing)}")
    if dataframe["example_id"].duplicated().any():
        raise ValueError("Le fichier contient des example_id dupliqués.")
    dataframe.attrs["recovered_excel_header"] = recovered_excel_header
    return dataframe, delimiter


def row_is_complete(row: pd.Series) -> bool:
    return (
        row["human_winner"] in WINNER_VALUES
        and row["human_a_faithfulness"] in FAITHFULNESS_VALUES
        and row["human_b_faithfulness"] in FAITHFULNESS_VALUES
        and row["human_a_correctness_0_to_4"] in CORRECTNESS_VALUES
        and row["human_b_correctness_0_to_4"] in CORRECTNESS_VALUES
    )


def make_backup_once(path: Path) -> Path:
    session_key = f"backup::{path}"
    if session_key in st.session_state:
        return Path(st.session_state[session_key])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}.backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    st.session_state[session_key] = str(backup_path)
    return backup_path


def atomic_save(dataframe: pd.DataFrame, path: Path, delimiter: str) -> Path:
    backup_path = make_backup_once(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        dataframe.to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
            sep=delimiter,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        verification, _ = load_annotations(temporary_path)
        if len(verification) != len(dataframe):
            raise ValueError("La vérification de sauvegarde a détecté un nombre de lignes différent.")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return backup_path


def clean(value: object) -> str:
    return "" if value is None else str(value).strip()


def render_panel(title: str, content: str, css_class: str = "panel") -> None:
    safe_title = html.escape(title)
    safe_content = html.escape(content or "—").replace("\n", "<br>")
    st.markdown(
        f'<section class="{css_class}"><h3>{safe_title}</h3>'
        f'<div class="panel-content">{safe_content}</div></section>',
        unsafe_allow_html=True,
    )


def build_external_review_text(row: pd.Series, row_index: int, total_rows: int) -> str:
    """Crée un dossier autonome et aveugle pour un second évaluateur humain."""
    return f"""============================================================
IDENTIFICATION
============================================================
Exemple : {row_index + 1}/{total_rows}
Example ID : {row['example_id']}
Domaine : {row['domain']}

============================================================
QUESTION ACTUELLE
============================================================
{row['current_question']}

============================================================
HISTORIQUE DE CONVERSATION
============================================================
{row['conversation_history']}

============================================================
RÉPONSE DE RÉFÉRENCE
============================================================
{row['reference_answer']}

============================================================
PREUVES ATTENDUES DU DATASET
============================================================
{row['gold_evidence']}

============================================================
RÉPONSE A
============================================================
Statut de génération : {row['candidate_a_status']}

{row['candidate_a_answer']}

============================================================
SOURCES CITÉES PAR A
============================================================
{row['candidate_a_cited_sources'] or 'Aucune source citée.'}

============================================================
RÉPONSE B
============================================================
Statut de génération : {row['candidate_b_status']}

{row['candidate_b_answer']}

============================================================
SOURCES CITÉES PAR B
============================================================
{row['candidate_b_cited_sources'] or 'Aucune source citée.'}

============================================================
QUESTIONS D'ÉVALUATION
============================================================
1. Quelle est la meilleure réponse dans l'ensemble : A, B ou tie (égalité) ?
2. Faithfulness de A : pass ou fail ?
3. Faithfulness de B : pass ou fail ?
4. Correctness de A : quelle note de 0 à 4 ?
5. Correctness de B : quelle note de 0 à 4 ?

Pas besoin de m'expliquer tes choix. Donne-moi uniquement l'évaluation finale avec ce format exact :

human_winner: A | B | tie
human_a_faithfulness: pass | fail
human_b_faithfulness: pass | fail
human_a_correctness_0_to_4: 0 | 1 | 2 | 3 | 4
human_b_correctness_0_to_4: 0 | 1 | 2 | 3 | 4
"""


def move_to_next_incomplete(dataframe: pd.DataFrame, current_index: int) -> int:
    total = len(dataframe)
    for offset in range(1, total + 1):
        candidate_index = (current_index + offset) % total
        if not row_is_complete(dataframe.iloc[candidate_index]):
            return candidate_index
    return min(current_index + 1, total - 1)


def set_row_index(new_index: int, total: int) -> None:
    st.session_state["row_index"] = max(0, min(new_index, total - 1))
    st.rerun()


st.set_page_config(
    page_title="Calibration humaine RAG",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f5f7fb; }
      .block-container { max-width: 1500px; padding-top: 1.3rem; padding-bottom: 4rem; }
      .panel, .question-panel, .candidate-panel {
        background: white;
        border: 1px solid #dfe5ee;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin: 0.35rem 0 0.8rem 0;
        box-shadow: 0 2px 8px rgba(23, 35, 55, 0.04);
      }
      .question-panel { border-left: 5px solid #4f46e5; background: #f7f7ff; }
      .candidate-panel { min-height: 180px; border-top: 4px solid #0f766e; }
      .panel h3, .question-panel h3, .candidate-panel h3 {
        font-size: 0.95rem; margin: 0 0 0.65rem 0; color: #334155;
        text-transform: uppercase; letter-spacing: 0.04em;
      }
      .panel-content { color: #172033; line-height: 1.6; white-space: normal; }
      .source-box {
        max-height: 430px; overflow-y: auto; white-space: pre-wrap;
        background: #0f172a; color: #e2e8f0; border-radius: 10px;
        padding: 0.9rem; font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        font-size: 0.84rem; line-height: 1.5;
      }
      .status-complete { color: #15803d; font-weight: 700; }
      .status-pending { color: #b45309; font-weight: 700; }
      div[data-testid="stForm"] { background: white; border-radius: 14px; padding: 0.4rem 1rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Calibration humaine du RAG")
st.caption("Évalue un exemple à la fois. Les identités réelles de A et B restent volontairement masquées.")

calibration_files = find_calibration_files()
if not calibration_files:
    st.error(
        "Aucun fichier human_calibration_blind.csv trouvé. "
        "Exécute d’abord le notebook 11 jusqu’à la section Calibration humaine."
    )
    st.stop()

file_labels = {
    f"{path.parent.name} — {datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}": path
    for path in calibration_files
}
selected_label = st.sidebar.selectbox(
    "Expérience",
    list(file_labels),
    index=0,
    help="L’expérience la plus récente est sélectionnée automatiquement.",
)
csv_path = file_labels[selected_label]

try:
    dataframe, delimiter = load_annotations(csv_path)
except Exception as error:
    st.error(f"Impossible de lire le fichier : {error}")
    st.stop()

if dataframe.attrs.get("recovered_excel_header", False):
    st.warning(
        "Excel avait déplacé les vrais en-têtes dans la première ligne. "
        "L’application les a restaurés sans perdre de données. Le fichier sera "
        "remis définitivement au bon format lors de ton premier enregistrement."
    )

if dataframe.empty:
    st.error("Le fichier de calibration ne contient aucune ligne.")
    st.stop()

total_rows = len(dataframe)
if "row_index" not in st.session_state:
    first_incomplete = next(
        (index for index, (_, row) in enumerate(dataframe.iterrows()) if not row_is_complete(row)),
        0,
    )
    st.session_state["row_index"] = first_incomplete
st.session_state["row_index"] = min(st.session_state["row_index"], total_rows - 1)
row_index = int(st.session_state["row_index"])

complete_mask = dataframe.apply(row_is_complete, axis=1)
completed_count = int(complete_mask.sum())
progress = completed_count / total_rows

st.sidebar.metric("Progression", f"{completed_count} / {total_rows}")
st.sidebar.progress(progress)
st.sidebar.caption(f"{progress:.0%} terminé")
st.sidebar.divider()

jump_options = list(range(total_rows))
selected_jump = st.sidebar.selectbox(
    "Aller à un exemple",
    jump_options,
    index=row_index,
    format_func=lambda index: (
        f"{'✓' if complete_mask.iloc[index] else '○'} "
        f"{index + 1:02d}/{total_rows} · {dataframe.iloc[index]['domain'].upper()} · "
        f"{dataframe.iloc[index]['current_question'][:48]}"
    ),
)
if selected_jump != row_index:
    set_row_index(selected_jump, total_rows)

st.sidebar.caption(f"Fichier : {csv_path.parent.name}/human_calibration_blind.csv")
st.sidebar.caption("Sauvegarde automatique uniquement après validation du formulaire.")

row = dataframe.iloc[row_index]
is_complete = row_is_complete(row)

top_left, top_middle, top_right = st.columns([1, 2, 1])
with top_left:
    if st.button("← Précédent", use_container_width=True, disabled=row_index == 0):
        set_row_index(row_index - 1, total_rows)
with top_middle:
    if st.button("Prochain exemple non terminé", use_container_width=True, disabled=completed_count == total_rows):
        set_row_index(move_to_next_incomplete(dataframe, row_index), total_rows)
with top_right:
    if st.button("Suivant →", use_container_width=True, disabled=row_index == total_rows - 1):
        set_row_index(row_index + 1, total_rows)

status_text = (
    '<span class="status-complete">✓ Exemple entièrement annoté</span>'
    if is_complete
    else '<span class="status-pending">○ Annotation à terminer</span>'
)
st.markdown(
    f"**Exemple {row_index + 1}/{total_rows}** · Domaine `{html.escape(row['domain'].upper())}` · "
    f"ID `{html.escape(row['example_id'])}` · {status_text}",
    unsafe_allow_html=True,
)

external_review_text = build_external_review_text(row, row_index, total_rows)
st.download_button(
    "⬇️ Télécharger cet exemple pour une seconde évaluation",
    data=external_review_text.encode("utf-8-sig"),
    file_name=f"evaluation_humaine_{row_index + 1:02d}_{row['example_id']}.txt",
    mime="text/plain; charset=utf-8",
    use_container_width=True,
    help="Le fichier ne contient pas tes choix actuels afin que la seconde évaluation reste indépendante.",
)

render_panel("Question actuelle", row["current_question"], "question-panel")

context_tab, reference_tab = st.tabs(["💬 Historique de conversation", "🎯 Référence et preuves attendues"])
with context_tab:
    render_panel("Conversation avant la question", row["conversation_history"])
with reference_tab:
    render_panel("Réponse de référence", row["reference_answer"])
    with st.expander("Afficher les preuves de référence", expanded=True):
        st.markdown(
            f'<div class="source-box">{html.escape(row["gold_evidence"])}</div>',
            unsafe_allow_html=True,
        )

st.subheader("Comparer les deux réponses")
candidate_a_column, candidate_b_column = st.columns(2, gap="large")

with candidate_a_column:
    st.markdown("### Réponse A")
    st.caption(f"Statut de génération : {row['candidate_a_status']}")
    render_panel("Réponse proposée", row["candidate_a_answer"], "candidate-panel")
    with st.expander("Sources citées par A", expanded=False):
        source_text = row["candidate_a_cited_sources"] or "Aucune source citée."
        st.markdown(f'<div class="source-box">{html.escape(source_text)}</div>', unsafe_allow_html=True)

with candidate_b_column:
    st.markdown("### Réponse B")
    st.caption(f"Statut de génération : {row['candidate_b_status']}")
    render_panel("Réponse proposée", row["candidate_b_answer"], "candidate-panel")
    with st.expander("Sources citées par B", expanded=False):
        source_text = row["candidate_b_cited_sources"] or "Aucune source citée."
        st.markdown(f'<div class="source-box">{html.escape(source_text)}</div>', unsafe_allow_html=True)

st.divider()
st.subheader("Ton évaluation")
st.info(
    "Faithfulness = tous les faits importants de la réponse sont soutenus par ses sources citées. "
    "Correctness = exactitude par rapport à la question, à la référence et aux preuves."
)

winner_index = WINNER_VALUES.index(row["human_winner"]) if row["human_winner"] in WINNER_VALUES else None
a_faithfulness_index = (
    FAITHFULNESS_VALUES.index(row["human_a_faithfulness"])
    if row["human_a_faithfulness"] in FAITHFULNESS_VALUES
    else None
)
b_faithfulness_index = (
    FAITHFULNESS_VALUES.index(row["human_b_faithfulness"])
    if row["human_b_faithfulness"] in FAITHFULNESS_VALUES
    else None
)
a_correctness_index = (
    CORRECTNESS_VALUES.index(row["human_a_correctness_0_to_4"]) + 1
    if row["human_a_correctness_0_to_4"] in CORRECTNESS_VALUES
    else 0
)
b_correctness_index = (
    CORRECTNESS_VALUES.index(row["human_b_correctness_0_to_4"]) + 1
    if row["human_b_correctness_0_to_4"] in CORRECTNESS_VALUES
    else 0
)

with st.form(f"annotation_form::{row['example_id']}"):
    winner = st.radio(
        "1. Quelle est la meilleure réponse dans l’ensemble ?",
        WINNER_VALUES,
        index=winner_index,
        horizontal=True,
        format_func=lambda value: {"A": "Réponse A", "B": "Réponse B", "tie": "Égalité"}[value],
    )

    faith_a_column, faith_b_column = st.columns(2, gap="large")
    with faith_a_column:
        a_faithfulness = st.radio(
            "2. Les faits de A sont-ils soutenus par les sources citées ?",
            FAITHFULNESS_VALUES,
            index=a_faithfulness_index,
            horizontal=True,
            format_func=lambda value: "Oui — pass" if value == "pass" else "Non — fail",
        )
    with faith_b_column:
        b_faithfulness = st.radio(
            "3. Les faits de B sont-ils soutenus par les sources citées ?",
            FAITHFULNESS_VALUES,
            index=b_faithfulness_index,
            horizontal=True,
            format_func=lambda value: "Oui — pass" if value == "pass" else "Non — fail",
        )

    correctness_help = (
        "4 = entièrement correcte · 3 = correcte avec omission mineure · "
        "2 = partiellement correcte / omission importante · 1 = surtout incorrecte · 0 = inutilisable"
    )
    score_a_column, score_b_column = st.columns(2, gap="large")
    with score_a_column:
        a_correctness = st.selectbox(
            "4. Note de correctness pour A",
            ("Non noté",) + CORRECTNESS_VALUES,
            index=a_correctness_index,
            help=correctness_help,
        )
    with score_b_column:
        b_correctness = st.selectbox(
            "5. Note de correctness pour B",
            ("Non noté",) + CORRECTNESS_VALUES,
            index=b_correctness_index,
            help=correctness_help,
        )

    notes = st.text_area(
        "Notes facultatives",
        value=row["human_notes"],
        placeholder="Exemple : B répond mieux, mais sa deuxième citation ne soutient pas entièrement la phrase finale.",
        height=100,
    )

    save_column, save_next_column = st.columns(2)
    with save_column:
        save_only = st.form_submit_button("Enregistrer", use_container_width=True)
    with save_next_column:
        save_and_next = st.form_submit_button("Enregistrer et suivant →", type="primary", use_container_width=True)

if save_only or save_and_next:
    missing_labels = []
    if winner is None:
        missing_labels.append("le gagnant")
    if a_faithfulness is None:
        missing_labels.append("la faithfulness de A")
    if b_faithfulness is None:
        missing_labels.append("la faithfulness de B")
    if a_correctness == "Non noté":
        missing_labels.append("la correctness de A")
    if b_correctness == "Non noté":
        missing_labels.append("la correctness de B")

    if missing_labels:
        st.error("Complète avant d’enregistrer : " + ", ".join(missing_labels) + ".")
    else:
        try:
            latest_dataframe, latest_delimiter = load_annotations(csv_path)
            matching_indices = latest_dataframe.index[latest_dataframe["example_id"] == row["example_id"]].tolist()
            if len(matching_indices) != 1:
                raise ValueError("L’exemple courant est introuvable ou dupliqué dans le fichier.")
            target_index = matching_indices[0]
            latest_dataframe.loc[target_index, "human_winner"] = clean(winner)
            latest_dataframe.loc[target_index, "human_a_faithfulness"] = clean(a_faithfulness)
            latest_dataframe.loc[target_index, "human_b_faithfulness"] = clean(b_faithfulness)
            latest_dataframe.loc[target_index, "human_a_correctness_0_to_4"] = clean(a_correctness)
            latest_dataframe.loc[target_index, "human_b_correctness_0_to_4"] = clean(b_correctness)
            latest_dataframe.loc[target_index, "human_notes"] = clean(notes)
            backup_path = atomic_save(latest_dataframe, csv_path, latest_delimiter)
        except PermissionError:
            st.error("Impossible d’enregistrer : ferme le fichier CSV dans Excel, puis réessaie.")
        except Exception as error:
            st.error(f"Sauvegarde impossible : {error}")
        else:
            st.toast("Annotation enregistrée", icon="✅")
            st.session_state["last_backup"] = str(backup_path)
            if save_and_next:
                st.session_state["row_index"] = move_to_next_incomplete(latest_dataframe, row_index)
            st.rerun()

if completed_count == total_rows:
    st.success(
        "Les 40 exemples sont entièrement annotés. Ferme l’application puis relance le notebook 11 avec Run All."
    )
