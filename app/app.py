
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

from remote_live_pipeline import create_encounter, list_live_encounters, load_encounter_metadata, update_encounter_metadata, transcribe_encounter



from remote_clinical_pipeline import (
    extract_clinical_facts,
    generate_soap,
    load_live_facts,
    load_live_soap,
    save_reviewed_facts,
    save_reviewed_soap,
    approve_live_record
)

st.set_page_config(
    page_title="Medical Consultation AI",
    page_icon="🩺",
    layout="wide"
)

st.markdown('\n<style>\n\n/* ---------------------------------------------------------\n   LIVE APP ACCESSIBILITY FIXES\n   --------------------------------------------------------- */\n\n/* Disabled/read-only transcript must remain readable. */\n[data-testid="stTextArea"] textarea:disabled {\n    color: #1A2420 !important;\n    -webkit-text-fill-color: #1A2420 !important;\n    opacity: 1 !important;\n    background-color: #FFFFFF !important;\n}\n\n/* Editable textarea. */\n[data-testid="stTextArea"] textarea {\n    color: #1A2420 !important;\n    -webkit-text-fill-color: #1A2420 !important;\n}\n\n/* Primary action buttons. */\n[data-testid="stBaseButton-primary"] {\n    color: #FFFFFF !important;\n}\n\n[data-testid="stBaseButton-primary"] p,\n[data-testid="stBaseButton-primary"] span {\n    color: #FFFFFF !important;\n}\n\n/* Sidebar buttons use clear contrast. */\n[data-testid="stSidebar"] button[kind="primary"] {\n    color: #FFFFFF !important;\n}\n\n[data-testid="stSidebar"] button[kind="primary"] p,\n[data-testid="stSidebar"] button[kind="primary"] span {\n    color: #FFFFFF !important;\n}\n\n</style>\n', unsafe_allow_html=True)





# ============================================================
# DOWNLOAD BUTTON VISUAL FIX
# ============================================================

st.markdown(
    """
    <style>

    /* MEDICAL_AI_DOWNLOAD_BUTTON_STYLE */

    [data-testid="stDownloadButton"] button {
        background-color: #1F5D4E !important;
        border: 1px solid #1F5D4E !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }

    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span {
        color: #FFFFFF !important;
    }

    [data-testid="stDownloadButton"] button svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        background-color: #184C40 !important;
        border-color: #184C40 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stDownloadButton"] button:hover p,
    [data-testid="stDownloadButton"] button:hover span {
        color: #FFFFFF !important;
    }

    [data-testid="stDownloadButton"] button:focus,
    [data-testid="stDownloadButton"] button:active {
        background-color: #184C40 !important;
        border-color: #184C40 !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(31, 93, 78, 0.20) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# MEDICAL_CONSULTATION_AI_DESIGN_SYSTEM_V1

st.markdown(
    """
    <style>

    :root {
        --ink: #1A2420;
        --muted: #5C6B63;
        --green: #1F5D4E;
        --green-dark: #184C40;
        --green-soft: #E1F0EC;
        --panel: #F7F7F5;
        --border: #DDE3DF;
        --amber: #7A4E0A;
        --amber-soft: #FBEFD9;
        --red: #9C3B34;
        --red-soft: #FBEAE8;
    }

    /* APP */

    html {
        color-scheme: light;
    }

    body,
    .stApp,
    [data-testid="stAppViewContainer"] {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        font-family:
            "IBM Plex Sans",
            "Avenir Next",
            "Helvetica Neue",
            Arial,
            sans-serif !important;
    }

    .block-container {
        max-width: 1480px !important;
        padding-top: 2.5rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        padding-bottom: 4rem !important;
    }

    /* HEADER */

    header[data-testid="stHeader"] {
        background: rgba(255,255,255,0.97) !important;
        border-bottom: 1px solid #EDF0EE;
    }

    /* TYPOGRAPHY */

    h1, h2, h3, h4 {
        color: var(--ink) !important;
        font-family:
            "IBM Plex Sans",
            "Avenir Next",
            "Helvetica Neue",
            Arial,
            sans-serif !important;
        letter-spacing: -0.025em !important;
    }

    h1 {
        font-size: 2.35rem !important;
        font-weight: 600 !important;
        line-height: 1.1 !important;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 600 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }

    p {
        color: var(--ink);
        line-height: 1.55;
    }

    .clinical-note-text {
        font-family:
            "IBM Plex Serif",
            Georgia,
            "Times New Roman",
            serif !important;
        font-size: 1.02rem;
        line-height: 1.65;
        color: var(--ink);
    }

    /* SIDEBAR */

    section[data-testid="stSidebar"] {
        background: var(--panel) !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: var(--muted) !important;
    }

    /* INPUTS */

    [data-baseweb="select"] > div,
    input,
    textarea {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border-color: #C9D2CD !important;
        border-radius: 7px !important;
        box-shadow: none !important;
    }

    textarea:focus,
    input:focus {
        border-color: var(--green) !important;
        box-shadow: 0 0 0 1px var(--green) !important;
    }

    /* METRICS */

    [data-testid="stMetric"] {
        background: transparent !important;
        border: none !important;
        padding: 0.2rem 0 1rem 0 !important;
        box-shadow: none !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--ink) !important;
        font-size: 1.6rem !important;
        font-weight: 500 !important;
    }

    /* TABS */

    [data-baseweb="tab-list"] {
        gap: 1.6rem !important;
        border-bottom: 1px solid var(--border) !important;
    }

    [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--muted) !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        border-radius: 0 !important;
        font-weight: 500 !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: var(--green) !important;
        font-weight: 600 !important;
    }

    [data-baseweb="tab-highlight"] {
        background-color: var(--green) !important;
        height: 2px !important;
    }

    /* BUTTONS */

    .stButton > button {
        min-height: 42px !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--green) !important;
        color: white !important;
        border: 1px solid var(--green) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--green-dark) !important;
        border-color: var(--green-dark) !important;
    }

    .stButton > button[kind="secondary"] {
        background: #FFFFFF !important;
        color: var(--ink) !important;
        border: 1px solid #C7D0CB !important;
    }

    /* TABLES */

    [data-testid="stDataFrame"],
    [data-testid="stDataEditor"] {
        border: 1px solid var(--border) !important;
        border-radius: 7px !important;
        overflow: hidden;
        box-shadow: none !important;
    }

    /* ALERTS */

    [data-testid="stAlert"] {
        border-radius: 7px !important;
        box-shadow: none !important;
    }

    /* DIVIDERS */

    hr {
        border: none !important;
        border-top: 1px solid var(--border) !important;
    }

    /* CAPTIONS */

    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div style="color:#5C6B63;'
    'font-size:0.9rem;'
    'font-weight:500;'
    'margin-bottom:0.4rem;">'
    'Clinical documentation workspace'
    '</div>',
    unsafe_allow_html=True
)



# ============================================================
# PROJECT
# ============================================================

# Portable project root.
# app.py is located inside PROJECT/app/,
# therefore parent.parent resolves to the project folder.
PROJECT = Path(__file__).resolve().parent.parent


CONSULTATIONS = [
    "day1_consultation01",
    "day1_consultation07",
    "day2_consultation01",
    "day2_consultation05",
    "day3_consultation06",
    "day3_consultation09",
    "day4_consultation03",
    "day4_consultation10",
    "day5_consultation03",
    "day5_consultation12"
]


# ============================================================
# LOADERS
# ============================================================

def load_json(path):

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_text(path):

    if not path.exists():
        return None

    return path.read_text(
        encoding="utf-8"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "Medical Consultation AI"
    )

    st.caption(
        "Clinical Documentation Prototype"
    )

    st.divider()

    # ========================================================
    # NEW LIVE CONSULTATION
    # ========================================================

    st.subheader(
        "New Consultation"
    )

    st.caption(
        "Upload a doctor-patient consultation "
        "to begin the AI documentation workflow."
    )

    uploaded_consultation = st.file_uploader(
        "Upload consultation audio",
        type=[
            "wav",
            "mp3",
            "m4a"
        ],
        key="live_consultation_uploader",
        help=(
            "Supported formats: WAV, MP3 and M4A. "
            "The uploaded file is stored as a new encounter "
            "before AI processing begins."
        )
    )

    if uploaded_consultation is not None:

        st.audio(
            uploaded_consultation
        )

        st.caption(
            f"Selected: {uploaded_consultation.name}"
        )

        if st.button(
            "Create Consultation",
            type="primary",
            use_container_width=True,
            key="create_live_consultation"
        ):

            try:

                new_encounter = create_encounter(
                    uploaded_consultation
                )

                st.session_state[
                    "active_live_encounter"
                ] = new_encounter[
                    "encounter_id"
                ]

                st.success(
                    "Consultation created successfully."
                )

                st.caption(
                    "Encounter ID: "
                    + new_encounter[
                        "encounter_id"
                    ]
                )

            except Exception as exc:

                st.error(
                    "Could not create consultation: "
                    + str(exc)
                )


    # ========================================================
    # RECORD CONSULTATION
    # ========================================================

    st.markdown(
        "**Or record the consultation**"
    )

    st.caption(
        "Record doctor-patient audio directly "
        "from the browser microphone."
    )

    recorded_consultation = st.audio_input(
        "Record consultation",
        key="live_consultation_recorder"
    )

    if recorded_consultation is not None:

        st.audio(
            recorded_consultation
        )

        if st.button(
            "Create Consultation from Recording",
            use_container_width=True,
            key="create_recorded_consultation"
        ):

            try:

                recorded_encounter = create_encounter(
                    recorded_consultation
                )

                st.session_state[
                    "active_live_encounter"
                ] = recorded_encounter[
                    "encounter_id"
                ]

                st.success(
                    "Recorded consultation created."
                )

                st.rerun()

            except Exception as exc:

                st.error(
                    "Could not save recording: "
                    + str(exc)
                )


    st.divider()



    # ========================================================
    # ACTIVE LIVE ENCOUNTER
    # ========================================================

    active_live_encounter = (
        st.session_state.get(
            "active_live_encounter"
        )
    )

    if active_live_encounter:

        active_metadata = (
            load_encounter_metadata(
                active_live_encounter
            )
        )

        if active_metadata:

            st.markdown(
                "#### Active Consultation"
            )

            st.caption(
                active_live_encounter
            )

            asr_status = (
                active_metadata
                .get(
                    "pipeline",
                    {}
                )
                .get(
                    "asr",
                    "pending"
                )
            )

            st.write(
                "Transcription status:",
                asr_status.capitalize()
            )


            if asr_status != "complete":

                if st.button(
                    "Transcribe with Whisper Large-v3",
                    type="primary",
                    use_container_width=True,
                    key="run_live_whisper"
                ):

                    try:

                        with st.spinner(
                            "Whisper Large-v3 is transcribing "
                            "the consultation..."
                        ):

                            live_asr_result = (
                                transcribe_encounter(
                                    active_live_encounter
                                )
                            )

                        st.session_state[
                            "live_transcript"
                        ] = live_asr_result[
                            "transcript"
                        ]

                        st.success(
                            "Transcription complete."
                        )

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            "Transcription failed: "
                            + str(exc)
                        )


            else:

                transcript_relative = (
                    active_metadata.get(
                        "transcript"
                    )
                )

                if transcript_relative:

                    live_transcript_path = (
                        PROJECT
                        / transcript_relative
                    )

                    if (
                        live_transcript_path.exists()
                    ):

                        live_transcript_text = (
                            live_transcript_path
                            .read_text(
                                encoding="utf-8"
                            )
                        )

                        st.success(
                            "Whisper transcription complete."
                        )

                        with st.expander(
                            "Preview transcript"
                        ):

                            st.write(
                                live_transcript_text
                            )



    existing_live_encounters = (
        list_live_encounters()
    )

    if existing_live_encounters:

        st.markdown(
            "#### Saved Consultations"
        )

        st.caption(
            f"{len(existing_live_encounters)} "
            "live consultation(s) saved"
        )

        saved_live_ids = [
            item["encounter_id"]
            for item
            in existing_live_encounters
            if item.get("encounter_id")
        ]

        if saved_live_ids:

            current_active = (
                st.session_state.get(
                    "active_live_encounter"
                )
            )

            default_index = 0

            if (
                current_active
                in saved_live_ids
            ):

                default_index = (
                    saved_live_ids.index(
                        current_active
                    )
                )

            selected_saved_live = (
                st.selectbox(
                    "Open saved consultation",
                    saved_live_ids,
                    index=default_index,
                    key="saved_live_encounter_selector"
                )
            )

            if st.button(
                "Open Consultation",
                use_container_width=True,
                key="open_saved_live_consultation"
            ):

                st.session_state[
                    "active_live_encounter"
                ] = selected_saved_live

                st.rerun()


    # LEGACY EVALUATION SIDEBAR HIDDEN
    # The historical pilot/evaluation interface is retained
    # later in this file for research reproducibility, but it
    # is not displayed in the professor-facing product.

    consultation_id = (
        CONSULTATIONS[0]
        if CONSULTATIONS
        else None
    )


# ============================================================
# FILE PATHS
# ============================================================

# ============================================================
# LIVE CONSULTATION WORKSPACE
# ============================================================

active_live_id = st.session_state.get(
    "active_live_encounter"
)

if active_live_id:

    live_metadata = load_encounter_metadata(
        active_live_id
    )

    if live_metadata:

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        st.title(
            "Medical Consultation AI"
        )

        st.caption(
            "AI-assisted clinical documentation workspace"
        )

        st.warning(
            "AI-generated medical documentation is a draft. "
            "A clinician must review and approve the record "
            "before clinical use."
        )


        pipeline_status = (
            live_metadata.get(
                "pipeline",
                {}
            )
        )

        asr_status = pipeline_status.get(
            "asr",
            "pending"
        )

        fact_status = pipeline_status.get(
            "clinical_fact_extraction",
            "pending"
        )

        soap_status = pipeline_status.get(
            "soap_generation",
            "pending"
        )

        approval_status = pipeline_status.get(
            "approval",
            "pending"
        )


        # ----------------------------------------------------
        # Status cards
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(
            4
        )

        with c1:

            st.metric(
                "Consultation",
                "Live"
            )

        with c2:

            st.metric(
                "Transcription",
                asr_status.capitalize()
            )

        with c3:

            st.metric(
                "Clinical Facts",
                fact_status.capitalize()
            )

        with c4:

            st.metric(
                "Review Status",
                (
                    "Approved"
                    if approval_status == "complete"
                    else "Draft"
                )
            )


        st.caption(
            "Encounter ID: "
            + active_live_id
        )

        st.divider()


        # ----------------------------------------------------
        # Resolve paths
        # ----------------------------------------------------

        original_audio_path = None

        stored_audio_relative = (
            live_metadata.get(
                "stored_audio"
            )
        )

        if stored_audio_relative:

            candidate = (
                PROJECT
                / stored_audio_relative
            )

            if candidate.exists():

                original_audio_path = candidate


        transcript_path_live = None
        live_transcript = None

        transcript_relative = (
            live_metadata.get(
                "transcript"
            )
        )

        if transcript_relative:

            candidate = (
                PROJECT
                / transcript_relative
            )

            if candidate.exists():

                transcript_path_live = candidate

                live_transcript = (
                    candidate.read_text(
                        encoding="utf-8"
                    )
                )


        # ----------------------------------------------------
        # Main workflow tabs
        # ----------------------------------------------------

        (
            live_audio_tab,
            live_transcript_tab,
            live_facts_tab,
            live_soap_tab,
            live_review_tab
        ) = st.tabs(
            [
                "1. Audio",
                "2. Transcript",
                "3. Clinical Facts",
                "4. SOAP Note",
                "5. Review & Approval"
            ]
        )


        # ====================================================
        # AUDIO
        # ====================================================

        with live_audio_tab:

            st.subheader(
                "Consultation Audio"
            )

            if original_audio_path:

                st.audio(
                    str(
                        original_audio_path
                    )
                )

                st.caption(
                    "Original file: "
                    + live_metadata.get(
                        "original_filename",
                        original_audio_path.name
                    )
                )

            else:

                st.error(
                    "Original consultation audio "
                    "could not be located."
                )


            if asr_status == "complete":

                st.success(
                    "Whisper Large-v3 transcription completed."
                )

            else:

                st.info(
                    "Use the Transcribe with Whisper "
                    "Large-v3 button in the sidebar "
                    "to process this consultation."
                )


        # ====================================================
        # TRANSCRIPT
        # ====================================================

        with live_transcript_tab:

            st.subheader(
                "Whisper Large-v3 Transcript"
            )

            st.caption(
                "Original AI-generated transcription"
            )

            if live_transcript:

                st.text_area(
                    "Original transcript",
                    value=live_transcript,
                    height=500,
                    disabled=True,
                    key=(
                        "live_original_transcript_"
                        + active_live_id
                    )
                )

            else:

                st.info(
                    "No transcript has been generated yet."
                )


        
        # ====================================================
        # CLINICAL FACTS
        # ====================================================

        with live_facts_tab:

            st.subheader(
                "Clinical Facts"
            )

            current_fact_status = (
                load_encounter_metadata(
                    active_live_id
                )
                .get(
                    "pipeline",
                    {}
                )
                .get(
                    "clinical_fact_extraction",
                    "pending"
                )
            )


            if not live_transcript:

                st.info(
                    "Complete the Whisper transcription first."
                )


            elif (
                current_fact_status
                != "complete"
            ):

                st.write(
                    "The selected Qwen3-8B model will extract "
                    "clinically relevant information from the "
                    "consultation transcript."
                )

                if st.button(
                    "Extract Clinical Facts with Qwen3-8B",
                    type="primary",
                    key=(
                        "extract_live_facts_"
                        + active_live_id
                    )
                ):

                    try:

                        with st.spinner(
                            "Qwen3-8B is extracting and validating "
                            "clinical facts. The first run also "
                            "downloads the model..."
                        ):

                            extract_clinical_facts(
                                active_live_id
                            )

                        st.success(
                            "Clinical fact extraction complete."
                        )

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            "Clinical fact extraction failed: "
                            + str(exc)
                        )


            else:

                fact_document = (
                    load_live_facts(
                        active_live_id,
                        prefer_reviewed=True
                    )
                )

                fact_items = (
                    fact_document.get(
                        "clinical_facts",
                        []
                    )
                    if fact_document
                    else []
                )


                if not fact_items:

                    st.warning(
                        "Qwen returned no evidence-grounded "
                        "clinical facts for this consultation."
                    )

                else:

                    st.caption(
                        "Edit, add or remove facts before SOAP "
                        "generation. The original AI extraction "
                        "remains stored separately."
                    )

                    fact_df = pd.DataFrame(
                        fact_items
                    )

                    expected_columns = [
                        "fact_id",
                        "category",
                        "status",
                        "importance",
                        "fact",
                        "evidence_quote"
                    ]

                    for column in expected_columns:

                        if column not in fact_df.columns:
                            fact_df[column] = ""


                    fact_df = fact_df[
                        expected_columns
                    ]


                    edited_fact_df = (
                        st.data_editor(
                            fact_df,
                            num_rows="dynamic",
                            hide_index=True,
                            disabled=[
                                "fact_id"
                            ],
                            use_container_width=True,
                            key=(
                                "live_fact_editor_"
                                + active_live_id
                            )
                        )
                    )


                    if st.button(
                        "Save Clinical Fact Review",
                        type="primary",
                        key=(
                            "save_live_facts_"
                            + active_live_id
                        )
                    ):

                        records = (
                            edited_fact_df
                            .fillna("")
                            .to_dict(
                                orient="records"
                            )
                        )

                        save_reviewed_facts(
                            active_live_id,
                            records
                        )

                        st.success(
                            "Clinical fact corrections saved."
                        )

                        st.rerun()


        # ====================================================
        # SOAP
        # ====================================================

        with live_soap_tab:

            st.subheader(
                "SOAP Note"
            )

            st.caption(
                "Qwen3-8B draft with deterministic "
                "source-fact validation"
            )


            refreshed_metadata = (
                load_encounter_metadata(
                    active_live_id
                )
            )

            current_fact_status = (
                refreshed_metadata
                .get(
                    "pipeline",
                    {}
                )
                .get(
                    "clinical_fact_extraction",
                    "pending"
                )
            )

            current_soap_status = (
                refreshed_metadata
                .get(
                    "pipeline",
                    {}
                )
                .get(
                    "soap_generation",
                    "pending"
                )
            )


            if (
                current_fact_status
                != "complete"
            ):

                st.info(
                    "Extract and review Clinical Facts first."
                )


            elif (
                current_soap_status
                != "complete"
            ):

                st.write(
                    "SOAP generation uses only validated "
                    "clinical facts. Unsupported examination "
                    "findings, diagnoses or treatment must not "
                    "be introduced."
                )

                if st.button(
                    "Generate SOAP Draft with Qwen3-8B",
                    type="primary",
                    key=(
                        "generate_live_soap_"
                        + active_live_id
                    )
                ):

                    try:

                        with st.spinner(
                            "Generating and validating the SOAP draft..."
                        ):

                            generate_soap(
                                active_live_id
                            )

                        st.success(
                            "SOAP draft generated."
                        )

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            "SOAP generation failed: "
                            + str(exc)
                        )


            else:

                ai_soap_path = (
                    PROJECT
                    / "results"
                    / "live_encounters"
                    / active_live_id
                    / "soap"
                    / "ai_soap_draft.json"
                )

                if ai_soap_path.exists():

                    ai_soap = json.loads(
                        ai_soap_path.read_text(
                            encoding="utf-8"
                        )
                    )

                else:

                    ai_soap = (
                        load_live_soap(
                            active_live_id,
                            prefer_reviewed=False
                        )
                        or {}
                    )


                working_soap = (
                    load_live_soap(
                        active_live_id,
                        prefer_reviewed=True
                    )
                    or ai_soap
                )


                st.info(
                    "The AI draft is preserved separately. "
                    "Edit the working SOAP sections below. "
                    "Moving text between sections is allowed."
                )


                ai_sections = (
                    ai_soap.get(
                        "soap",
                        {}
                    )
                )

                working_sections = (
                    working_soap.get(
                        "soap",
                        {}
                    )
                )


                with st.expander(
                    "View original AI SOAP draft"
                ):

                    for section in [
                        "subjective",
                        "objective",
                        "assessment",
                        "plan"
                    ]:

                        st.markdown(
                            f"**{section.title()}**"
                        )

                        items = (
                            ai_sections.get(
                                section,
                                []
                            )
                        )

                        if items:

                            for item in items:

                                if isinstance(
                                    item,
                                    dict
                                ):

                                    st.write(
                                        "• "
                                        + str(
                                            item.get(
                                                "statement",
                                                ""
                                            )
                                        )
                                    )

                                else:

                                    st.write(
                                        "• "
                                        + str(
                                            item
                                        )
                                    )

                        else:

                            st.caption(
                                "Not documented."
                            )


                # ====================================================
                # SOAP_MOVE_BETWEEN_SECTIONS_CONTROL
                # Explicit HITL control required by the project specification.
                # The original AI SOAP draft remains unchanged.
                # The moved working draft is persisted when the reviewer
                # clicks "Save SOAP Review".
                # ====================================================

                st.markdown(
                    "#### Move content between SOAP sections"
                )

                st.caption(
                    "Move one statement from one SOAP section to another. "
                    "After moving it, click **Save SOAP Review** to persist "
                    "the change and include it in the review history."
                )


                soap_section_names = [
                    "subjective",
                    "objective",
                    "assessment",
                    "plan"
                ]


                def _current_soap_editor_text(section):

                    editor_key = (
                        "live_soap_"
                        + section
                        + "_"
                        + active_live_id
                    )

                    # If this section has already been displayed/edited,
                    # use the current reviewer text.
                    if editor_key in st.session_state:

                        return str(
                            st.session_state.get(
                                editor_key,
                                ""
                            )
                            or ""
                        )


                    # Otherwise initialise from the current working SOAP.
                    section_items = (
                        working_sections.get(
                            section,
                            []
                        )
                    )

                    section_lines = []

                    for section_item in section_items:

                        if isinstance(
                            section_item,
                            dict
                        ):

                            statement = str(
                                section_item.get(
                                    "statement",
                                    ""
                                )
                            ).strip()

                        else:

                            statement = str(
                                section_item
                            ).strip()


                        if statement:

                            section_lines.append(
                                statement
                            )


                    return "\n".join(
                        section_lines
                    )


                move_notice_key = (
                    "soap_move_notice_"
                    + active_live_id
                )

                if move_notice_key in st.session_state:

                    st.success(
                        st.session_state.pop(
                            move_notice_key
                        )
                    )


                move_col_1, move_col_2 = (
                    st.columns(2)
                )


                with move_col_1:

                    move_source = (
                        st.selectbox(
                            "Move from",
                            options=soap_section_names,
                            format_func=lambda x: x.title(),
                            key=(
                                "soap_move_source_"
                                + active_live_id
                            )
                        )
                    )


                with move_col_2:

                    move_targets = [
                        section
                        for section in soap_section_names
                        if section != move_source
                    ]

                    move_target = (
                        st.selectbox(
                            "Move to",
                            options=move_targets,
                            format_func=lambda x: x.title(),
                            key=(
                                "soap_move_target_"
                                + active_live_id
                            )
                        )
                    )


                source_text = (
                    _current_soap_editor_text(
                        move_source
                    )
                )

                source_lines = [
                    line.strip()
                    for line in source_text.splitlines()
                    if line.strip()
                ]


                if source_lines:

                    move_index = (
                        st.selectbox(
                            "Statement to move",
                            options=list(
                                range(
                                    len(
                                        source_lines
                                    )
                                )
                            ),
                            format_func=(
                                lambda index:
                                    source_lines[index]
                            ),
                            key=(
                                "soap_move_statement_"
                                + active_live_id
                                + "_"
                                + move_source
                            )
                        )
                    )

                else:

                    move_index = None

                    st.caption(
                        move_source.title()
                        + " currently contains no statements to move."
                    )


                move_clicked = (
                    st.button(
                        "Move Statement",
                        disabled=(
                            move_index is None
                        ),
                        key=(
                            "move_soap_statement_"
                            + active_live_id
                        )
                    )
                )


                if (
                    move_clicked
                    and
                    move_index is not None
                ):

                    source_key = (
                        "live_soap_"
                        + move_source
                        + "_"
                        + active_live_id
                    )

                    target_key = (
                        "live_soap_"
                        + move_target
                        + "_"
                        + active_live_id
                    )


                    current_source_lines = [
                        line.strip()
                        for line in (
                            _current_soap_editor_text(
                                move_source
                            )
                        ).splitlines()
                        if line.strip()
                    ]


                    if (
                        move_index
                        >= len(
                            current_source_lines
                        )
                    ):

                        st.error(
                            "The selected SOAP statement changed. "
                            "Please select it again."
                        )

                    else:

                        statement_to_move = (
                            current_source_lines.pop(
                                move_index
                            )
                        )

                        current_target_lines = [
                            line.strip()
                            for line in (
                                _current_soap_editor_text(
                                    move_target
                                )
                            ).splitlines()
                            if line.strip()
                        ]


                        current_target_lines.append(
                            statement_to_move
                        )


                        # These widget keys have not yet been instantiated
                        # during this script run, so they can safely be
                        # updated before the text areas below are rendered.
                        st.session_state[
                            source_key
                        ] = "\n".join(
                            current_source_lines
                        )

                        st.session_state[
                            target_key
                        ] = "\n".join(
                            current_target_lines
                        )


                        st.session_state[
                            move_notice_key
                        ] = (
                            'Moved "'
                            + statement_to_move
                            + '" from '
                            + move_source.title()
                            + " to "
                            + move_target.title()
                            + "."
                        )


                        st.rerun()


                st.divider()


                edited_sections = {}

                for section in [
                    "subjective",
                    "objective",
                    "assessment",
                    "plan"
                ]:

                    items = (
                        working_sections.get(
                            section,
                            []
                        )
                    )


                    current_lines = []

                    for item in items:

                        if isinstance(
                            item,
                            dict
                        ):

                            current_lines.append(
                                str(
                                    item.get(
                                        "statement",
                                        ""
                                    )
                                )
                            )

                        else:

                            current_lines.append(
                                str(
                                    item
                                )
                            )


                    value = "\n".join(
                        line
                        for line
                        in current_lines
                        if line.strip()
                    )


                    edited_text = (
                        st.text_area(
                            section.title(),
                            value=value,
                            height=180,
                            key=(
                                "live_soap_"
                                + section
                                + "_"
                                + active_live_id
                            )
                        )
                    )


                    edited_sections[
                        section
                    ] = [
                        line.strip()
                        for line
                        in edited_text.splitlines()
                        if line.strip()
                    ]


                if st.button(
                    "Save SOAP Review",
                    type="primary",
                    key=(
                        "save_live_soap_"
                        + active_live_id
                    )
                ):

                    save_reviewed_soap(
                        active_live_id,
                        edited_sections
                    )

                    st.success(
                        "Reviewed SOAP note saved. "
                        "The AI draft remains unchanged."
                    )

                    st.rerun()


        # ====================================================
        # HUMAN REVIEW & APPROVAL
        # ====================================================

        with live_review_tab:

            st.subheader(
                "Review & Approval"
            )

            st.info(
                "The final record cannot be approved until "
                "the transcript, clinical facts and SOAP note "
                "have each been reviewed."
            )


            if live_transcript:

                working_path = (
                    PROJECT
                    / "results"
                    / "live_encounters"
                    / active_live_id
                    / "review"
                    / "working_transcript.txt"
                )


                if working_path.exists():

                    initial_working = (
                        working_path.read_text(
                            encoding="utf-8"
                        )
                    )

                else:

                    initial_working = (
                        live_transcript
                    )


                review_col1, review_col2 = (
                    st.columns(
                        2
                    )
                )


                with review_col1:

                    st.markdown(
                        "#### Original Whisper Transcript"
                    )

                    st.text_area(
                        "Original AI transcript",
                        value=live_transcript,
                        height=420,
                        disabled=True,
                        key=(
                            "review_original_"
                            + active_live_id
                        )
                    )


                with review_col2:

                    st.markdown(
                        "#### Reviewer Working Transcript"
                    )

                    reviewed_transcript = (
                        st.text_area(
                            "Editable transcript",
                            value=initial_working,
                            height=420,
                            key=(
                                "review_working_"
                                + active_live_id
                            )
                        )
                    )


                if st.button(
                    "Save Transcript Review",
                    type="primary",
                    key=(
                        "save_review_transcript_"
                        + active_live_id
                    )
                ):

                    working_path.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    previous = (
                        working_path.read_text(
                            encoding="utf-8"
                        )
                        if working_path.exists()
                        else live_transcript
                    )

                    working_path.write_text(
                        reviewed_transcript,
                        encoding="utf-8"
                    )


                    log_path = (
                        working_path.parent
                        / "correction_log.jsonl"
                    )

                    with open(
                        log_path,
                        "a",
                        encoding="utf-8"
                    ) as log_file:

                        log_file.write(
                            json.dumps(
                                {
                                    "timestamp":
                                        datetime.now().isoformat(),

                                    "event_type":
                                        "transcript_review_saved",

                                    "previous":
                                        previous,

                                    "new":
                                        reviewed_transcript
                                },
                                ensure_ascii=False
                            )
                            + "\n"
                        )


                    refreshed_metadata = (
                        load_encounter_metadata(
                            active_live_id
                        )
                    )

                    refreshed_metadata[
                        "reviewed_transcript"
                    ] = str(
                        working_path.relative_to(
                            PROJECT
                        )
                    )

                    refreshed_metadata[
                        "pipeline"
                    ][
                        "human_review"
                    ] = "in_progress"

                    update_encounter_metadata(
                        active_live_id,
                        refreshed_metadata
                    )


                    st.success(
                        "Transcript review saved."
                    )


            st.divider()

            approval_metadata = (
                load_encounter_metadata(
                    active_live_id
                )
            )

            if (
                approval_metadata
                .get(
                    "pipeline",
                    {}
                )
                .get(
                    "approval"
                )
                == "complete"
            ):

                st.success(
                    "This consultation has completed "
                    "the review and approval workflow."
                )

                st.caption(
                    "The approved record can now be exported. "
                    "Prototype output — not for clinical use."
                )


                # ------------------------------------------------
                # APPROVED RECORD EXPORTS
                # ------------------------------------------------

                exports_dir = (
                    PROJECT
                    / "results"
                    / "live_encounters"
                    / active_live_id
                    / "exports"
                )

                pdf_export_path = (
                    exports_dir
                    / "approved_medical_record.pdf"
                )

                json_export_path = (
                    exports_dir
                    / "approved_medical_record.json"
                )


                if (
                    not pdf_export_path.exists()
                    or
                    not json_export_path.exists()
                ):

                    try:

                        from live_export import (
                            export_approved_record
                        )

                        export_approved_record(
                            active_live_id
                        )

                    except Exception as exc:

                        st.error(
                            "Could not generate approved "
                            f"record exports: {exc}"
                        )


                if (
                    pdf_export_path.exists()
                    and
                    json_export_path.exists()
                ):

                    download_pdf_col, download_json_col = (
                        st.columns(2)
                    )


                    with download_pdf_col:

                        st.download_button(
                            label="Download Medical Record PDF",
                            data=pdf_export_path.read_bytes(),
                            file_name=(
                                active_live_id
                                + "_medical_record.pdf"
                            ),
                            mime="application/pdf",
                            use_container_width=True,
                            key=(
                                "download_live_pdf_"
                                + active_live_id
                            )
                        )


                    with download_json_col:

                        st.download_button(
                            label="Download Medical Record JSON",
                            data=json_export_path.read_bytes(),
                            file_name=(
                                active_live_id
                                + "_medical_record.json"
                            ),
                            mime="application/json",
                            use_container_width=True,
                            key=(
                                "download_live_json_"
                                + active_live_id
                            )
                        )


                    st.caption(
                        "PDF: reviewed SOAP note, clinical facts "
                        "with evidence, and reviewed transcript. "
                        "JSON: structured approved record."
                    )

            else:

                st.warning(
                    "Approval confirms that the documentation "
                    "has completed the prototype human-review workflow."
                )

                if st.button(
                    "Approve Final Record",
                    type="primary",
                    key=(
                        "approve_live_record_"
                        + active_live_id
                    )
                ):

                    try:

                        approve_live_record(
                            active_live_id
                        )

                        st.success(
                            "Final record approved."
                        )

                        st.rerun()

                    except Exception as exc:

                        st.error(
                            str(exc)
                        )


# Close live encounter
        # ----------------------------------------------------

        st.divider()

        if st.button(
            "Close Consultation",
            key="close_live_workspace"
        ):

            st.session_state.pop(
                "active_live_encounter",
                None
            )

            st.session_state.pop(
                "live_transcript",
                None
            )

            st.rerun()


        # CRITICAL:
        # Do not render the old pilot application underneath
        # the live consultation workspace.
        st.stop()




transcript_path = (
    PROJECT
    / "results/asr/whisper_large_v3/datalab_pilot"
    / f"{consultation_id}_transcript.txt"
)

facts_path = (
    PROJECT
    / "results/soap/inputs"
    / f"{consultation_id}_soap_input.json"
)

soap_path = (
    PROJECT
    / "results/soap/qwen3_8b/canonical"
    / f"{consultation_id}_canonical_soap.json"
)

audio_path = (
    PROJECT
    / "data/pilot_mixed_audio"
    / f"{consultation_id}_mixed.wav"
)


# ============================================================
# LOAD CURRENT CONSULTATION
# ============================================================

transcript = load_text(
    transcript_path
)

facts_data = load_json(
    facts_path
)

soap_data = load_json(
    soap_path
)


# ============================================================
# HITL TRANSCRIPT REVIEW STATE
# ============================================================

review_dir = (
    PROJECT
    / "results/hitl/app_review_sessions"
    / consultation_id
)

review_dir.mkdir(
    parents=True,
    exist_ok=True
)

working_transcript_path = (
    review_dir
    / "working_transcript.txt"
)

correction_log_path = (
    review_dir
    / "correction_log.json"
)

working_transcript_key = (
    f"working_transcript_{consultation_id}"
)

saved_transcript_key = (
    f"saved_transcript_{consultation_id}"
)


# Load an existing reviewer-edited transcript if available.
# Otherwise begin from the original Whisper transcript.

if working_transcript_key not in st.session_state:

    if working_transcript_path.exists():

        initial_working_transcript = (
            working_transcript_path.read_text(
                encoding="utf-8"
            )
        )

    else:

        initial_working_transcript = (
            transcript or ""
        )

    st.session_state[
        working_transcript_key
    ] = initial_working_transcript


if saved_transcript_key not in st.session_state:

    st.session_state[
        saved_transcript_key
    ] = st.session_state[
        working_transcript_key
    ]


# ============================================================
# HEADER
# ============================================================

# MEDICAL_CONSULTATION_AI_PRODUCT_HOME_V1
# ============================================================
# PRODUCT HOME
# ============================================================

st.title(
    "Medical Consultation AI"
)

st.caption(
    "AI-assisted clinical documentation workspace"
)

st.warning(
    "Research prototype. AI-generated documentation must "
    "be reviewed before approval and is not intended for "
    "independent clinical decision-making."
)

st.markdown(
    "### Start a Consultation"
)

st.write(
    "Upload an existing doctor-patient consultation or record "
    "one directly from the browser using the controls in the "
    "sidebar. You can also reopen a saved consultation."
)

home_col1, home_col2, home_col3 = st.columns(
    3
)

with home_col1:

    st.markdown(
        "**1. Capture**"
    )

    st.caption(
        "Upload or record consultation audio."
    )


with home_col2:

    st.markdown(
        "**2. Generate**"
    )

    st.caption(
        "Whisper transcription, Qwen clinical facts "
        "and structured SOAP documentation."
    )


with home_col3:

    st.markdown(
        "**3. Review & Export**"
    )

    st.caption(
        "Human review, approval, PDF and JSON export."
    )


st.info(
    "To continue, create a new consultation or open one "
    "from Saved Consultations in the sidebar."
)

# Stop before rendering the historical evaluation archive.
st.stop()


# ============================================================
# HISTORICAL EVALUATION INTERFACE
# Retained in source for research reproducibility.
# Not rendered in normal product mode.
# ============================================================

st.title(
    "Medical Consultation AI"
)

st.caption(
    "Automatic transcription, clinical information extraction, "
    "structured SOAP generation and human-in-the-loop review"
)

st.warning(
    "AI-generated medical documentation is a draft and must "
    "be reviewed before final approval."
)


# ============================================================
# STATUS METRICS
# ============================================================

facts = []

if facts_data:
    facts = facts_data.get(
        "clinical_facts",
        []
    )


soap = {}

if soap_data:
    soap = soap_data.get(
        "soap",
        {}
    )


soap_statement_count = sum(
    len(
        soap.get(
            section,
            []
        )
    )
    for section in [
        "subjective",
        "objective",
        "assessment",
        "plan"
    ]
)


col1, col2, col3, col4 = st.columns(
    4
)


with col1:

    st.metric(
        "Consultation",
        consultation_id.replace(
            "_",
            " "
        )
    )


with col2:

    st.metric(
        "Clinical Facts",
        len(facts)
    )


with col3:

    st.metric(
        "SOAP Statements",
        soap_statement_count
    )


with col4:

    st.metric(
        "Review Status",
        "Draft"
    )


st.divider()


# ============================================================
# TABS
# ============================================================

tab_audio, tab_transcript, tab_facts, tab_soap, tab_review = st.tabs(
    [
        "Audio",
        "Transcript",
        "Clinical Facts",
        "SOAP Record",
        "Human Review"
    ]
)


# ============================================================
# AUDIO TAB
# ============================================================

with tab_audio:

    st.subheader(
        "Consultation Audio"
    )

    if audio_path.exists():

        st.audio(
            str(
                audio_path
            )
        )

        st.success(
            "Consultation audio loaded."
        )

    else:

        st.error(
            "Audio file not found."
        )

# ============================================================
# TRANSCRIPT TAB
# ============================================================

with tab_transcript:

    st.subheader(
        "Whisper Large-v3 Transcript"
    )

    st.caption(
        "Original AI-generated transcription"
    )

    if transcript:

        st.text_area(
            "Transcript",
            value=transcript,
            height=520,
            disabled=True,
            label_visibility="collapsed"
        )

    else:

        st.error(
            "Transcript file not found."
        )


# ============================================================
# CLINICAL FACTS TAB
# ============================================================

with tab_facts:

    st.subheader(
        "Validated Clinical Facts"
    )

    st.caption(
        "Qwen3-8B extraction after deterministic validation"
    )

    if facts:

        fact_rows = []

        for item in facts:

            fact_rows.append({

                "Fact ID":
                    item.get(
                        "fact_id",
                        ""
                    ),

                "Category":
                    item.get(
                        "category",
                        ""
                    ),

                "Status":
                    item.get(
                        "status",
                        ""
                    ),

                "Clinical Fact":
                    item.get(
                        "fact",
                        ""
                    ),

                "Evidence Quote":
                    item.get(
                        "evidence_quote",
                        ""
                    )
            })


        fact_df = pd.DataFrame(
            fact_rows
        )


        st.dataframe(
            fact_df,
            use_container_width=True,
            hide_index=True
        )


        st.caption(
            f"{len(fact_df)} clinical facts"
        )


    else:

        st.error(
            "Clinical facts not found."
        )


# ============================================================
# SOAP TAB
# ============================================================

with tab_soap:

    st.subheader(
        "Canonical SOAP Draft"
    )

    st.caption(
        "Qwen3-8B SOAP generation after deterministic "
        "structural validation"
    )


    if soap_data:

        section_titles = {

            "subjective":
                "Subjective",

            "objective":
                "Objective",

            "assessment":
                "Assessment",

            "plan":
                "Plan"
        }


        for section, title in (
            section_titles.items()
        ):

            st.markdown(
                f"### {title}"
            )

            items = soap.get(
                section,
                []
            )


            if not items:

                if section == "objective":

                    st.info(
                        "No explicit objective findings "
                        "were available."
                    )

                else:

                    st.caption(
                        "No information available."
                    )

                continue


            for item in items:

                statement = item.get(
                    "statement",
                    ""
                )

                source_ids = item.get(
                    "source_fact_ids",
                    []
                )

                st.write(
                    statement
                )

                st.caption(
                    "Source facts: "
                    + ", ".join(
                        source_ids
                    )
                )


    else:

        st.error(
            "SOAP record not found."
        )


# ============================================================
# HUMAN REVIEW TAB
# ============================================================

with tab_review:

    st.subheader(
        "Human-in-the-Loop Review"
    )

    st.info(
        "The original AI transcript is preserved separately. "
        "Reviewer edits are stored as a working copy and "
        "recorded in the correction log."
    )


    review_col1, review_col2 = st.columns(
        2
    )


    # --------------------------------------------------------
    # ORIGINAL TRANSCRIPT
    # --------------------------------------------------------

    with review_col1:

        st.markdown(
            "#### Original Whisper Transcript"
        )

        st.caption(
            "Read-only AI output"
        )

        st.text_area(
            "Original transcript",
            value=transcript or "",
            height=500,
            disabled=True,
            key=f"original_review_{consultation_id}",
            label_visibility="collapsed"
        )


    # --------------------------------------------------------
    # WORKING REVIEW COPY
    # --------------------------------------------------------

    with review_col2:

        st.markdown(
            "#### Reviewer Working Transcript"
        )

        st.caption(
            "Editable human-reviewed copy"
        )

        st.text_area(
            "Reviewer transcript",
            height=500,
            key=working_transcript_key,
            label_visibility="collapsed"
        )


    st.divider()


    # --------------------------------------------------------
    # SAVE CORRECTION
    # --------------------------------------------------------

    if st.button(
        "Save Transcript Correction",
        type="primary",
        use_container_width=True
    ):

        previous_text = (
            st.session_state[
                saved_transcript_key
            ]
        )

        new_text = (
            st.session_state[
                working_transcript_key
            ]
        )


        if new_text == previous_text:

            st.info(
                "No transcript changes detected."
            )

        else:

            # Save reviewer working copy

            working_transcript_path.write_text(
                new_text,
                encoding="utf-8"
            )


            # Load existing correction log

            existing_log = load_json(
                correction_log_path
            )

            if isinstance(
                existing_log,
                list
            ):

                correction_events = (
                    existing_log
                )

            elif isinstance(
                existing_log,
                dict
            ):

                correction_events = (
                    existing_log.get(
                        "events",
                        []
                    )
                )

            else:

                correction_events = []


            # Create new audit event

            event_id = (
                f"E{len(correction_events) + 1:03d}"
            )

            correction_events.append({

                "event_id":
                    event_id,

                "consultation_id":
                    consultation_id,

                "action":
                    "edit_transcript",

                "timestamp_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "previous_text":
                    previous_text,

                "new_text":
                    new_text
            })


            with open(
                correction_log_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    correction_events,
                    f,
                    indent=2,
                    ensure_ascii=False
                )


            st.session_state[
                saved_transcript_key
            ] = new_text


            st.success(
                f"Transcript correction saved "
                f"as {event_id}."
            )


    # --------------------------------------------------------
    # CORRECTION LOG STATUS
    # --------------------------------------------------------

    saved_log = load_json(
        correction_log_path
    )

    if isinstance(saved_log, list):

        saved_events = saved_log

    elif isinstance(saved_log, dict):

        saved_events = saved_log.get(
            "events",
            []
        )

    else:

        saved_events = []


    st.caption(
        f"Saved transcript correction events: "
        f"{len(saved_events)}"
    )

    st.warning(
        "Transcript correction does not mean the record "
        "has been clinically approved."
    )


    # ========================================================
    # CLINICAL FACT REVIEW
    # ========================================================

    st.divider()

    st.markdown(
        "### Clinical Fact Review"
    )

    st.caption(
        "Edit, add or remove extracted clinical facts. "
        "The original Qwen3-8B output remains unchanged."
    )


    working_facts_path = (
        review_dir
        / "working_clinical_facts.json"
    )


    # --------------------------------------------------------
    # LOAD WORKING FACTS
    # --------------------------------------------------------

    if working_facts_path.exists():

        working_facts_data = load_json(
            working_facts_path
        )

        working_facts = (
            working_facts_data.get(
                "clinical_facts",
                []
            )
            if working_facts_data
            else []
        )

    else:

        working_facts = [
            dict(item)
            for item in facts
        ]


    # --------------------------------------------------------
    # BUILD EDITABLE TABLE
    # --------------------------------------------------------

    review_fact_rows = []

    for item in working_facts:

        review_fact_rows.append({

            "Fact ID":
                item.get(
                    "fact_id",
                    ""
                ),

            "Category":
                item.get(
                    "category",
                    ""
                ),

            "Status":
                item.get(
                    "status",
                    ""
                ),

            "Importance":
                item.get(
                    "importance",
                    ""
                ),

            "Clinical Fact":
                item.get(
                    "fact",
                    ""
                ),

            "Evidence Quote":
                item.get(
                    "evidence_quote",
                    ""
                )
        })


    review_fact_df = pd.DataFrame(
        review_fact_rows
    )


    edited_fact_df = st.data_editor(
        review_fact_df,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        disabled=[
            "Fact ID"
        ],
        key=f"facts_editor_{consultation_id}"
    )


    st.caption(
        "Fact IDs are locked for existing facts. "
        "New facts receive an ID automatically when saved."
    )


    # --------------------------------------------------------
    # SAVE FACT CORRECTIONS
    # --------------------------------------------------------

    if st.button(
        "Save Clinical Fact Corrections",
        type="primary",
        width="stretch"
    ):

        edited_records = (
            edited_fact_df
            .fillna("")
            .to_dict(
                orient="records"
            )
        )


        # Existing numeric IDs

        used_numbers = []

        for row in edited_records:

            fact_id = str(
                row.get(
                    "Fact ID",
                    ""
                )
            ).strip()

            if (
                fact_id.startswith("F")
                and
                fact_id[1:].isdigit()
            ):

                used_numbers.append(
                    int(
                        fact_id[1:]
                    )
                )


        next_number = (
            max(
                used_numbers,
                default=0
            )
            + 1
        )


        new_facts = []


        for row in edited_records:

            fact_text = str(
                row.get(
                    "Clinical Fact",
                    ""
                )
            ).strip()


            # Ignore completely empty new rows

            if not fact_text:
                continue


            fact_id = str(
                row.get(
                    "Fact ID",
                    ""
                )
            ).strip()


            # Assign ID to newly added fact

            if not fact_id:

                fact_id = (
                    f"F{next_number:02d}"
                )

                next_number += 1


            new_facts.append({

                "fact_id":
                    fact_id,

                "category":
                    str(
                        row.get(
                            "Category",
                            ""
                        )
                    ).strip(),

                "fact":
                    fact_text,

                "status":
                    str(
                        row.get(
                            "Status",
                            ""
                        )
                    ).strip(),

                "importance":
                    str(
                        row.get(
                            "Importance",
                            ""
                        )
                    ).strip(),

                "evidence_quote":
                    str(
                        row.get(
                            "Evidence Quote",
                            ""
                        )
                    ).strip()
            })


        # ----------------------------------------------------
        # COMPARE PREVIOUS VS NEW FACTS
        # ----------------------------------------------------

        previous_facts = [
            dict(item)
            for item in working_facts
        ]


        old_by_id = {
            item.get("fact_id"):
                item
            for item in previous_facts
        }

        new_by_id = {
            item.get("fact_id"):
                item
            for item in new_facts
        }


        added_ids = sorted(
            set(new_by_id)
            -
            set(old_by_id)
        )

        removed_ids = sorted(
            set(old_by_id)
            -
            set(new_by_id)
        )

        modified_ids = sorted([

            fact_id

            for fact_id
            in (
                set(old_by_id)
                &
                set(new_by_id)
            )

            if (
                old_by_id[fact_id]
                !=
                new_by_id[fact_id]
            )
        ])


        if (
            not added_ids
            and
            not removed_ids
            and
            not modified_ids
        ):

            st.info(
                "No clinical fact changes detected."
            )


        else:

            # -----------------------------------------------
            # SAVE WORKING FACT COPY
            # -----------------------------------------------

            working_payload = {

                "consultation_id":
                    consultation_id,

                "source":
                    "human_reviewed_working_copy",

                "clinical_facts":
                    new_facts
            }


            with open(
                working_facts_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    working_payload,
                    f,
                    indent=2,
                    ensure_ascii=False
                )


            # -----------------------------------------------
            # LOAD SHARED CORRECTION LOG
            # -----------------------------------------------

            existing_log = load_json(
                correction_log_path
            )


            if isinstance(
                existing_log,
                list
            ):

                correction_events = (
                    existing_log
                )

            elif isinstance(
                existing_log,
                dict
            ):

                correction_events = (
                    existing_log.get(
                        "events",
                        []
                    )
                )

            else:

                correction_events = []


            event_id = (
                f"E{len(correction_events) + 1:03d}"
            )


            correction_events.append({

                "event_id":
                    event_id,

                "consultation_id":
                    consultation_id,

                "action":
                    "edit_clinical_facts",

                "timestamp_utc":
                    datetime.now(
                        timezone.utc
                    ).isoformat(),

                "added_fact_ids":
                    added_ids,

                "removed_fact_ids":
                    removed_ids,

                "modified_fact_ids":
                    modified_ids,

                "previous_facts":
                    previous_facts,

                "new_facts":
                    new_facts
            })


            with open(
                correction_log_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    correction_events,
                    f,
                    indent=2,
                    ensure_ascii=False
                )


            st.success(
                f"Clinical fact corrections saved "
                f"as {event_id}."
            )

            st.caption(
                f"Added: {len(added_ids)} | "
                f"Modified: {len(modified_ids)} | "
                f"Removed: {len(removed_ids)}"
            )




# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Medical Consultation AI | Research Prototype"
)
