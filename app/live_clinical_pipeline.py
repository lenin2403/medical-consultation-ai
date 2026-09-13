
from pathlib import Path
from datetime import datetime
import gc
import json
import re
import time

try:
    import torch
except ImportError:
    torch = None

from live_pipeline import (
    PROJECT,
    ENCOUNTERS_ROOT,
    load_encounter_metadata,
    update_encounter_metadata
)


QWEN_MODEL_ID = "unsloth/Qwen3-8B-bnb-4bit"
QWEN_BASE_MODEL_ID = "Qwen/Qwen3-8B"
QWEN_CACHE_DIR = PROJECT / "models" / "qwen3_8b_4bit_cache"

_QWEN_MODEL = None
_QWEN_TOKENIZER = None


# ============================================================
# SCHEMA
# ============================================================

ALLOWED_CATEGORIES = {
    "presenting_complaint",
    "symptom",
    "temporal_detail",
    "medication",
    "allergy",
    "medical_history",
    "family_history",
    "social_history",
    "exposure_history",
    "objective_finding",
    "vital_sign",
    "investigation_result",
    "assessment",
    "diagnosis",
    "plan",
    "safety_netting",
    "referral",
    "follow_up"
}


SECTION_CATEGORIES = {

    "subjective": {
        "presenting_complaint",
        "symptom",
        "temporal_detail",
        "medication",
        "allergy",
        "medical_history",
        "family_history",
        "social_history",
        "exposure_history"
    },

    "objective": {
        "objective_finding",
        "vital_sign",
        "investigation_result"
    },

    "assessment": {
        "assessment",
        "diagnosis"
    },

    "plan": {
        "plan",
        "safety_netting",
        "referral",
        "follow_up"
    }
}


def _now():
    return datetime.now().isoformat()


def _encounter_dir(encounter_id):

    return (
        ENCOUNTERS_ROOT
        / encounter_id
    )


def _normalize_grounding(text):

    text = str(
        text or ""
    ).lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def _extract_json(text):

    cleaned = (
        text
        .strip()
    )

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    ).strip()

    first = cleaned.find("{")
    last = cleaned.rfind("}")

    if (
        first >= 0
        and
        last > first
    ):
        cleaned = cleaned[
            first:last + 1
        ]

    return json.loads(
        cleaned
    )


# ============================================================
# GPU MODEL MANAGEMENT
# ============================================================

def release_qwen_model():

    global _QWEN_MODEL
    global _QWEN_TOKENIZER

    if _QWEN_MODEL is not None:

        try:
            del _QWEN_MODEL
        except Exception:
            pass

    if _QWEN_TOKENIZER is not None:

        try:
            del _QWEN_TOKENIZER
        except Exception:
            pass

    _QWEN_MODEL = None
    _QWEN_TOKENIZER = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_whisper():

    try:

        import live_pipeline

        if getattr(
            live_pipeline,
            "_WHISPER_MODEL",
            None
        ) is not None:

            try:
                del live_pipeline._WHISPER_MODEL
            except Exception:
                pass

            live_pipeline._WHISPER_MODEL = None

    except Exception:

        pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_qwen_model():

    global _QWEN_MODEL
    global _QWEN_TOKENIZER

    if (
        _QWEN_MODEL is not None
        and
        _QWEN_TOKENIZER is not None
    ):
        return (
            _QWEN_MODEL,
            _QWEN_TOKENIZER
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "Qwen3-8B live inference requires a CUDA GPU."
        )


    # --------------------------------------------------------
    # Whisper and Qwen are never kept on the T4 together.
    # --------------------------------------------------------

    _release_whisper()

    gc.collect()
    torch.cuda.empty_cache()


    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer
    )


    # --------------------------------------------------------
    # Locate the already-downloaded pre-quantized snapshot.
    # No full-precision model is downloaded or quantized here.
    # --------------------------------------------------------

    snapshots_root = (
        QWEN_CACHE_DIR
        / "models--unsloth--Qwen3-8B-bnb-4bit"
        / "snapshots"
    )

    if not snapshots_root.exists():

        raise RuntimeError(
            "Qwen3-8B 4-bit model cache was not found. "
            "Download the deployment model before launching the app."
        )


    snapshots = [
        path
        for path
        in snapshots_root.iterdir()
        if path.is_dir()
    ]


    if not snapshots:

        raise RuntimeError(
            "No completed Qwen3-8B 4-bit snapshot was found."
        )


    model_path = max(
        snapshots,
        key=lambda path:
            path.stat().st_mtime
    )


    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    _QWEN_TOKENIZER = (
        AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True
        )
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    _QWEN_MODEL = (
        AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True
        )
    )

    _QWEN_MODEL.eval()


    return (
        _QWEN_MODEL,
        _QWEN_TOKENIZER
    )


def _generate_json(
    prompt,
    max_new_tokens
):

    model, tokenizer = (
        get_qwen_model()
    )

    messages = [
        {
            "role":
                "user",

            "content":
                prompt
        }
    ]

    chat_text = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt"
    ).to(
        "cuda"
    )

    input_tokens = (
        inputs[
            "input_ids"
        ].shape[-1]
    )

    started = (
        time.perf_counter()
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(
                tokenizer.eos_token_id
            )
        )

    seconds = (
        time.perf_counter()
        -
        started
    )

    generated = output[
        0,
        input_tokens:
    ]

    response = (
        tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()
    )

    return (
        response,
        seconds
    )


# ============================================================
# TRANSCRIPT SELECTION
# ============================================================

def get_review_source_transcript(
    encounter_id
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    reviewed = (
        encounter_dir
        / "review"
        / "working_transcript.txt"
    )

    if reviewed.exists():

        return (
            reviewed.read_text(
                encoding="utf-8"
            ),
            "human_reviewed_transcript"
        )

    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    if metadata is None:

        raise FileNotFoundError(
            encounter_id
        )

    transcript_relative = (
        metadata.get(
            "transcript"
        )
    )

    if not transcript_relative:

        raise RuntimeError(
            "Generate the Whisper transcript first."
        )

    transcript_path = (
        PROJECT
        / transcript_relative
    )

    return (
        transcript_path.read_text(
            encoding="utf-8"
        ),
        "original_whisper_transcript"
    )


# ============================================================
# CLINICAL FACT EXTRACTION
# ============================================================

def extract_clinical_facts(
    encounter_id
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    facts_dir = (
        encounter_dir
        / "clinical_facts"
    )

    facts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    metadata[
        "pipeline"
    ][
        "clinical_fact_extraction"
    ] = "processing"

    metadata[
        "updated_at"
    ] = _now()

    update_encounter_metadata(
        encounter_id,
        metadata
    )


    transcript, source_name = (
        get_review_source_transcript(
            encounter_id
        )
    )


    prompt = f"""
You are extracting clinical facts from a doctor-patient
consultation transcript for a CLINICIAN-REVIEWED draft record.

Return ONLY facts that are explicitly supported by the transcript.

STRICT SAFETY RULES:

1. Do not invent, infer or assume clinical information.
2. Do not independently diagnose the patient.
3. A diagnosis or assessment may be extracted ONLY when the clinician
   explicitly states it in the transcript.
4. Preserve negation exactly. Example: "no vomiting" must remain absent,
   not present.
5. Preserve uncertainty. Use status="uncertain" when the speaker expresses
   uncertainty or a possible finding.
6. Preserve medication names, allergies, doses, durations and numerical
   information as accurately as the transcript permits.
7. Every fact must include a short evidence_quote copied from the transcript.
8. Do not extract greetings, administrative scheduling or unrelated
   conversation unless clinically relevant.
9. Do not turn a clinician question into a positive patient finding.
10. Output valid JSON only. No Markdown or explanation.

Allowed categories:

presenting_complaint
symptom
temporal_detail
medication
allergy
medical_history
family_history
social_history
exposure_history
objective_finding
vital_sign
investigation_result
assessment
diagnosis
plan
safety_netting
referral
follow_up

Allowed status values:

present
absent
uncertain

Allowed importance values:

critical
important
routine

Return exactly this structure:

{{
  "clinical_facts": [
    {{
      "category": "symptom",
      "fact": "Example clinical fact",
      "status": "present",
      "importance": "important",
      "evidence_quote": "exact supporting words from transcript"
    }}
  ]
}}

TRANSCRIPT:

{transcript}
"""


    raw_response, generation_seconds = (
        _generate_json(
            prompt,
            max_new_tokens=2600
        )
    )


    raw_path = (
        facts_dir
        / "qwen3_raw_response.txt"
    )

    raw_path.write_text(
        raw_response,
        encoding="utf-8"
    )


    parsed = _extract_json(
        raw_response
    )

    raw_facts = parsed.get(
        "clinical_facts",
        []
    )

    transcript_normalized = (
        _normalize_grounding(
            transcript
        )
    )

    validated = []

    for item in raw_facts:

        if not isinstance(
            item,
            dict
        ):
            continue

        category = str(
            item.get(
                "category",
                ""
            )
        ).strip()

        fact = str(
            item.get(
                "fact",
                ""
            )
        ).strip()

        status = str(
            item.get(
                "status",
                ""
            )
        ).strip()

        importance = str(
            item.get(
                "importance",
                ""
            )
        ).strip()

        quote = str(
            item.get(
                "evidence_quote",
                ""
            )
        ).strip()


        if (
            category
            not in ALLOWED_CATEGORIES
        ):
            continue

        if not fact:
            continue

        if status not in {
            "present",
            "absent",
            "uncertain"
        }:
            continue

        if importance not in {
            "critical",
            "important",
            "routine"
        }:
            importance = "important"

        if not quote:
            continue


        quote_normalized = (
            _normalize_grounding(
                quote
            )
        )


        if (
            not quote_normalized
            or
            quote_normalized
            not in transcript_normalized
        ):
            # Unsupported evidence is not accepted
            # into the canonical live facts.
            continue


        validated.append({
            "fact_id":
                f"F{len(validated) + 1:02d}",

            "category":
                category,

            "status":
                status,

            "importance":
                importance,

            "fact":
                fact,

            "evidence_quote":
                quote
        })


    canonical = {
        "encounter_id":
            encounter_id,

        "model":
            QWEN_BASE_MODEL_ID,

        "deployment_model":
            QWEN_MODEL_ID,

        "quantization":
            "4-bit bitsandbytes",

        "source":
            source_name,

        "review_status":
            "ai_draft_unapproved",

        "clinical_facts":
            validated,

        "generation_seconds":
            round(
                generation_seconds,
                2
            )
    }


    canonical_path = (
        facts_dir
        / "ai_clinical_facts.json"
    )

    canonical_path.write_text(
        json.dumps(
            canonical,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    metadata[
        "pipeline"
    ][
        "clinical_fact_extraction"
    ] = "complete"

    metadata[
        "clinical_facts"
    ] = str(
        canonical_path.relative_to(
            PROJECT
        )
    )

    metadata[
        "clinical_fact_count"
    ] = len(
        validated
    )

    metadata[
        "updated_at"
    ] = _now()

    update_encounter_metadata(
        encounter_id,
        metadata
    )

    return canonical


# ============================================================
# FACT REVIEW
# ============================================================

def load_live_facts(
    encounter_id,
    prefer_reviewed=True
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    reviewed = (
        encounter_dir
        / "review"
        / "working_clinical_facts.json"
    )

    ai_path = (
        encounter_dir
        / "clinical_facts"
        / "ai_clinical_facts.json"
    )


    if (
        prefer_reviewed
        and
        reviewed.exists()
    ):

        return json.loads(
            reviewed.read_text(
                encoding="utf-8"
            )
        )


    if ai_path.exists():

        return json.loads(
            ai_path.read_text(
                encoding="utf-8"
            )
        )

    return None


def _append_log(
    encounter_id,
    event_type,
    payload
):

    log_path = (
        _encounter_dir(
            encounter_id
        )
        / "review"
        / "correction_log.jsonl"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    entry = {
        "timestamp":
            _now(),

        "event_type":
            event_type,

        **payload
    }

    with open(
        log_path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                entry,
                ensure_ascii=False
            )
            + "\n"
        )


def save_reviewed_facts(
    encounter_id,
    facts
):

    clean_facts = []

    for index, item in enumerate(
        facts,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):
            continue

        fact = str(
            item.get(
                "fact",
                ""
            )
        ).strip()

        if not fact:
            continue

        clean_facts.append({
            "fact_id":
                f"F{index:02d}",

            "category":
                str(
                    item.get(
                        "category",
                        "symptom"
                    )
                ),

            "status":
                str(
                    item.get(
                        "status",
                        "present"
                    )
                ),

            "importance":
                str(
                    item.get(
                        "importance",
                        "important"
                    )
                ),

            "fact":
                fact,

            "evidence_quote":
                str(
                    item.get(
                        "evidence_quote",
                        ""
                    )
                )
        })


    output = {
        "encounter_id":
            encounter_id,

        "review_status":
            "human_reviewed",

        "clinical_facts":
            clean_facts
    }


    path = (
        _encounter_dir(
            encounter_id
        )
        / "review"
        / "working_clinical_facts.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    previous = (
        path.read_text(
            encoding="utf-8"
        )
        if path.exists()
        else None
    )

    path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    _append_log(
        encounter_id,
        "clinical_fact_review_saved",
        {
            "previous":
                previous,

            "new":
                output
        }
    )


    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    metadata[
        "reviewed_clinical_facts"
    ] = str(
        path.relative_to(
            PROJECT
        )
    )

    metadata[
        "pipeline"
    ][
        "human_review"
    ] = "in_progress"

    update_encounter_metadata(
        encounter_id,
        metadata
    )

    return output


# ============================================================
# SOAP GENERATION
# ============================================================

def _section_for_fact(
    category
):

    for section, categories in (
        SECTION_CATEGORIES.items()
    ):

        if category in categories:
            return section

    return None


def generate_soap(
    encounter_id
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    soap_dir = (
        encounter_dir
        / "soap"
    )

    soap_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    facts_document = (
        load_live_facts(
            encounter_id,
            prefer_reviewed=True
        )
    )

    if not facts_document:

        raise RuntimeError(
            "Extract clinical facts first."
        )

    source_facts = (
        facts_document.get(
            "clinical_facts",
            []
        )
    )

    if not source_facts:

        raise RuntimeError(
            "No validated clinical facts are available."
        )


    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    metadata[
        "pipeline"
    ][
        "soap_generation"
    ] = "processing"

    update_encounter_metadata(
        encounter_id,
        metadata
    )


    facts_text = json.dumps(
        source_facts,
        indent=2,
        ensure_ascii=False
    )


    prompt = f"""
Create a concise SOAP medical record draft using ONLY the
validated clinical facts supplied below.

This is clinical DOCUMENTATION, not autonomous diagnosis.

STRICT RULES:

1. Do not use information that is not contained in the supplied facts.
2. Do not invent examination findings, vital signs, tests,
   diagnoses, treatments or history.
3. Preserve all negative findings correctly.
4. Preserve uncertainty.
5. Every SOAP statement MUST contain one or more source_fact_ids.
6. source_fact_ids may ONLY use the supplied Fxx identifiers.
7. Do not cite a fact that does not support the statement.
8. Keep the record concise and clinically readable.
9. The output remains an unapproved draft until clinician review.
10. Return valid JSON only.
11. No Markdown, commentary or reasoning.

SOAP RULES:

SUBJECTIVE:
Use presenting complaint, symptoms, history of present illness,
medications, allergies, relevant medical/family/social/exposure history
and patient-reported context.

OBJECTIVE:
Use ONLY explicitly documented objective findings, physical examination
findings, vital signs or investigation results.
Never invent a normal examination.
If no objective information was documented, return [].

ASSESSMENT:
Use ONLY an assessment or diagnosis explicitly stated by the clinician.
Never independently infer a diagnosis from symptoms.

PLAN:
Use ONLY explicitly documented plans, treatment, safety-netting,
referrals or follow-up.

Return:

{{
  "encounter_id": "{encounter_id}",
  "record_type": "SOAP",
  "review_status": "draft_unapproved",
  "soap": {{
    "subjective": [],
    "objective": [],
    "assessment": [],
    "plan": []
  }}
}}

Each item inside a SOAP section must be:

{{
  "statement": "clinical statement",
  "source_fact_ids": ["F01"]
}}

VALIDATED CLINICAL FACTS:

{facts_text}
"""


    raw_response, generation_seconds = (
        _generate_json(
            prompt,
            max_new_tokens=2000
        )
    )


    raw_path = (
        soap_dir
        / "qwen3_soap_raw.txt"
    )

    raw_path.write_text(
        raw_response,
        encoding="utf-8"
    )


    parsed = _extract_json(
        raw_response
    )


    valid_fact_ids = {
        item.get(
            "fact_id"
        )
        for item
        in source_facts
    }

    fact_lookup = {
        item.get(
            "fact_id"
        ):
        item
        for item
        in source_facts
    }


    canonical_sections = {
        "subjective": [],
        "objective": [],
        "assessment": [],
        "plan": []
    }

    covered_ids = set()

    generated_sections = (
        parsed.get(
            "soap",
            {}
        )
    )


    # --------------------------------------------------------
    # Keep only structurally valid generated statements
    # --------------------------------------------------------

    for section in canonical_sections:

        items = generated_sections.get(
            section,
            []
        )

        if not isinstance(
            items,
            list
        ):
            continue

        for item in items:

            if not isinstance(
                item,
                dict
            ):
                continue

            statement = str(
                item.get(
                    "statement",
                    ""
                )
            ).strip()

            ids = [
                fact_id
                for fact_id
                in item.get(
                    "source_fact_ids",
                    []
                )
                if fact_id
                in valid_fact_ids
            ]


            compatible_ids = []

            for fact_id in ids:

                source_item = (
                    fact_lookup[
                        fact_id
                    ]
                )

                expected_section = (
                    _section_for_fact(
                        source_item.get(
                            "category"
                        )
                    )
                )

                if (
                    expected_section
                    == section
                ):
                    compatible_ids.append(
                        fact_id
                    )


            if (
                statement
                and
                compatible_ids
            ):

                canonical_sections[
                    section
                ].append({
                    "statement":
                        statement,

                    "source_fact_ids":
                        compatible_ids
                })

                covered_ids.update(
                    compatible_ids
                )


    # --------------------------------------------------------
    # Deterministic restoration of any source facts Qwen missed
    # --------------------------------------------------------

    for fact in source_facts:

        fact_id = fact.get(
            "fact_id"
        )

        if fact_id in covered_ids:
            continue

        section = (
            _section_for_fact(
                fact.get(
                    "category"
                )
            )
        )

        if section is None:
            continue

        canonical_sections[
            section
        ].append({
            "statement":
                str(
                    fact.get(
                        "fact",
                        ""
                    )
                ),

            "source_fact_ids":
                [
                    fact_id
                ]
        })


    canonical = {
        "encounter_id":
            encounter_id,

        "record_type":
            "SOAP",

        "review_status":
            "draft_unapproved",

        "model":
            QWEN_BASE_MODEL_ID,

        "deployment_model":
            QWEN_MODEL_ID,

        "quantization":
            "4-bit bitsandbytes",

        "generation_seconds":
            round(
                generation_seconds,
                2
            ),

        "soap":
            canonical_sections
    }


    soap_path = (
        soap_dir
        / "ai_soap_draft.json"
    )

    soap_path.write_text(
        json.dumps(
            canonical,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    metadata[
        "pipeline"
    ][
        "soap_generation"
    ] = "complete"

    metadata[
        "soap_draft"
    ] = str(
        soap_path.relative_to(
            PROJECT
        )
    )

    metadata[
        "updated_at"
    ] = _now()

    update_encounter_metadata(
        encounter_id,
        metadata
    )

    return canonical


# ============================================================
# SOAP REVIEW
# ============================================================

def load_live_soap(
    encounter_id,
    prefer_reviewed=True
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    reviewed = (
        encounter_dir
        / "review"
        / "working_soap.json"
    )

    ai_path = (
        encounter_dir
        / "soap"
        / "ai_soap_draft.json"
    )


    if (
        prefer_reviewed
        and
        reviewed.exists()
    ):

        return json.loads(
            reviewed.read_text(
                encoding="utf-8"
            )
        )


    if ai_path.exists():

        return json.loads(
            ai_path.read_text(
                encoding="utf-8"
            )
        )

    return None


def save_reviewed_soap(
    encounter_id,
    sections
):

    document = {
        "encounter_id":
            encounter_id,

        "record_type":
            "SOAP",

        "review_status":
            "human_reviewed",

        "soap":
            {
                "subjective":
                    sections.get(
                        "subjective",
                        []
                    ),

                "objective":
                    sections.get(
                        "objective",
                        []
                    ),

                "assessment":
                    sections.get(
                        "assessment",
                        []
                    ),

                "plan":
                    sections.get(
                        "plan",
                        []
                    )
            }
    }


    path = (
        _encounter_dir(
            encounter_id
        )
        / "review"
        / "working_soap.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    previous = (
        path.read_text(
            encoding="utf-8"
        )
        if path.exists()
        else None
    )

    path.write_text(
        json.dumps(
            document,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    _append_log(
        encounter_id,
        "soap_review_saved",
        {
            "previous":
                previous,

            "new":
                document
        }
    )


    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    metadata[
        "reviewed_soap"
    ] = str(
        path.relative_to(
            PROJECT
        )
    )

    metadata[
        "pipeline"
    ][
        "human_review"
    ] = "in_progress"

    update_encounter_metadata(
        encounter_id,
        metadata
    )

    return document


# ============================================================
# FINAL APPROVAL
# ============================================================

def approve_live_record(
    encounter_id
):

    encounter_dir = (
        _encounter_dir(
            encounter_id
        )
    )

    reviewed_transcript = (
        encounter_dir
        / "review"
        / "working_transcript.txt"
    )

    reviewed_facts = (
        encounter_dir
        / "review"
        / "working_clinical_facts.json"
    )

    reviewed_soap = (
        encounter_dir
        / "review"
        / "working_soap.json"
    )


    if not reviewed_transcript.exists():

        raise RuntimeError(
            "Save the transcript review before approval."
        )

    if not reviewed_facts.exists():

        raise RuntimeError(
            "Save the clinical fact review before approval."
        )

    if not reviewed_soap.exists():

        raise RuntimeError(
            "Save the SOAP review before approval."
        )


    final_record = {
        "encounter_id":
            encounter_id,

        "approval_status":
            "clinician_approved",

        "approved_at":
            _now(),

        "reviewed_transcript":
            reviewed_transcript.read_text(
                encoding="utf-8"
            ),

        "clinical_facts":
            json.loads(
                reviewed_facts.read_text(
                    encoding="utf-8"
                )
            ).get(
                "clinical_facts",
                []
            ),

        "soap":
            json.loads(
                reviewed_soap.read_text(
                    encoding="utf-8"
                )
            ).get(
                "soap",
                {}
            )
    }


    final_path = (
        encounter_dir
        / "review"
        / "approved_final_record.json"
    )

    final_path.write_text(
        json.dumps(
            final_record,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    _append_log(
        encounter_id,
        "record_approved",
        {
            "approved_record":
                str(
                    final_path.relative_to(
                        PROJECT
                    )
                )
        }
    )


    metadata = (
        load_encounter_metadata(
            encounter_id
        )
    )

    metadata[
        "pipeline"
    ][
        "approval"
    ] = "complete"

    metadata[
        "pipeline"
    ][
        "human_review"
    ] = "complete"

    metadata[
        "status"
    ] = "approved"

    metadata[
        "approved_final_record"
    ] = str(
        final_path.relative_to(
            PROJECT
        )
    )

    metadata[
        "updated_at"
    ] = _now()

    update_encounter_metadata(
        encounter_id,
        metadata
    )

    return final_record
