from pathlib import Path
from difflib import SequenceMatcher
import argparse
import hashlib
import json
import re
import time

import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)


MODEL_ID = "Qwen/Qwen3-8B"

FINAL_CASES = [
    "day1_consultation03",
    "day1_consultation04",
    "day1_consultation08",
    "day1_consultation09",
    "day1_consultation10",
    "day1_consultation11",
    "day2_consultation02",
    "day2_consultation03",
    "day2_consultation04",
    "day2_consultation06",
    "day2_consultation07",
    "day3_consultation08",
    "day4_consultation07",
    "day5_consultation01",
    "day5_consultation05",
    "day5_consultation06",
    "day5_consultation09",
]

ALLOWED_CATEGORIES = {
    "presenting_complaint",
    "symptom",
    "temporal_detail",
    "medication",
    "allergy",
    "medical_history",
    "exposure_history",
    "social_history",
    "assessment",
    "plan",
    "safety_netting"
}

VALID_STATUS = {
    "present",
    "absent"
}

VALID_IMPORTANCE = {
    "critical",
    "important"
}


def sha256(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def norm(text):

    text = str(text).lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def extract_json(text):

    cleaned = text.strip()

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

    if first >= 0 and last > first:
        cleaned = cleaned[
            first:last + 1
        ]

    return json.loads(
        cleaned
    )


def repair_quote(
    quote,
    transcript
):

    if quote in transcript:

        return (
            quote,
            1.0,
            False
        )

    transcript_words = (
        transcript.split()
    )

    quote_words = (
        quote.split()
    )

    target = norm(
        quote
    )

    n = len(
        quote_words
    )

    if n == 0:

        return (
            quote,
            0.0,
            False
        )

    best_score = 0.0
    best_text = None

    min_size = max(
        3,
        n - 5
    )

    max_size = min(
        len(transcript_words),
        n + 6
    )

    for window_size in range(
        min_size,
        max_size + 1
    ):

        for start in range(
            0,
            len(transcript_words)
            - window_size
            + 1
        ):

            candidate = " ".join(
                transcript_words[
                    start:
                    start + window_size
                ]
            )

            score = SequenceMatcher(
                None,
                target,
                norm(candidate)
            ).ratio()

            if score > best_score:

                best_score = score
                best_text = candidate

    if (
        best_text
        and
        best_score >= 0.78
    ):

        return (
            best_text,
            best_score,
            True
        )

    return (
        quote,
        best_score,
        False
    )


def standardize_negative_fact(
    fact,
    status,
    evidence
):

    fact = str(fact).strip()

    lower_fact = (
        fact.lower()
    )

    lower_evidence = (
        str(evidence).lower()
    )

    negative = False

    if lower_fact.startswith(
        "no "
    ):
        negative = True

    if lower_fact in {
        "non-smoker",
        "non smoker",
        "non-alcoholic",
        "non alcoholic"
    }:
        negative = True

    patterns = [
        "no blood",
        "don't smoke",
        "do not smoke",
        "doesn't smoke",
        "does not smoke",
        "don't drink alcohol",
        "do not drink alcohol",
        "doesn't drink alcohol",
        "does not drink alcohol",
        "no vomiting",
        "no fever",
        "no pain",
        "no discharge",
        "no weakness",
        "no numbness",
        "no tingling",
        "no visual",
        "no chest pain",
        "no medication",
        "no medications"
    ]

    if any(
        p in lower_evidence
        for p in patterns
    ):
        negative = True

    if negative:

        status = "absent"

        if lower_fact.startswith(
            "no "
        ):

            fact = fact[
                3:
            ].strip()

            if fact:

                fact = (
                    fact[0].upper()
                    + fact[1:]
                )

        elif lower_fact in {
            "non-smoker",
            "non smoker"
        }:

            fact = "Smoking"

        elif lower_fact in {
            "non-alcoholic",
            "non alcoholic"
        }:

            fact = "Alcohol consumption"

    return (
        fact,
        status
    )


def build_prompt(
    consultation_id,
    transcript
):

    return f"""
Extract the most clinically important facts from this
doctor-patient consultation.

Use ONLY information explicitly stated in the transcript.

STRICT REQUIREMENTS:

1. Return between 15 and 20 clinical facts TOTAL.
2. Do not duplicate the same clinical information.
3. Read the entire transcript, including the doctor's
   assessment, treatment plan and follow-up advice.
4. Prioritize clinically important information over ordinary
   social details.
5. Never invent or infer information.

CATEGORY DEFINITIONS:

presenting_complaint
- Main reason for consultation.

symptom
- Positive or explicitly denied symptom/finding.

temporal_detail
- Important onset, duration, frequency or severity.

medication
- Current medication only.

allergy
- Drug or substance allergy only.

medical_history
- Diagnosed past or current medical condition only.

exposure_history
- Relevant food, travel, infectious or environmental exposure.

social_history
- Smoking, alcohol, occupation or living situation only if
  clinically relevant.

assessment
- Doctor's working diagnosis or differential.

plan
- Treatment, medication advice, investigations or management.

safety_netting
- Follow-up instructions, return precautions or escalation advice.

STATUS RULE:

Use "present" when the clinical fact is affirmed.

Use "absent" when the patient explicitly denies the clinical fact.

Examples:

Fact: "Blood in vomit"
status: "absent"

Fact: "Smoking"
status: "absent"

Fact: "Asthma"
status: "present"

Do NOT write:
"No blood in vomit" with status "present".

EVIDENCE RULE:

Every evidence_quote MUST:
- be copied exactly from ONE continuous span of the transcript
- contain approximately 3 to 18 words
- not contain ellipses
- not combine separate parts of the transcript
- not paraphrase or correct the transcript

Do NOT use a doctor's question as evidence unless the patient's
answer confirming or denying the fact is included in the same
short continuous quote.

BALANCE THE OUTPUT:

Include:
- main complaint
- major symptoms
- clinically important negations
- important duration/frequency/numerical detail
- relevant history/medications/allergies/exposure
- assessment if stated
- important treatment plan
- follow-up or safety-netting

Do not spend most of the 15-20 facts only on symptoms.

Return JSON only.
No reasoning.
No explanation.
No Markdown.

Required structure:

{{
  "consultation_id": "{consultation_id}",
  "clinical_facts": [
    {{
      "category": "",
      "fact": "",
      "status": "present",
      "importance": "critical",
      "evidence_quote": ""
    }}
  ]
}}

importance must be exactly:
"critical" or "important"

TRANSCRIPT:

{transcript}
"""


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project",
        required=True
    )

    args = parser.parse_args()

    project = Path(
        args.project
    )

    transcript_dir = (
        project
        / "results/asr/whisper_large_v3/"
          "final_test/transcripts"
    )

    output_root = (
        project
        / "results/nlp/qwen3_8b/final_test"
    )

    raw_dir = (
        output_root
        / "raw"
    )

    extraction_dir = (
        output_root
        / "extractions"
    )

    canonical_dir = (
        output_root
        / "canonical"
    )

    metadata_dir = (
        output_root
        / "metadata"
    )

    for folder in [
        raw_dir,
        extraction_dir,
        canonical_dir,
        metadata_dir
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


    print("=" * 80)
    print("FINAL HELD-OUT QWEN3-8B EXTRACTION")
    print("=" * 80)

    print(
        "Model:",
        MODEL_ID
    )

    print(
        "dtype: bfloat16"
    )

    print(
        "Thinking: disabled"
    )

    print(
        "Sampling: disabled"
    )

    print(
        "Cases:",
        len(FINAL_CASES)
    )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print(
        "\nLoading Qwen3-8B..."
    )

    load_started = (
        time.perf_counter()
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_ID
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            device_map="auto"
        )
    )

    model.eval()

    print(
        "Model load seconds:",
        round(
            time.perf_counter()
            - load_started,
            2
        )
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        )
    )


    summary_rows = []

    batch_started = (
        time.perf_counter()
    )


    for index, cid in enumerate(
        FINAL_CASES,
        start=1
    ):

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"[{index}/17] {cid}"
        )

        print(
            "=" * 80
        )

        transcript_path = (
            transcript_dir
            / f"{cid}_transcript.txt"
        )

        if not transcript_path.exists():

            raise FileNotFoundError(
                transcript_path
            )

        transcript = (
            transcript_path.read_text(
                encoding="utf-8"
            )
        )

        prompt = build_prompt(
            cid,
            transcript
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
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        )

        inputs = (
            tokenizer(
                chat_text,
                return_tensors="pt"
            )
            .to(
                model.device
            )
        )

        input_tokens = (
            inputs[
                "input_ids"
            ].shape[-1]
        )

        torch.cuda.reset_peak_memory_stats()

        started = (
            time.perf_counter()
        )

        with torch.inference_mode():

            output_ids = (
                model.generate(
                    **inputs,
                    max_new_tokens=2200,
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.eos_token_id
                    )
                )
            )

        generation_seconds = (
            time.perf_counter()
            - started
        )

        generated_ids = (
            output_ids[
                0,
                input_tokens:
            ]
        )

        generated_tokens = len(
            generated_ids
        )

        response = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            .strip()
        )

        peak_gpu_gb = (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        )

        raw_path = (
            raw_dir
            / f"{cid}_qwen3_raw.txt"
        )

        raw_path.write_text(
            response,
            encoding="utf-8"
        )

        json_valid = False
        json_error = None
        parsed = None

        try:

            parsed = (
                extract_json(
                    response
                )
            )

            json_valid = True

        except Exception as e:

            json_error = str(
                e
            )


        raw_facts = []

        if json_valid:

            raw_facts = (
                parsed.get(
                    "clinical_facts",
                    []
                )
            )

            extraction_path = (
                extraction_dir
                / f"{cid}_qwen3_extraction.json"
            )

            extraction_path.write_text(
                json.dumps(
                    parsed,
                    indent=2,
                    ensure_ascii=False
                ),
                encoding="utf-8"
            )


        canonical_facts = []

        quote_repairs = 0
        status_corrections = 0
        invalid_items = 0


        for item in raw_facts:

            if not isinstance(
                item,
                dict
            ):

                invalid_items += 1
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

                invalid_items += 1
                continue

            if not fact:

                invalid_items += 1
                continue

            if (
                status
                not in VALID_STATUS
            ):

                invalid_items += 1
                continue

            if (
                importance
                not in VALID_IMPORTANCE
            ):

                invalid_items += 1
                continue

            if not quote:

                invalid_items += 1
                continue


            original_fact = fact
            original_status = status

            fact, status = (
                standardize_negative_fact(
                    fact,
                    status,
                    quote
                )
            )

            if (
                fact != original_fact
                or
                status != original_status
            ):

                status_corrections += 1


            repaired_quote, score, repaired = (
                repair_quote(
                    quote,
                    transcript
                )
            )

            if repaired:

                quote = (
                    repaired_quote
                )

                quote_repairs += 1


            # Frozen grounding requirement:
            # canonical evidence must be an exact transcript span.
            if quote not in transcript:

                invalid_items += 1
                continue


            canonical_facts.append({
                "fact_id":
                    f"F{len(canonical_facts)+1:02d}",

                "category":
                    category,

                "fact":
                    fact,

                "status":
                    status,

                "importance":
                    importance,

                "evidence_quote":
                    quote
            })


        canonical = {
            "consultation_id":
                cid,

            "model":
                MODEL_ID,

            "evaluation_configuration":
                "bfloat16",

            "thinking_enabled":
                False,

            "sampling_enabled":
                False,

            "post_processing":
                True,

            "source":
                "held_out_whisper_large_v3_transcript",

            "clinical_facts":
                canonical_facts
        }


        canonical_path = (
            canonical_dir
            / f"{cid}_qwen3_canonical.json"
        )

        canonical_path.write_text(
            json.dumps(
                canonical,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )


        metadata = {
            "consultation_id":
                cid,

            "model":
                MODEL_ID,

            "dtype":
                "bfloat16",

            "thinking_enabled":
                False,

            "do_sample":
                False,

            "max_new_tokens":
                2200,

            "input_tokens":
                int(
                    input_tokens
                ),

            "generated_tokens":
                int(
                    generated_tokens
                ),

            "generation_seconds":
                round(
                    generation_seconds,
                    4
                ),

            "peak_gpu_gb":
                round(
                    peak_gpu_gb,
                    4
                ),

            "json_valid":
                bool(
                    json_valid
                ),

            "json_error":
                json_error,

            "raw_facts":
                len(
                    raw_facts
                ),

            "canonical_facts":
                len(
                    canonical_facts
                ),

            "status_corrections":
                status_corrections,

            "quote_repairs":
                quote_repairs,

            "invalid_or_rejected_items":
                invalid_items,

            "transcript_sha256":
                sha256(
                    transcript_path
                )
        }


        metadata_path = (
            metadata_dir
            / f"{cid}_qwen3_metadata.json"
        )

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2
            ),
            encoding="utf-8"
        )


        summary_rows.append(
            metadata
        )


        print(
            "JSON valid:",
            json_valid
        )

        print(
            "Raw facts:",
            len(
                raw_facts
            )
        )

        print(
            "Canonical facts:",
            len(
                canonical_facts
            )
        )

        print(
            "Quote repairs:",
            quote_repairs
        )

        print(
            "Rejected items:",
            invalid_items
        )

        print(
            "Generation seconds:",
            round(
                generation_seconds,
                2
            )
        )


    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_csv = (
        output_root
        / "final_qwen_extraction_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False
    )


    summary_json = {
        "model":
            MODEL_ID,

        "evaluation_configuration":
            "bfloat16",

        "thinking_enabled":
            False,

        "do_sample":
            False,

        "max_new_tokens":
            2200,

        "consultations":
            len(
                summary_df
            ),

        "valid_json":
            int(
                summary_df[
                    "json_valid"
                ].sum()
            ),

        "total_raw_facts":
            int(
                summary_df[
                    "raw_facts"
                ].sum()
            ),

        "total_canonical_facts":
            int(
                summary_df[
                    "canonical_facts"
                ].sum()
            ),

        "total_generation_seconds":
            float(
                summary_df[
                    "generation_seconds"
                ].sum()
            ),

        "mean_generation_seconds":
            float(
                summary_df[
                    "generation_seconds"
                ].mean()
            ),

        "batch_wall_seconds":
            round(
                time.perf_counter()
                - batch_started,
                4
            )
    }


    summary_json_path = (
        output_root
        / "final_qwen_extraction_summary.json"
    )

    summary_json_path.write_text(
        json.dumps(
            summary_json,
            indent=2
        ),
        encoding="utf-8"
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL QWEN EXTRACTION RUN COMPLETE"
    )

    print(
        "=" * 80
    )

    print(
        json.dumps(
            summary_json,
            indent=2
        )
    )


if __name__ == "__main__":
    main()
