
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import re
import statistics
import unicodedata

import pandas as pd


TARGET_JIWER = "4.0.0"


def sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def package_version(name):

    try:
        return metadata.version(name)

    except metadata.PackageNotFoundError:
        return None


def normalize_for_wer(text):
    """
    Frozen development/final normalization:

    1. Unicode NFKC
    2. lowercase
    3. punctuation and symbols -> spaces
    4. collapse repeated whitespace
    """

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    cleaned = []

    for character in text:

        category = unicodedata.category(
            character
        )

        if (
            category.startswith("P")
            or
            category.startswith("S")
        ):

            cleaned.append(" ")

        else:

            cleaned.append(
                character
            )

    text = "".join(
        cleaned
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Score frozen Whisper Large-v3 output "
            "on PRI-MOCK57 held-out final-test set."
        )
    )

    parser.add_argument(
        "--project",
        required=True
    )

    args = parser.parse_args()

    project = Path(
        args.project
    ).resolve()

    if not project.exists():
        raise FileNotFoundError(
            project
        )


    # ========================================================
    # SOFTWARE GATE
    # ========================================================

    jiwer_version = package_version(
        "jiwer"
    )

    if jiwer_version != TARGET_JIWER:

        raise RuntimeError(
            "jiwer version mismatch. "
            f"Expected {TARGET_JIWER}, "
            f"found {jiwer_version}"
        )


    from jiwer import process_words


    # ========================================================
    # INPUT PATHS
    # ========================================================

    lock_path = (
        project
        / "documentation"
        / "final_evaluation"
        / "final_test_lock_manifest.json"
    )

    protocol_path = (
        project
        / "documentation"
        / "final_evaluation"
        / "final_evaluation_protocol.json"
    )

    runner_freeze_path = (
        project
        / "documentation"
        / "final_evaluation"
        / "final_whisper_runner_freeze.json"
    )

    reference_dir = (
        project
        / "data"
        / "prepared"
        / "evaluation_references"
        / "final_test"
    )

    result_root = (
        project
        / "results"
        / "asr"
        / "whisper_large_v3"
        / "final_test"
    )

    transcript_dir = (
        result_root
        / "transcripts"
    )

    inference_manifest_path = (
        result_root
        / "final_test_inference_manifest.csv"
    )

    run_manifest_path = (
        result_root
        / "final_test_run_manifest.json"
    )


    for path in [
        lock_path,
        protocol_path,
        runner_freeze_path
    ]:

        if not path.exists():
            raise FileNotFoundError(
                path
            )


    if not transcript_dir.exists():

        raise RuntimeError(
            "Final-test Whisper inference has not been run."
        )


    if not inference_manifest_path.exists():

        raise FileNotFoundError(
            inference_manifest_path
        )


    if not run_manifest_path.exists():

        raise FileNotFoundError(
            run_manifest_path
        )


    # ========================================================
    # LOCKED CONSULTATIONS
    # ========================================================

    lock = json.loads(
        lock_path.read_text(
            encoding="utf-8"
        )
    )

    consultation_ids = lock[
        "final_test_consultation_ids"
    ]

    if len(consultation_ids) != 17:

        raise RuntimeError(
            f"Expected 17 final-test cases; "
            f"found {len(consultation_ids)}."
        )


    inference_df = pd.read_csv(
        inference_manifest_path
    )


    # ========================================================
    # PER-CASE SCORING
    # ========================================================

    rows = []

    corpus_reference_parts = []
    corpus_hypothesis_parts = []


    print("=" * 92)
    print("FINAL HELD-OUT ASR SCORING")
    print("=" * 92)


    for index, consultation_id in enumerate(
        consultation_ids,
        start=1
    ):

        print(
            f"\n[{index:02d}/17] "
            f"{consultation_id}"
        )


        reference_path = (
            reference_dir
            / (
                f"{consultation_id}"
                "_reference_normalized.txt"
            )
        )

        transcript_path = (
            transcript_dir
            / (
                f"{consultation_id}"
                "_transcript.txt"
            )
        )


        # ----------------------------------------------------
        # FAILED / MISSING CASE POLICY
        # ----------------------------------------------------

        if not reference_path.exists():

            rows.append({
                "consultation_id":
                    consultation_id,

                "status":
                    "reference_missing"
            })

            print(
                "  FAIL — reference missing"
            )

            continue


        if not transcript_path.exists():

            rows.append({
                "consultation_id":
                    consultation_id,

                "status":
                    "transcript_missing"
            })

            print(
                "  FAIL — transcript missing"
            )

            continue


        reference = (
            reference_path.read_text(
                encoding="utf-8"
            )
            .strip()
        )


        hypothesis_raw = (
            transcript_path.read_text(
                encoding="utf-8"
            )
            .strip()
        )


        hypothesis = normalize_for_wer(
            hypothesis_raw
        )


        if not reference:

            rows.append({
                "consultation_id":
                    consultation_id,

                "status":
                    "empty_reference"
            })

            print(
                "  FAIL — empty reference"
            )

            continue


        result = process_words(
            reference,
            hypothesis
        )


        reference_words = len(
            reference.split()
        )

        hypothesis_words = len(
            hypothesis.split()
        )


        total_errors = (
            result.substitutions
            +
            result.deletions
            +
            result.insertions
        )


        # ----------------------------------------------------
        # LATENCY / RTF
        # ----------------------------------------------------

        inference_row = inference_df[
            inference_df[
                "consultation_id"
            ]
            ==
            consultation_id
        ]


        if len(inference_row) == 1:

            inference_seconds = float(
                inference_row.iloc[0][
                    "inference_seconds"
                ]
            )

            audio_duration_seconds = float(
                inference_row.iloc[0][
                    "audio_duration_seconds"
                ]
            )

            rtf = float(
                inference_row.iloc[0][
                    "real_time_factor"
                ]
            )

        else:

            inference_seconds = None
            audio_duration_seconds = None
            rtf = None


        row = {

            "consultation_id":
                consultation_id,

            "status":
                "success",

            "reference_words":
                reference_words,

            "hypothesis_words":
                hypothesis_words,

            "hits":
                int(
                    result.hits
                ),

            "substitutions":
                int(
                    result.substitutions
                ),

            "deletions":
                int(
                    result.deletions
                ),

            "insertions":
                int(
                    result.insertions
                ),

            "total_errors":
                int(
                    total_errors
                ),

            "wer":
                float(
                    result.wer
                ),

            "wer_percent":
                float(
                    result.wer * 100
                ),

            "mer":
                float(
                    result.mer
                ),

            "wil":
                float(
                    result.wil
                ),

            "audio_duration_seconds":
                audio_duration_seconds,

            "inference_seconds":
                inference_seconds,

            "real_time_factor":
                rtf,

            "reference_sha256":
                sha256(
                    reference_path
                ),

            "transcript_sha256":
                sha256(
                    transcript_path
                )
        }


        rows.append(
            row
        )


        corpus_reference_parts.append(
            reference
        )

        corpus_hypothesis_parts.append(
            hypothesis
        )


        print(
            "  WER:",
            f"{result.wer * 100:.2f}%"
        )

        print(
            "  Errors:",
            total_errors,
            "| Ref words:",
            reference_words
        )


    # ========================================================
    # SAVE PER-CASE RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        rows
    )


    per_case_path = (
        result_root
        / "final_test_asr_per_consultation.csv"
    )


    results_df.to_csv(
        per_case_path,
        index=False
    )


    successful = results_df[
        results_df[
            "status"
        ]
        ==
        "success"
    ].copy()


    # ========================================================
    # ENFORCE REPORT-ALL-17 POLICY
    # ========================================================

    successful_cases = len(
        successful
    )

    failed_cases = (
        17
        -
        successful_cases
    )


    if successful_cases == 0:

        raise RuntimeError(
            "No final-test cases could be scored."
        )


    # ========================================================
    # CORPUS METRICS
    # ========================================================

    corpus_reference = " ".join(
        corpus_reference_parts
    )

    corpus_hypothesis = " ".join(
        corpus_hypothesis_parts
    )


    corpus_result = process_words(
        corpus_reference,
        corpus_hypothesis
    )


    macro_wer = float(
        successful[
            "wer"
        ].mean()
    )


    median_wer = float(
        successful[
            "wer"
        ].median()
    )


    minimum_wer = float(
        successful[
            "wer"
        ].min()
    )


    maximum_wer = float(
        successful[
            "wer"
        ].max()
    )


    total_reference_words = int(
        successful[
            "reference_words"
        ].sum()
    )


    total_hypothesis_words = int(
        successful[
            "hypothesis_words"
        ].sum()
    )


    # ========================================================
    # TIMING METRICS
    # ========================================================

    valid_rtf = successful[
        "real_time_factor"
    ].dropna()


    valid_inference = successful[
        "inference_seconds"
    ].dropna()


    total_inference_seconds = (
        float(
            valid_inference.sum()
        )
        if len(valid_inference)
        else None
    )


    mean_rtf = (
        float(
            valid_rtf.mean()
        )
        if len(valid_rtf)
        else None
    )


    median_rtf = (
        float(
            valid_rtf.median()
        )
        if len(valid_rtf)
        else None
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "evaluation_name":
            "PRI-MOCK57 held-out final ASR evaluation",

        "created_at":
            datetime.now().isoformat(),

        "model":
            "Whisper Large-v3",

        "implementation":
            "faster-whisper",

        "jiwer_version":
            TARGET_JIWER,

        "locked_final_test_cases":
            17,

        "successfully_scored_cases":
            successful_cases,

        "failed_or_missing_cases":
            failed_cases,

        "primary_metrics": {

            "corpus_wer":
                float(
                    corpus_result.wer
                ),

            "corpus_wer_percent":
                float(
                    corpus_result.wer
                    *
                    100
                ),

            "macro_wer":
                macro_wer,

            "macro_wer_percent":
                macro_wer * 100,

            "median_wer":
                median_wer,

            "median_wer_percent":
                median_wer * 100,

            "min_wer":
                minimum_wer,

            "min_wer_percent":
                minimum_wer * 100,

            "max_wer":
                maximum_wer,

            "max_wer_percent":
                maximum_wer * 100
        },

        "corpus_error_counts": {

            "reference_words":
                total_reference_words,

            "hypothesis_words":
                total_hypothesis_words,

            "hits":
                int(
                    corpus_result.hits
                ),

            "substitutions":
                int(
                    corpus_result.substitutions
                ),

            "deletions":
                int(
                    corpus_result.deletions
                ),

            "insertions":
                int(
                    corpus_result.insertions
                ),

            "total_errors":
                int(
                    corpus_result.substitutions
                    +
                    corpus_result.deletions
                    +
                    corpus_result.insertions
                )
        },

        "corpus_secondary_metrics": {

            "mer":
                float(
                    corpus_result.mer
                ),

            "wil":
                float(
                    corpus_result.wil
                )
        },

        "timing": {

            "total_inference_seconds":
                total_inference_seconds,

            "mean_real_time_factor":
                mean_rtf,

            "median_real_time_factor":
                median_rtf
        },

        "reporting_policy": {

            "all_17_cases_retained":
                True,

            "failures_silently_excluded":
                False,

            "primary_metric":
                "corpus WER"
        },

        "evidence_hashes": {

            "lock_manifest_sha256":
                sha256(
                    lock_path
                ),

            "evaluation_protocol_sha256":
                sha256(
                    protocol_path
                ),

            "runner_freeze_sha256":
                sha256(
                    runner_freeze_path
                ),

            "run_manifest_sha256":
                sha256(
                    run_manifest_path
                ),

            "inference_manifest_sha256":
                sha256(
                    inference_manifest_path
                )
        }
    }


    summary_path = (
        result_root
        / "final_test_asr_summary.json"
    )


    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    # ========================================================
    # HUMAN-READABLE SUMMARY
    # ========================================================

    report_path = (
        result_root
        / "final_test_asr_summary.txt"
    )


    report_text = f"""
PRI-MOCK57 HELD-OUT FINAL ASR EVALUATION
========================================

Model
-----
Whisper Large-v3
faster-whisper

Cases
-----
Locked final-test cases: 17
Successfully scored: {successful_cases}
Failed/missing: {failed_cases}

Primary result
--------------
Corpus WER: {corpus_result.wer * 100:.2f}%

Case-level WER
--------------
Macro WER: {macro_wer * 100:.2f}%
Median WER: {median_wer * 100:.2f}%
Minimum WER: {minimum_wer * 100:.2f}%
Maximum WER: {maximum_wer * 100:.2f}%

Corpus counts
-------------
Reference words: {total_reference_words}
Hypothesis words: {total_hypothesis_words}
Hits: {corpus_result.hits}
Substitutions: {corpus_result.substitutions}
Deletions: {corpus_result.deletions}
Insertions: {corpus_result.insertions}

Secondary error metrics
-----------------------
MER: {corpus_result.mer:.4f}
WIL: {corpus_result.wil:.4f}

Timing
------
Total inference seconds: {total_inference_seconds}
Mean RTF: {mean_rtf}
Median RTF: {median_rtf}

Reporting policy
----------------
All 17 locked cases remain visible.
Failed or missing cases are not silently excluded.
Corpus WER is the primary ASR metric.
"""


    report_path.write_text(
        report_text.strip()
        + "\n",
        encoding="utf-8"
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 92)
    print("FINAL ASR SCORING COMPLETE")
    print("=" * 92)

    print(
        "Successfully scored:",
        successful_cases,
        "/ 17"
    )

    print(
        "Corpus WER:",
        f"{corpus_result.wer * 100:.2f}%"
    )

    print(
        "Macro WER:",
        f"{macro_wer * 100:.2f}%"
    )

    print(
        "Median WER:",
        f"{median_wer * 100:.2f}%"
    )

    print(
        "Per-case CSV:",
        per_case_path
    )

    print(
        "Summary JSON:",
        summary_path
    )

    print(
        "Summary TXT:",
        report_path
    )


if __name__ == "__main__":
    main()
