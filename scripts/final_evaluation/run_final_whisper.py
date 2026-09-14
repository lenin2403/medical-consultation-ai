
from pathlib import Path
from datetime import datetime
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import time


TARGET_FASTER_WHISPER = "1.2.1"
TARGET_JIWER = "4.0.0"

MODEL_NAME = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

LANGUAGE = "en"
BEAM_SIZE = 5
VAD_FILTER = True
WORD_TIMESTAMPS = True
CONDITION_ON_PREVIOUS_TEXT = True


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


def verify_gpu():

    try:

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader"
            ],
            capture_output=True,
            text=True,
            timeout=20
        )

    except Exception as error:

        raise RuntimeError(
            f"NVIDIA GPU check failed: {error}"
        )

    if (
        result.returncode != 0
        or
        not result.stdout.strip()
    ):

        raise RuntimeError(
            "No usable NVIDIA GPU detected."
        )

    return result.stdout.strip()


def verify_environment():

    fw_version = package_version(
        "faster-whisper"
    )

    jiwer_version = package_version(
        "jiwer"
    )

    if fw_version != TARGET_FASTER_WHISPER:

        raise RuntimeError(
            "faster-whisper version mismatch. "
            f"Expected {TARGET_FASTER_WHISPER}, "
            f"found {fw_version}"
        )

    if jiwer_version != TARGET_JIWER:

        raise RuntimeError(
            "jiwer version mismatch. "
            f"Expected {TARGET_JIWER}, "
            f"found {jiwer_version}"
        )

    gpu = verify_gpu()

    return {
        "faster_whisper":
            fw_version,

        "jiwer":
            jiwer_version,

        "gpu":
            gpu
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run frozen Whisper Large-v3 inference "
            "on the 17 PRI-MOCK57 held-out final-test cases."
        )
    )

    parser.add_argument(
        "--project",
        required=True,
        help="Medical Consultation AI project root"
    )

    args = parser.parse_args()

    project = Path(
        args.project
    ).resolve()

    if not project.exists():
        raise FileNotFoundError(project)


    # ========================================================
    # FROZEN INPUTS
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

    preparation_manifest_path = (
        project
        / "data"
        / "prepared"
        / "manifests"
        / "final_test_preparation_manifest.csv"
    )

    audio_dir = (
        project
        / "data"
        / "prepared"
        / "final_test"
        / "audio"
    )

    for path in [
        lock_path,
        protocol_path,
        preparation_manifest_path
    ]:

        if not path.exists():
            raise FileNotFoundError(path)


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
            f"Expected 17 locked cases, "
            f"found {len(consultation_ids)}."
        )


    # ========================================================
    # ENVIRONMENT GATE
    # ========================================================

    environment = verify_environment()

    print("=" * 90)
    print("FINAL WHISPER LARGE-v3 HELD-OUT EVALUATION")
    print("=" * 90)

    print(
        "GPU:",
        environment["gpu"]
    )

    print(
        "faster-whisper:",
        environment["faster_whisper"]
    )

    print(
        "jiwer:",
        environment["jiwer"]
    )


    # Import only AFTER all checks pass.
    from faster_whisper import WhisperModel


    # ========================================================
    # OUTPUT LOCATION
    # ========================================================

    output_root = (
        project
        / "results"
        / "asr"
        / "whisper_large_v3"
        / "final_test"
    )

    transcript_dir = (
        output_root
        / "transcripts"
    )

    segment_dir = (
        output_root
        / "segments"
    )

    transcript_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    segment_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Do not silently overwrite prior final inference.
    existing = list(
        transcript_dir.glob(
            "*_transcript.txt"
        )
    )

    if existing:

        raise RuntimeError(
            "Final-test Whisper transcripts already exist. "
            "Runner will not overwrite them."
        )


    # ========================================================
    # LOAD MODEL ONCE
    # ========================================================

    print("\nLoading frozen ASR model...")

    model_load_start = time.perf_counter()

    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    model_load_seconds = (
        time.perf_counter()
        -
        model_load_start
    )

    print(
        "Model loaded in",
        round(
            model_load_seconds,
            2
        ),
        "seconds"
    )


    # ========================================================
    # INFERENCE
    # ========================================================

    result_rows = []

    run_started = (
        datetime.now().isoformat()
    )


    for index, consultation_id in enumerate(
        consultation_ids,
        start=1
    ):

        audio_path = (
            audio_dir
            / f"{consultation_id}_mixed.wav"
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                audio_path
            )

        print(
            f"\n[{index:02d}/17] "
            f"{consultation_id}"
        )


        start = time.perf_counter()


        segments_generator, info = (
            model.transcribe(
                str(audio_path),
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=VAD_FILTER,
                word_timestamps=WORD_TIMESTAMPS,
                condition_on_previous_text=(
                    CONDITION_ON_PREVIOUS_TEXT
                )
            )
        )


        segments = []

        transcript_parts = []


        for segment in segments_generator:

            segment_record = {

                "id":
                    int(segment.id),

                "start":
                    float(segment.start),

                "end":
                    float(segment.end),

                "text":
                    segment.text.strip(),

                "words": []
            }


            if segment.words:

                for word in segment.words:

                    segment_record[
                        "words"
                    ].append({

                        "start":
                            None
                            if word.start is None
                            else float(word.start),

                        "end":
                            None
                            if word.end is None
                            else float(word.end),

                        "word":
                            word.word,

                        "probability":
                            None
                            if word.probability is None
                            else float(
                                word.probability
                            )
                    })


            segments.append(
                segment_record
            )

            transcript_parts.append(
                segment.text.strip()
            )


        inference_seconds = (
            time.perf_counter()
            -
            start
        )


        transcript = " ".join(
            part
            for part in transcript_parts
            if part
        ).strip()


        duration_seconds = float(
            info.duration
        )


        rtf = (
            inference_seconds
            /
            duration_seconds
            if duration_seconds > 0
            else None
        )


        transcript_path = (
            transcript_dir
            / f"{consultation_id}_transcript.txt"
        )


        segment_path = (
            segment_dir
            / f"{consultation_id}_segments.json"
        )


        transcript_path.write_text(
            transcript + "\n",
            encoding="utf-8"
        )


        segment_path.write_text(
            json.dumps(
                {
                    "consultation_id":
                        consultation_id,

                    "model":
                        "Whisper Large-v3",

                    "settings": {
                        "language":
                            LANGUAGE,

                        "beam_size":
                            BEAM_SIZE,

                        "vad_filter":
                            VAD_FILTER,

                        "word_timestamps":
                            WORD_TIMESTAMPS,

                        "condition_on_previous_text":
                            CONDITION_ON_PREVIOUS_TEXT,

                        "compute_type":
                            COMPUTE_TYPE
                    },

                    "detected_language":
                        info.language,

                    "language_probability":
                        float(
                            info.language_probability
                        ),

                    "duration_seconds":
                        duration_seconds,

                    "inference_seconds":
                        inference_seconds,

                    "real_time_factor":
                        rtf,

                    "segments":
                        segments
                },
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )


        result_rows.append({

            "consultation_id":
                consultation_id,

            "status":
                "success",

            "audio_duration_seconds":
                duration_seconds,

            "inference_seconds":
                inference_seconds,

            "real_time_factor":
                rtf,

            "transcript_words_raw":
                len(
                    transcript.split()
                ),

            "detected_language":
                info.language,

            "language_probability":
                float(
                    info.language_probability
                ),

            "audio_sha256":
                sha256(
                    audio_path
                ),

            "transcript_sha256":
                sha256(
                    transcript_path
                )
        })


        print(
            "  duration:",
            round(
                duration_seconds,
                2
            ),
            "s"
        )

        print(
            "  inference:",
            round(
                inference_seconds,
                2
            ),
            "s"
        )

        print(
            "  RTF:",
            round(
                rtf,
                4
            )
        )


    # ========================================================
    # SAVE INFERENCE MANIFEST
    # ========================================================

    csv_path = (
        output_root
        / "final_test_inference_manifest.csv"
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = list(
            result_rows[0].keys()
        )

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            result_rows
        )


    run_manifest = {

        "run_started_at":
            run_started,

        "run_completed_at":
            datetime.now().isoformat(),

        "number_of_cases":
            len(
                result_rows
            ),

        "model":
            "Whisper Large-v3",

        "implementation":
            "faster-whisper",

        "software": {
            "faster_whisper":
                environment[
                    "faster_whisper"
                ],

            "jiwer":
                environment[
                    "jiwer"
                ]
        },

        "hardware":
            environment[
                "gpu"
            ],

        "settings": {
            "device":
                DEVICE,

            "compute_type":
                COMPUTE_TYPE,

            "language":
                LANGUAGE,

            "beam_size":
                BEAM_SIZE,

            "vad_filter":
                VAD_FILTER,

            "word_timestamps":
                WORD_TIMESTAMPS,

            "condition_on_previous_text":
                CONDITION_ON_PREVIOUS_TEXT
        },

        "model_load_seconds":
            model_load_seconds,

        "protocol_sha256":
            sha256(
                protocol_path
            ),

        "lock_manifest_sha256":
            sha256(
                lock_path
            ),

        "preparation_manifest_sha256":
            sha256(
                preparation_manifest_path
            )
    }


    run_manifest_path = (
        output_root
        / "final_test_run_manifest.json"
    )


    run_manifest_path.write_text(
        json.dumps(
            run_manifest,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    print("\n" + "=" * 90)
    print("INFERENCE COMPLETE")
    print("=" * 90)

    print(
        "Cases:",
        len(
            result_rows
        )
    )

    print(
        "Inference manifest:",
        csv_path
    )

    print(
        "Run manifest:",
        run_manifest_path
    )

    print(
        "\nWER has NOT been calculated by this runner."
    )


if __name__ == "__main__":
    main()
