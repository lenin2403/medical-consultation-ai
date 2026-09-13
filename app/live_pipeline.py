
from pathlib import Path
from datetime import datetime
import hashlib
import json
import re
import shutil

import subprocess
import time
import tempfile
import gc


PROJECT = Path(__file__).resolve().parent.parent

ENCOUNTERS_ROOT = (
    PROJECT
    / "results/live_encounters"
)


def sanitize_name(name: str) -> str:
    """
    Convert an uploaded filename into a safe identifier component.
    """

    stem = Path(name).stem

    stem = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        stem
    )

    stem = stem.strip("_")

    return stem or "consultation"


def create_encounter_id(
    original_filename: str
) -> str:
    """
    Create a unique encounter ID without using patient information.
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    safe_name = sanitize_name(
        original_filename
    )

    short_hash = hashlib.sha256(
        f"{timestamp}_{original_filename}".encode(
            "utf-8"
        )
    ).hexdigest()[:8]

    return (
        f"{timestamp}_"
        f"{safe_name}_"
        f"{short_hash}"
    )


def create_encounter(
    uploaded_file
):
    """
    Save a newly uploaded consultation into its own encounter folder.

    No AI processing occurs here yet.
    """

    encounter_id = create_encounter_id(
        uploaded_file.name
    )

    encounter_dir = (
        ENCOUNTERS_ROOT
        / encounter_id
    )

    audio_dir = (
        encounter_dir
        / "audio"
    )

    transcript_dir = (
        encounter_dir
        / "transcript"
    )

    facts_dir = (
        encounter_dir
        / "clinical_facts"
    )

    soap_dir = (
        encounter_dir
        / "soap"
    )

    review_dir = (
        encounter_dir
        / "review"
    )

    export_dir = (
        encounter_dir
        / "exports"
    )


    for folder in [
        audio_dir,
        transcript_dir,
        facts_dir,
        soap_dir,
        review_dir,
        export_dir
    ]:

        folder.mkdir(
            parents=True,
            exist_ok=True
        )


    suffix = (
        Path(
            uploaded_file.name
        )
        .suffix
        .lower()
    )

    if not suffix:
        suffix = ".audio"


    audio_path = (
        audio_dir
        / f"original{suffix}"
    )


    with open(
        audio_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    metadata = {

        "encounter_id":
            encounter_id,

        "created_at":
            datetime.now().isoformat(),

        "original_filename":
            uploaded_file.name,

        "stored_audio":
            str(
                audio_path.relative_to(
                    PROJECT
                )
            ),

        "status":
            "uploaded",

        "pipeline": {

            "asr":
                "pending",

            "clinical_fact_extraction":
                "pending",

            "soap_generation":
                "pending",

            "human_review":
                "pending",

            "approval":
                "pending"
        }
    }


    metadata_path = (
        encounter_dir
        / "encounter_metadata.json"
    )


    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )


    return {
        "encounter_id":
            encounter_id,

        "encounter_dir":
            encounter_dir,

        "audio_path":
            audio_path,

        "metadata_path":
            metadata_path,

        "metadata":
            metadata
    }


def load_encounter_metadata(
    encounter_id: str
):

    metadata_path = (
        ENCOUNTERS_ROOT
        / encounter_id
        / "encounter_metadata.json"
    )

    if not metadata_path.exists():
        return None

    return json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )


def list_live_encounters():

    if not ENCOUNTERS_ROOT.exists():
        return []

    encounters = []

    for folder in sorted(
        ENCOUNTERS_ROOT.iterdir(),
        reverse=True
    ):

        if not folder.is_dir():
            continue

        metadata_path = (
            folder
            / "encounter_metadata.json"
        )

        if not metadata_path.exists():
            continue

        try:

            metadata = json.loads(
                metadata_path.read_text(
                    encoding="utf-8"
                )
            )

            encounters.append(
                metadata
            )

        except Exception:

            continue

    return encounters



# ============================================================
# LIVE ASR — WHISPER LARGE-V3
# ============================================================

_WHISPER_MODEL = None


def update_encounter_metadata(
    encounter_id: str,
    metadata: dict
):
    """
    Persist encounter metadata.
    """

    metadata_path = (
        ENCOUNTERS_ROOT
        / encounter_id
        / "encounter_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )


def prepare_audio_for_asr(
    encounter_id: str
):
    """
    Convert the original uploaded consultation into
    16 kHz mono PCM WAV for consistent ASR inference.
    """

    encounter_dir = (
        ENCOUNTERS_ROOT
        / encounter_id
    )

    metadata = load_encounter_metadata(
        encounter_id
    )

    if metadata is None:
        raise FileNotFoundError(
            f"Encounter not found: {encounter_id}"
        )

    original_path = (
        PROJECT
        / metadata["stored_audio"]
    )

    if not original_path.exists():
        raise FileNotFoundError(
            original_path
        )

    prepared_path = (
        encounter_dir
        / "audio"
        / "prepared_16khz_mono.wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(original_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(prepared_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Audio preprocessing failed.\n"
            + result.stderr[-3000:]
        )

    metadata["prepared_audio"] = str(
        prepared_path.relative_to(
            PROJECT
        )
    )

    metadata["pipeline"][
        "audio_preprocessing"
    ] = "complete"

    metadata["updated_at"] = (
        datetime.now().isoformat()
    )

    update_encounter_metadata(
        encounter_id,
        metadata
    )

    return prepared_path


def get_whisper_model():
    """
    Load Whisper Large-v3 once per Python process.

    The cache is stored in temporary runtime storage,
    not Google Drive.
    """

    global _WHISPER_MODEL

    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    # Release Qwen before loading Whisper on the 15 GB T4.
    try:
        from live_clinical_pipeline import release_qwen_model
        release_qwen_model()
    except Exception:
        pass

    import torch
    from faster_whisper import WhisperModel

    if torch.cuda.is_available():

        device = "cuda"
        compute_type = "float16"

    else:

        device = "cpu"
        compute_type = "int8"

    cache_dir = (
        Path(tempfile.gettempdir())
        / "medical_consultation_ai_models"
        / "whisper"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    _WHISPER_MODEL = WhisperModel(
        "large-v3",
        device=device,
        compute_type=compute_type,
        download_root=str(cache_dir)
    )

    return _WHISPER_MODEL


def transcribe_encounter(
    encounter_id: str
):
    """
    Run the frozen Whisper Large-v3 ASR protocol on
    a live uploaded consultation.

    Frozen GPU configuration:
        model: large-v3
        language: English
        beam_size: 5
        vad_filter: True
        word_timestamps: True
        condition_on_previous_text: True
    """

    import torch

    encounter_dir = (
        ENCOUNTERS_ROOT
        / encounter_id
    )

    metadata = load_encounter_metadata(
        encounter_id
    )

    if metadata is None:

        raise FileNotFoundError(
            f"Encounter not found: {encounter_id}"
        )

    metadata["pipeline"]["asr"] = (
        "processing"
    )

    metadata["updated_at"] = (
        datetime.now().isoformat()
    )

    update_encounter_metadata(
        encounter_id,
        metadata
    )


    # --------------------------------------------------------
    # PREPROCESS AUDIO
    # --------------------------------------------------------

    prepared_audio = (
        encounter_dir
        / "audio"
        / "prepared_16khz_mono.wav"
    )

    if not prepared_audio.exists():

        prepared_audio = (
            prepare_audio_for_asr(
                encounter_id
            )
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model_load_start = (
        time.perf_counter()
    )

    model = get_whisper_model()

    model_load_seconds = (
        time.perf_counter()
        -
        model_load_start
    )


    # --------------------------------------------------------
    # TRANSCRIBE
    # --------------------------------------------------------

    inference_start = (
        time.perf_counter()
    )

    segments_generator, info = (
        model.transcribe(
            str(prepared_audio),
            language="en",
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=True
        )
    )

    segments = []

    transcript_parts = []


    for segment in segments_generator:

        text = (
            segment.text
            or ""
        ).strip()

        if text:
            transcript_parts.append(
                text
            )

        words = []

        if segment.words:

            for word in segment.words:

                words.append({

                    "start":
                        round(
                            word.start,
                            3
                        )
                        if word.start is not None
                        else None,

                    "end":
                        round(
                            word.end,
                            3
                        )
                        if word.end is not None
                        else None,

                    "word":
                        word.word,

                    "probability":
                        round(
                            word.probability,
                            4
                        )
                        if word.probability is not None
                        else None
                })


        segments.append({

            "id":
                segment.id,

            "start":
                round(
                    segment.start,
                    3
                ),

            "end":
                round(
                    segment.end,
                    3
                ),

            "text":
                text,

            "words":
                words
        })


    inference_seconds = (
        time.perf_counter()
        -
        inference_start
    )


    transcript = " ".join(
        transcript_parts
    ).strip()


    if not transcript:

        metadata["pipeline"]["asr"] = (
            "failed"
        )

        metadata["asr_error"] = (
            "Whisper returned an empty transcript."
        )

        update_encounter_metadata(
            encounter_id,
            metadata
        )

        raise RuntimeError(
            "Whisper returned an empty transcript."
        )


    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------

    transcript_dir = (
        encounter_dir
        / "transcript"
    )

    transcript_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    transcript_path = (
        transcript_dir
        / "whisper_transcript.txt"
    )

    segments_path = (
        transcript_dir
        / "whisper_segments.json"
    )

    asr_metadata_path = (
        transcript_dir
        / "whisper_metadata.json"
    )


    transcript_path.write_text(
        transcript,
        encoding="utf-8"
    )


    segments_path.write_text(
        json.dumps(
            segments,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    audio_duration = float(
        info.duration
    )

    rtf = (
        inference_seconds
        / audio_duration
        if audio_duration > 0
        else None
    )


    asr_metadata = {

        "model":
            "Whisper Large-v3",

        "implementation":
            "faster-whisper 1.2.1",

        "language":
            "en",

        "beam_size":
            5,

        "vad_filter":
            True,

        "word_timestamps":
            True,

        "condition_on_previous_text":
            True,

        "device":
            (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "audio_duration_seconds":
            audio_duration,

        "model_load_seconds":
            model_load_seconds,

        "transcription_seconds":
            inference_seconds,

        "real_time_factor":
            rtf,

        "segments":
            len(segments),

        "transcript_words":
            len(
                transcript.split()
            ),

        "completed_at":
            datetime.now().isoformat()
    }


    asr_metadata_path.write_text(
        json.dumps(
            asr_metadata,
            indent=2
        ),
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # UPDATE ENCOUNTER
    # --------------------------------------------------------

    metadata = load_encounter_metadata(
        encounter_id
    )

    metadata["pipeline"]["asr"] = (
        "complete"
    )

    metadata["status"] = (
        "transcribed"
    )

    metadata["transcript"] = str(
        transcript_path.relative_to(
            PROJECT
        )
    )

    metadata["whisper_segments"] = str(
        segments_path.relative_to(
            PROJECT
        )
    )

    metadata["asr_metadata"] = str(
        asr_metadata_path.relative_to(
            PROJECT
        )
    )

    metadata["updated_at"] = (
        datetime.now().isoformat()
    )

    update_encounter_metadata(
        encounter_id,
        metadata
    )


    return {

        "encounter_id":
            encounter_id,

        "transcript":
            transcript,

        "transcript_path":
            transcript_path,

        "segments_path":
            segments_path,

        "metadata":
            asr_metadata
    }

