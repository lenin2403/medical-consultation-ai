
import os
import json
from datetime import datetime

import requests

import live_pipeline as local
from live_pipeline import *


def _backend_url():

    url = os.environ.get(
        "MEDICAL_AI_BACKEND_URL",
        ""
    ).strip().rstrip("/")

    if not url:
        raise RuntimeError(
            "MEDICAL_AI_BACKEND_URL is not configured."
        )

    return url


def transcribe_encounter(encounter_id):

    metadata = local.load_encounter_metadata(
        encounter_id
    )

    if metadata is None:
        raise FileNotFoundError(
            f"Encounter not found: {encounter_id}"
        )

    audio_path = (
        local.PROJECT
        / metadata["stored_audio"]
    )

    with open(audio_path, "rb") as f:

        response = requests.post(
            _backend_url() + "/transcribe",
            files={
                "audio": (
                    metadata.get(
                        "original_filename",
                        audio_path.name
                    ),
                    f,
                    "application/octet-stream"
                )
            },
            timeout=1800
        )

    response.raise_for_status()

    payload = response.json()

    transcript = payload["transcript"]

    encounter_dir = (
        local.ENCOUNTERS_ROOT
        / encounter_id
    )

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

    metadata_path = (
        transcript_dir
        / "whisper_metadata.json"
    )

    transcript_path.write_text(
        transcript,
        encoding="utf-8"
    )

    metadata_path.write_text(
        json.dumps(
            payload.get(
                "asr_metadata",
                {}
            ),
            indent=2
        ),
        encoding="utf-8"
    )

    metadata = local.load_encounter_metadata(
        encounter_id
    )

    metadata["pipeline"]["asr"] = "complete"
    metadata["status"] = "transcribed"

    metadata["remote_backend_encounter_id"] = (
        payload["encounter_id"]
    )

    metadata["transcript"] = str(
        transcript_path.relative_to(
            local.PROJECT
        )
    )

    metadata["asr_metadata"] = str(
        metadata_path.relative_to(
            local.PROJECT
        )
    )

    metadata["updated_at"] = (
        datetime.now().isoformat()
    )

    local.update_encounter_metadata(
        encounter_id,
        metadata
    )

    return {
        "encounter_id": encounter_id,
        "transcript": transcript,
        "transcript_path": transcript_path,
        "segments_path": None,
        "metadata": payload.get(
            "asr_metadata",
            {}
        )
    }
