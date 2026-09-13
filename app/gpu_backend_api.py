
from pathlib import Path
from datetime import datetime
import sys
import json
import shutil
import tempfile
import traceback

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

import torch


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    "/home/jovyan/Case_Study_2_Medical_Consultation_AI"
)

APP_DIR = PROJECT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(APP_DIR)
    )


# ============================================================
# EXISTING PROJECT PIPELINE
# ============================================================

import live_pipeline
import live_clinical_pipeline


# Redirect the old Colab deployment cache path to Data Lab.
DATA_LAB_QWEN_CACHE = (
    PROJECT
    / "models"
    / "qwen3_8b_4bit_cache"
)

DATA_LAB_QWEN_CACHE.mkdir(
    parents=True,
    exist_ok=True
)

live_clinical_pipeline.QWEN_CACHE_DIR = (
    DATA_LAB_QWEN_CACHE
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Medical Consultation AI GPU Backend",
    description=(
        "GPU inference backend for Whisper Large-v3 "
        "and Qwen3-8B clinical documentation."
    ),
    version="1.0"
)


# ============================================================
# STREAMLIT-UPLOAD COMPATIBILITY WRAPPER
# ============================================================

class UploadedFileAdapter:

    def __init__(
        self,
        name,
        data
    ):
        self.name = name
        self._data = data

    def getbuffer(self):
        return self._data


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "gpu_available":
            torch.cuda.is_available(),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "project":
            str(PROJECT),

        "qwen_cache":
            str(DATA_LAB_QWEN_CACHE),

        "timestamp":
            datetime.now().isoformat()
    }


# ============================================================
# CREATE + TRANSCRIBE
# ============================================================

@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...)
):

    try:

        payload = await audio.read()

        if not payload:
            raise RuntimeError(
                "Uploaded audio file is empty."
            )

        adapter = UploadedFileAdapter(
            audio.filename or "consultation.wav",
            payload
        )

        encounter = (
            live_pipeline.create_encounter(
                adapter
            )
        )

        encounter_id = encounter[
            "encounter_id"
        ]

        result = (
            live_pipeline.transcribe_encounter(
                encounter_id
            )
        )

        return {
            "success": True,
            "encounter_id":
                encounter_id,

            "transcript":
                result["transcript"],

            "asr_metadata":
                result["metadata"]
        }

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# CLINICAL FACT EXTRACTION
# ============================================================

@app.post("/extract-facts/{encounter_id}")
def extract_facts(
    encounter_id: str
):

    try:

        result = (
            live_clinical_pipeline
            .extract_clinical_facts(
                encounter_id
            )
        )

        return {
            "success": True,
            "encounter_id":
                encounter_id,

            "result":
                result
        }

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# SOAP
# ============================================================

@app.post("/generate-soap/{encounter_id}")
def generate_soap(
    encounter_id: str
):

    try:

        result = (
            live_clinical_pipeline
            .generate_soap(
                encounter_id
            )
        )

        return {
            "success": True,
            "encounter_id":
                encounter_id,

            "result":
                result
        }

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# SOAP FROM HUMAN-REVIEWED FACTS
# ============================================================

@app.post("/generate-soap-reviewed/{encounter_id}")
def generate_soap_reviewed(
    encounter_id: str,
    payload: dict
):

    try:

        facts = payload.get(
            "facts",
            []
        )

        if not isinstance(facts, list) or not facts:
            raise RuntimeError(
                "Reviewed clinical facts are required."
            )

        live_clinical_pipeline.save_reviewed_facts(
            encounter_id,
            facts
        )

        result = (
            live_clinical_pipeline
            .generate_soap(
                encounter_id
            )
        )

        return {
            "success": True,
            "encounter_id": encounter_id,
            "result": result
        }

    except Exception as exc:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )



# ============================================================
# STATUS
# ============================================================

@app.get("/encounter/{encounter_id}")
def encounter_status(
    encounter_id: str
):

    metadata = (
        live_pipeline
        .load_encounter_metadata(
            encounter_id
        )
    )

    if metadata is None:

        raise HTTPException(
            status_code=404,
            detail="Encounter not found."
        )

    return metadata
