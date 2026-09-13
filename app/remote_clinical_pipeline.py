
import os
import json

import requests

import live_clinical_pipeline as local
from live_clinical_pipeline import *


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


def _backend_encounter(encounter_id):

    metadata = local.load_encounter_metadata(
        encounter_id
    )

    if not metadata:
        raise RuntimeError(
            "Encounter metadata not found."
        )

    remote_id = metadata.get(
        "remote_backend_encounter_id"
    )

    if not remote_id:
        raise RuntimeError(
            "Generate the Whisper transcript first."
        )

    return remote_id


def extract_clinical_facts(encounter_id):

    remote_id = _backend_encounter(
        encounter_id
    )

    response = requests.post(
        (
            _backend_url()
            + "/extract-facts/"
            + remote_id
        ),
        timeout=1800
    )

    response.raise_for_status()

    canonical = response.json()["result"]

    canonical["encounter_id"] = encounter_id

    facts_dir = (
        local._encounter_dir(encounter_id)
        / "clinical_facts"
    )

    facts_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        facts_dir
        / "ai_clinical_facts.json"
    )

    path.write_text(
        json.dumps(
            canonical,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    metadata = local.load_encounter_metadata(
        encounter_id
    )

    metadata["pipeline"][
        "clinical_fact_extraction"
    ] = "complete"

    metadata["clinical_facts"] = str(
        path.relative_to(local.PROJECT)
    )

    metadata["clinical_fact_count"] = len(
        canonical.get(
            "clinical_facts",
            []
        )
    )

    metadata["updated_at"] = local._now()

    local.update_encounter_metadata(
        encounter_id,
        metadata
    )

    return canonical


def generate_soap(encounter_id):

    facts_document = local.load_live_facts(
        encounter_id,
        prefer_reviewed=True
    )

    if not facts_document:
        raise RuntimeError(
            "Extract clinical facts first."
        )

    facts = facts_document.get(
        "clinical_facts",
        []
    )

    if not facts:
        raise RuntimeError(
            "No reviewed clinical facts available."
        )

    remote_id = _backend_encounter(
        encounter_id
    )

    response = requests.post(
        (
            _backend_url()
            + "/generate-soap-reviewed/"
            + remote_id
        ),
        json={
            "facts": facts
        },
        timeout=1800
    )

    response.raise_for_status()

    canonical = response.json()["result"]

    canonical["encounter_id"] = encounter_id

    soap_dir = (
        local._encounter_dir(encounter_id)
        / "soap"
    )

    soap_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    path = (
        soap_dir
        / "ai_soap_draft.json"
    )

    path.write_text(
        json.dumps(
            canonical,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    metadata = local.load_encounter_metadata(
        encounter_id
    )

    metadata["pipeline"][
        "soap_generation"
    ] = "complete"

    metadata["soap_draft"] = str(
        path.relative_to(local.PROJECT)
    )

    metadata["updated_at"] = local._now()

    local.update_encounter_metadata(
        encounter_id,
        metadata
    )

    return canonical
