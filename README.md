# Medical Consultation AI

AI assisted medical consultation documentation prototype developed for Case Study 2 in the M.Sc. Applied Data Science and Analytics program at SRH Hochschule Heidelberg.

The application converts consultation audio into a structured medical record while keeping a human reviewer in control of the final output.

## Project Workflow

Consultation audio

→ Whisper Large v3 transcription

→ Qwen3 8B clinical fact extraction

→ Evidence grounded clinical facts

→ SOAP draft generation

→ Human review and correction

→ Final approval

→ JSON and PDF export

## Main Features

The application supports:

- Uploading consultation audio
- Recording consultation audio in the browser
- Automatic transcription using Whisper Large v3
- Clinical fact extraction using Qwen3 8B
- Transcript evidence checking
- Clinical fact review and correction
- SOAP note generation
- SOAP section editing
- Moving information between SOAP sections
- Correction logging
- Final human approval
- JSON export
- PDF export
- Saved consultation records

## Dataset

The project uses the PRI MOCK57 dataset.

PRI MOCK57 contains 57 mock primary care consultations with consultation audio, transcripts, clinical notes, and related reference information.

The project dataset was divided into:

- 32 development and optimisation consultations
- 8 internal validation consultations
- 17 held out final test consultations

## Models

### Automatic Speech Recognition

Whisper Large v3 was selected as the final speech recognition model.

Final held out test result:

- Corpus Word Error Rate: 18.07%
- Macro Word Error Rate: 18.23%
- Median Word Error Rate: 18.03%
- Mean Real Time Factor: 0.0452

### Clinical Information Extraction

Qwen3 8B was selected for clinical fact extraction and SOAP generation.

Final structured output result:

- 16 of 17 consultations produced valid structured output
- Structured output success rate: 94.12%
- 308 accepted clinical facts
- 308 of 308 accepted facts contained transcript evidence

Representative manual clinical review:

- Precision: 78.63%
- Recall: 58.97%
- F1 Score: 67.40%

Evidence grounding confirms that supporting transcript text exists. It does not guarantee that every clinical interpretation is correct.

## SOAP Generation

SOAP generation was evaluated on the 10 consultation development pilot.

Before deterministic validation:

- Average source fact coverage: 91.79%
- 16 missing source facts
- 13 wrong section references

After validation:

- Source fact coverage: 100%
- Missing source facts: 0
- Wrong section references: 0
- Invalid fact IDs: 0

The 100% result refers to coverage of the clinical facts already available to the SOAP stage. It does not mean that every fact from the original consultation was extracted.

## Human in the Loop Review

The system is designed as a documentation assistant rather than a fully automatic medical documentation system.

The reviewer can:

- Edit the transcript
- Correct clinical facts
- Add missing clinical facts
- Remove incorrect facts
- Change fact categories and status
- Edit SOAP sections
- Move information between SOAP sections
- Review the final record
- Approve the record before export

Original AI outputs and reviewed outputs are stored separately.

## Project Architecture

The system contains two main deployment components.

### Frontend

- Streamlit web application
- Audio upload and browser recording
- Transcript review
- Clinical fact review
- SOAP review
- Final approval
- PDF and JSON export

### GPU Backend

- FastAPI backend
- Whisper Large v3 inference
- Qwen3 8B clinical fact extraction
- Qwen3 8B SOAP generation
- GPU inference on the university Data Lab Compute environment

## Repository Structure

```text
medical-consultation-ai/
│
├── app/
│   ├── app.py
│   ├── gpu_backend_api.py
│   ├── live_pipeline.py
│   ├── live_clinical_pipeline.py
│   ├── live_export.py
│   ├── remote_live_pipeline.py
│   └── remote_clinical_pipeline.py
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── requirements-frontend.txt
└── README.md
