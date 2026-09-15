# Medical Consultation AI

AI assisted medical consultation documentation prototype developed for Case Study 2 in the M.Sc. Applied Data Science and Analytics program at SRH Hochschule Heidelberg.

The application converts medical consultation audio into a structured medical record while keeping a human reviewer in control of the final output.

## Project Workflow

```text
Consultation Audio
        |
        v
Whisper Large v3 Transcription
        |
        v
Transcript Review
        |
        v
Qwen3 8B Clinical Fact Extraction
        |
        v
Evidence Grounded Clinical Facts
        |
        v
Clinical Fact Review
        |
        v
SOAP Draft Generation
        |
        v
SOAP Review and Correction
        |
        v
Human Approval
        |
        v
JSON and PDF Export
```

## Main Features

The application supports:

- Uploading consultation audio
- Recording consultation audio in the browser
- Automatic transcription using Whisper Large v3
- Clinical fact extraction using Qwen3 8B
- Transcript evidence checking
- Clinical fact review and correction
- Adding missing clinical facts
- Removing incorrect clinical facts
- SOAP note generation
- SOAP section editing
- Moving information between SOAP sections
- Correction logging
- Final human review
- Final approval
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

The complete dataset contains approximately 8.63 hours of consultation audio.

## Models

### Automatic Speech Recognition

Whisper Large v3 was selected as the final automatic speech recognition model.

Final held out test results:

- Corpus Word Error Rate: 18.07%
- Macro Word Error Rate: 18.23%
- Median Word Error Rate: 18.03%
- Minimum consultation WER: 13.34%
- Maximum consultation WER: 25.79%
- Mean Real Time Factor: 0.0452

Final held out corpus statistics:

- Reference words: 27,404
- Hypothesis words: 25,824
- Hits: 23,258
- Substitutions: 1,760
- Deletions: 2,386
- Insertions: 806
- Total ASR errors: 4,952

### Clinical Information Extraction

Qwen3 8B was selected for clinical fact extraction and SOAP generation.

Final structured output results:

- 16 of 17 consultations produced valid structured output
- Structured output success rate: 94.12%
- 308 accepted clinical facts
- 308 of 308 accepted facts contained transcript evidence

Representative manual clinical review results:

- True Positives: 92
- False Positives: 25
- False Negatives: 64
- Precision: 78.63%
- Recall: 58.97%
- F1 Score: 67.40%

Evidence grounding confirms that supporting transcript text exists.

It does not guarantee that every clinical interpretation is medically correct.

## SOAP Generation

SOAP generation was evaluated on the 10 consultation development pilot.

Before deterministic validation:

- Average source fact coverage: 91.79%
- 16 missing source facts
- 13 wrong section references
- Traceability: 98.9%

After deterministic validation:

- Source fact coverage: 100%
- Missing source facts: 0
- Wrong section references: 0
- Invalid fact IDs: 0
- Restored source facts: 16
- Invalid statements removed: 2

The 100% source fact coverage result refers only to clinical facts that entered the SOAP generation stage.

It does not mean that every fact from the original consultation was extracted successfully.

## Human in the Loop Review

The system is designed as a documentation assistant rather than a fully automatic medical documentation system.

The reviewer can:

- Edit the transcript
- Review AI extracted clinical facts
- Correct clinical facts
- Add missing clinical facts
- Remove incorrect facts
- Change fact categories
- Change fact status
- Review supporting transcript evidence
- Edit SOAP sections
- Move information between SOAP sections
- Review the final structured record
- Approve the final record
- Export the approved record

Original AI generated outputs and reviewer approved outputs are stored separately.

## Project Architecture

The final Case Study 2 demonstration uses a hybrid architecture.

```text
Local MacBook
     |
     | Streamlit Frontend
     |
     v
Cloudflare HTTPS Tunnel
     |
     v
FastAPI GPU Backend
     |
     v
University Data Lab Compute
     |
     v
NVIDIA H200 NVL GPU
     |
     +---- Whisper Large v3
     |
     +---- Qwen3 8B
```

### Frontend

The Streamlit frontend runs locally on the demonstration computer.

The frontend provides:

- Audio upload
- Browser audio recording
- Transcript display and review
- Clinical fact review
- SOAP note review
- Human correction
- Final approval
- JSON export
- PDF export

### GPU Backend

The GPU backend runs remotely on university Data Lab Compute.

The backend provides:

- FastAPI API
- Backend health checking
- Whisper Large v3 inference
- Qwen3 8B clinical fact extraction
- Qwen3 8B SOAP generation
- CUDA GPU inference

The frontend communicates with the backend through a temporary Cloudflare HTTPS tunnel.

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
├── scripts/
│   └── final_evaluation/
│
├── notebooks/
│
├── documentation/
│   └── final_evaluation/
│
├── results/
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── requirements-frontend.txt
└── README.md
```

## Case Study 2 Demo Deployment

For Case Study 2, the application is demonstrated from a local computer.

The Streamlit frontend runs locally on the MacBook while the computationally intensive AI models run remotely on the university Data Lab GPU environment.

The final demonstrated architecture is:

```text
MacBook Streamlit Frontend
          |
          v
   Cloudflare Tunnel
          |
          v
     FastAPI Backend
          |
          v
 University Data Lab
          |
          v
   NVIDIA H200 GPU
```

Testing confirmed:

- Local Streamlit frontend started successfully
- Data Lab Compute environment started successfully
- CUDA was available
- NVIDIA H200 NVL GPU was available
- FastAPI backend started successfully
- Backend health endpoint returned successfully
- Cloudflare public connection worked successfully
- Whisper Large v3 completed live transcription
- Qwen3 8B completed live clinical fact extraction
- SOAP generation completed successfully
- Human review functions worked
- SOAP section movement and editing worked
- Final approval completed successfully
- Approved JSON export completed successfully
- Approved PDF export completed successfully

Portainer is not required for the Case Study 2 demonstration.

Docker and GitHub Actions are retained as additional containerization and reproducibility components.

## Docker and GitHub Actions

A frontend Docker image is built through GitHub Actions.

The frontend container image is available from GitHub Container Registry:

```text
ghcr.io/lenin2403/medical-consultation-ai-frontend:latest
```

Docker containerization is included for reproducibility and future deployment options.

It is not required for the final Case Study 2 local demonstration.

## Running the Frontend

Clone the repository:

```bash
git clone https://github.com/lenin2403/medical-consultation-ai.git
```

Enter the project directory:

```bash
cd medical-consultation-ai
```

Install the frontend requirements:

```bash
python3 -m pip install -r requirements-frontend.txt
```

The frontend requires the URL of the running GPU backend.

Start Streamlit using the current Cloudflare backend URL:

```bash
MEDICAL_AI_BACKEND_URL="https://YOUR-CLOUDFLARE-URL.trycloudflare.com" \
python3 -m streamlit run app/app.py
```

The local Streamlit application is normally available at:

```text
http://localhost:8501
```

The Cloudflare quick tunnel URL is temporary.

The current URL must therefore be supplied each time a new backend tunnel is started.

## Running the GPU Backend

The GPU backend is implemented in:

```text
app/gpu_backend_api.py
```

The demonstrated backend was run on the university Data Lab Compute environment using an NVIDIA H200 NVL GPU.

Enter the application directory:

```bash
cd /home/jovyan/Case_Study_2_Medical_Consultation_AI/app
```

Start FastAPI:

```bash
python -m uvicorn gpu_backend_api:app --host 0.0.0.0 --port 8000
```

The backend will then be available locally on the Compute environment at:

```text
http://127.0.0.1:8000
```

The backend provides functionality for:

- Health checking
- Speech transcription
- Clinical fact extraction
- SOAP generation

## Backend Health Check

The backend can be checked using:

```bash
curl http://127.0.0.1:8000/health
```

A successful response confirms that the API is running.

When GPU resources are available, the response also reports the active CUDA GPU.

## Cloudflare Connection

A temporary Cloudflare tunnel is used to make the remote FastAPI backend accessible to the local Streamlit application.

After starting FastAPI, start Cloudflare using:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflare generates a temporary HTTPS URL similar to:

```text
https://example-name.trycloudflare.com
```

This URL is supplied to the frontend through:

```text
MEDICAL_AI_BACKEND_URL
```

The temporary Cloudflare URL changes when the tunnel is restarted.

## AI Model Setup

### Whisper Large v3

Whisper Large v3 is loaded by the backend when transcription is requested.

The model may be downloaded automatically the first time the transcription pipeline is executed.

### Qwen3 8B

The demonstrated clinical extraction model uses the 4 bit Qwen3 8B model:

```text
unsloth/Qwen3-8B-bnb-4bit
```

The model cache is stored under:

```text
models/qwen3_8b_4bit_cache
```

On a fresh Data Lab Compute session, the model may need to be downloaded again because the Compute environment does not guarantee permanent storage.

## Fresh Data Lab Session

A new Data Lab Compute session can be prepared by cloning the repository:

```bash
cd /home/jovyan
git clone https://github.com/lenin2403/medical-consultation-ai.git Case_Study_2_Medical_Consultation_AI
```

Install the required backend packages if they are not already available:

```bash
python -m pip install faster-whisper==1.2.1 fastapi uvicorn python-multipart transformers accelerate bitsandbytes huggingface_hub
```

Then start the backend from:

```bash
cd /home/jovyan/Case_Study_2_Medical_Consultation_AI/app
python -m uvicorn gpu_backend_api:app --host 0.0.0.0 --port 8000
```

## Downloading the Qwen Deployment Model

If the Qwen model cache is not available, it can be downloaded using Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="unsloth/Qwen3-8B-bnb-4bit",
    cache_dir="/home/jovyan/Case_Study_2_Medical_Consultation_AI/models/qwen3_8b_4bit_cache"
)
```

After the download completes, the clinical extraction pipeline can load the model from the project cache.

## Final Demonstrated Workflow

The complete end to end application workflow was successfully tested using consultation audio containing both doctor and patient speech.

```text
Consultation Audio
        |
        v
Whisper Large v3
        |
        v
Transcript
        |
        v
Transcript Review
        |
        v
Qwen3 8B
        |
        v
Clinical Facts
        |
        v
Clinical Fact Review
        |
        v
SOAP Generation
        |
        v
SOAP Review
        |
        v
Human Approval
        |
        v
Approved Medical Record
        |
        +---- JSON
        |
        +---- PDF
```

## Export System

After the reviewer approves the final medical record, the application can generate:

- Approved medical record in JSON format
- Approved medical record in PDF format

The export system uses a project relative path so that the application can work both on the local MacBook and in the Data Lab environment.

## Example Demonstration Audio

The final application test used synchronized doctor and patient consultation audio combined into a single consultation recording.

The complete pipeline successfully processed the combined consultation from audio input through final approved PDF export.

## Deployment Note

The university Data Lab Compute environment uses shared GPU resources.

The environment is not guaranteed to remain permanently active and local storage should not be treated as permanent storage.

Important results, code, model configuration information, and evaluation outputs should therefore also be stored outside the active Compute session.

For a live demonstration:

1. Start the Data Lab GPU Compute session.
2. Clone or open the project repository.
3. Confirm required Python packages are installed.
4. Confirm the Qwen model cache is available.
5. Start the FastAPI backend.
6. Start the Cloudflare tunnel.
7. Copy the generated Cloudflare HTTPS URL.
8. Start the local Streamlit frontend using that backend URL.
9. Open the Streamlit interface at `http://localhost:8501`.
10. Upload or record a consultation and run the workflow.

## Limitations

The prototype has several important limitations.

- ASR errors can affect downstream clinical information extraction.
- Evidence grounding does not guarantee clinical correctness.
- Clinical fact recall is not perfect.
- The final test set contains mock consultations rather than real clinical deployments.
- GPU inference depends on access to suitable hardware.
- Data Lab Compute sessions are temporary.
- Cloudflare quick tunnel URLs change between sessions.
- The application requires human review before final approval.
- The system has not been validated for real clinical use.

## Safety

This project is an academic prototype.

It is not intended for clinical use.

AI generated documentation must be reviewed by a human reviewer before final approval.

Evidence grounding does not guarantee that an extracted medical statement is clinically correct.

The system does not independently diagnose patients.

The system does not independently prescribe treatment.

The system does not replace a qualified healthcare professional.

## Case Study Status

The final implementation demonstrates the complete intended prototype workflow:

```text
Audio
→ Transcription
→ Clinical Fact Extraction
→ SOAP Generation
→ Human Review
→ Human Approval
→ PDF and JSON Export
```

The final application was demonstrated using a local Streamlit frontend connected to a GPU enabled FastAPI backend running on university Data Lab Compute.

## Author

Lenin Carvalho

M.Sc. Applied Data Science and Analytics

SRH Hochschule Heidelberg

Case Study 2

2026
