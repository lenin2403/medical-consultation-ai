# Medical Consultation AI

AI-assisted medical consultation documentation prototype.

## Pipeline

Consultation audio  
→ Whisper Large-v3 transcription  
→ Qwen3-8B clinical fact extraction  
→ evidence-grounded clinical facts  
→ SOAP draft generation  
→ human review and correction  
→ clinician approval  
→ JSON/PDF export

## Architecture

The deployed prototype uses:

- Streamlit frontend
- FastAPI GPU backend
- Whisper Large-v3 for ASR
- Qwen3-8B for clinical extraction and SOAP generation
- Human-in-the-loop review before final approval
- Docker/Portainer frontend deployment
- GPU inference on the university Data Lab Compute environment

## Important

This system is an academic prototype and is not intended for clinical use.
