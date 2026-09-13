FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /home/jovyan/Case_Study_2_Medical_Consultation_AI

COPY requirements-frontend.txt .

RUN pip install --no-cache-dir \
    -r requirements-frontend.txt

COPY app ./app

RUN mkdir -p \
    results/live/encounters \
    documentation/project_freezes

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
