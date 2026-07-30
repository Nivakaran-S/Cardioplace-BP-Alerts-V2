# Hugging Face Spaces / any container host.
#
# The Docker SDK rather than the Gradio SDK: the UI is a hand-written vanilla-JS SPA served
# by FastAPI, so there is no gradio to give an sdk_version for.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY templates/ ./templates/
COPY data_schema/ ./data_schema/
COPY app.py main.py ./

# data/ (251 MB) is deliberately NOT copied. The API needs the bundle, not the corpus, and
# registry.detect_pulse() degrades gracefully when vip.csv is absent -- it logs that pulse
# is unavailable and leaves the HR-gated rules BLOCKED_ON_INPUTS, which is the same verdict
# it reaches with the real file, because HEMOBP carries no pulse column either way.
COPY final_model/ ./final_model/

RUN mkdir -p logs Artifacts

EXPOSE 7860

# One worker. Two would mean two independently hot-reloading predictors and two training
# managers racing for the same single-flight lock; the state is process-local by design.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
