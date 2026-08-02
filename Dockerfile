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
      libgomp1 curl ca-certificates \
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

# Resolve the serving bundle, whatever the checkout gave us.
#
# `final_model/model.pkl` is tracked with Git LFS. A local clone with git-lfs installed has
# the real 79 MB pickle here and this step is a no-op. A host that clones WITHOUT fetching
# LFS objects -- which is what Render does -- gets a ~130-byte pointer file instead, and
# that is the dangerous case: `ModelRegistry` catches the unpickle failure and serves in
# no-model mode, so the deploy goes green, the rule engine answers, and every forecast comes
# back empty. A silent half-working deployment is worse than a failed build, so this fetches
# the real object and makes a still-pointer-sized file a hard build failure.
#
# GitHub's media host serves LFS content for public repos with no client and no credentials,
# which is why no LFS binary or token is needed here. MODEL_URL is an ARG with a working
# default so the build does not depend on Render passing service env vars through as build
# args; override it to move the bundle to S3 or a release asset without editing this file.
ARG MODEL_URL=https://media.githubusercontent.com/media/Nivakaran-S/Cardioplace-BP-Alerts-V2/main/final_model/model.pkl
RUN set -eu; \
    sz="$(stat -c%s final_model/model.pkl 2>/dev/null || echo 0)"; \
    if [ "$sz" -lt 1000000 ]; then \
      echo "final_model/model.pkl is ${sz} bytes -- an LFS pointer, not a model. Fetching."; \
      curl -fsSL --retry 3 --retry-delay 2 "$MODEL_URL" -o final_model/model.pkl; \
      sz="$(stat -c%s final_model/model.pkl)"; \
    fi; \
    echo "serving bundle resolved: ${sz} bytes"; \
    if [ "$sz" -lt 10000000 ]; then \
      echo "FATAL: bundle is ${sz} bytes. Refusing to ship an image whose ML layer is dead"; \
      echo "       on arrival while the health check reports green."; \
      exit 1; \
    fi

RUN mkdir -p logs Artifacts

EXPOSE 7860

# One worker. Two would mean two independently hot-reloading predictors and two training
# managers racing for the same single-flight lock; the state is process-local by design.
#
# Shell form so ${PORT} expands: Render assigns the port and expects the service to bind it,
# and a hardcoded 7860 would leave the health check talking to nothing. The default keeps
# `docker run -p 7860:7860` working unchanged for every other host.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
