# Enterprise RAG Knowledge Assistant - application image.
#
# The image runs BOTH services (FastAPI API + Streamlit UI); docker-compose
# picks which command to run per service.
#
# torch is installed first from the official CPU index so the image stays
# small and CUDA-free (matches the local Windows setup). The GPU profile uses
# a separate vLLM container, not a CUDA build of this image.

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

COPY requirements.txt .

# CPU-only PyTorch first (small download, no CUDA bloat), then the rest.
# Pinned to the same version as requirements.txt so `pip install -r` never
# has to re-resolve torch from PyPI (a multi-GB CUDA build).
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Data + models are mounted at runtime (see docker-compose.yml) so upgrades
# never touch customer documents or weights.
VOLUME ["/app/data", "/app/models"]

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
