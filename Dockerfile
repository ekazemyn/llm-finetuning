FROM python:3.12-slim

ARG INSTALL_VLLM=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CONFIG_PATH=/app/configs/experiment.yaml \
    INFERENCE_BACKEND=transformers \
    LLM_METHOD=baseline \
    QUANTIZATION=none

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --upgrade pip \
    && if [ "$INSTALL_VLLM" = "true" ]; then \
         python -m pip install ".[qlora,vllm]"; \
       else \
         python -m pip install ".[qlora]"; \
       fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "llm_finetuning.api:app", "--host", "0.0.0.0", "--port", "8000"]
