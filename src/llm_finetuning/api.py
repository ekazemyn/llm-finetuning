"""FastAPI inference service with lazy startup model loading."""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from llm_finetuning.config import load_config
from llm_finetuning.serving import TransformersGenerator, VLLMGenerator


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=30000)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)


class BatchGenerateRequest(BaseModel):
    prompts: list[str] = Field(min_length=1, max_length=16)
    max_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, allow_inf_nan=False)


def _authorized(request: Request) -> None:
    expected = os.getenv("API_KEY")
    if expected:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied, f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="Unauthorized")


def create_app(generator=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if generator is not None:
            app.state.generator = generator
        else:
            config_path = Path(os.getenv("CONFIG_PATH", "configs/experiment.yaml"))
            config = load_config(config_path).raw
            method = os.getenv("LLM_METHOD", "baseline").lower()
            model_path = os.getenv("MODEL_PATH")
            if method in {"lora", "qlora"} and not model_path:
                raise RuntimeError("MODEL_PATH is required for LoRA/QLoRA serving.")
            backend = os.getenv("INFERENCE_BACKEND", "transformers").lower()
            quantization = os.getenv("QUANTIZATION", "4bit" if method == "qlora" else "none")
            if backend == "transformers":
                app.state.generator = TransformersGenerator(
                    config, method, model_path, quantization
                )
            elif backend == "vllm":
                app.state.generator = VLLMGenerator(config, method, model_path, quantization)
            else:
                raise RuntimeError("INFERENCE_BACKEND must be transformers or vllm.")
        yield

    app = FastAPI(title="FinQA LLM Inference", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request):
        loaded = hasattr(request.app.state, "generator")
        return {
            "status": "ready" if loaded else "starting",
            "backend": getattr(request.app.state.generator, "backend", None),
        }

    @app.post("/generate")
    async def generate(body: GenerateRequest, request: Request):
        _authorized(request)
        active_generator = getattr(request.app.state, "generator", None)
        if active_generator is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        outputs = await run_in_threadpool(
            active_generator.generate, [body.prompt], body.max_tokens, body.temperature
        )
        return {"text": outputs[0], "backend": active_generator.backend}

    @app.post("/generate/batch")
    async def generate_batch(body: BatchGenerateRequest, request: Request):
        _authorized(request)
        active_generator = getattr(request.app.state, "generator", None)
        if active_generator is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        if any(not prompt or len(prompt) > 30000 for prompt in body.prompts):
            raise HTTPException(
                status_code=422, detail="Each prompt must contain 1 to 30000 characters"
            )
        outputs = await run_in_threadpool(
            active_generator.generate, body.prompts, body.max_tokens, body.temperature
        )
        return {"texts": outputs, "backend": active_generator.backend}

    return app


app = create_app()
