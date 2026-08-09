"""LLM backends.

Three pluggable backends, chosen once at startup via ``RAG_LLM_BACKEND``:

* ``transformers`` (default) - in-process Hugging Face generation. Works on any
  CPU, zero extra services. We generate directly with ``AutoModelForSeq2SeqLM``
  + ``AutoTokenizer`` instead of the ``pipeline()`` API: the ``text2text``
  pipeline task was removed in Transformers v5, and the direct classes are a
  stable API across releases. Weights are downloaded once by
  ``scripts/download_model.py`` into ``models/`` (``_resolve_model_path``);
  every run is then fully offline.

* ``openai`` - any OpenAI-compatible inference server:
    - Ollama (CPU/GPU):  set ``RAG_LLM_BASE_URL`` to the Ollama host and
      ``RAG_LLM_MODEL`` to an installed tag (e.g. ``qwen2.5:7b-instruct-q4_K_M``).
    - vLLM (NVIDIA GPU): set ``RAG_LLM_BASE_URL`` to the vLLM server and
      ``RAG_LLM_MODEL`` to the served model name
      (e.g. ``Qwen/Qwen2.5-7B-Instruct-AWQ``).

Both Ollama and vLLM speak the OpenAI chat-completions protocol, so one client
covers them. ``RAG_LLM_API_KEY`` is sent as a bearer token when set (vLLM's
default dev key is ``EMPTY``).
"""

from __future__ import annotations

from typing import Protocol

import httpx
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.config import settings


class LLMBackend(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...

    def __call__(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# Backend 1: in-process Transformers (default)
# ---------------------------------------------------------------------------

def _resolve_model_path() -> str:
    """Prefer a locally downloaded copy (models/<name>) over the Hub id."""
    local = settings.ROOT_DIR / "models" / settings.LLM_MODEL.split("/")[-1]
    if (local / "model.safetensors").exists():
        return str(local)
    return settings.LLM_MODEL


class TransformersLLM:
    """Minimal seq2seq generation wrapper (e.g. FLAN-T5)."""

    name = "transformers"

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or _resolve_model_path()
        self.device = settings.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,  # room for context + question + answer prefix
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=settings.MAX_NEW_TOKENS,
            do_sample=False,  # greedy decoding = deterministic, reproducible
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


# Backwards-compatible alias.
LocalLLM = TransformersLLM


# ---------------------------------------------------------------------------
# Backend 2: OpenAI-compatible server (Ollama / vLLM)
# ---------------------------------------------------------------------------

class OpenAICompatLLM:
    """Client for any OpenAI-compatible /v1/chat/completions endpoint."""

    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        # vLLM ships with the development key "EMPTY"; Ollama ignores the key.
        self.api_key = api_key or settings.LLM_API_KEY or "EMPTY"
        self.model = model or settings.LLM_MODEL

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": settings.MAX_NEW_TOKENS,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)


_llm: LLMBackend | None = None


def get_llm() -> LLMBackend:
    """Return the process-wide LLM backend (loaded once, reused everywhere)."""
    global _llm
    if _llm is None:
        if settings.LLM_BACKEND == "openai":
            _llm = OpenAICompatLLM()
        else:
            _llm = TransformersLLM()
    return _llm
