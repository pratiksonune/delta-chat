from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Rough $ per 1K tokens, input/output, for cost estimation only.
COST_TABLE = {
    "mock": (0.0, 0.0),
    "claude-sonnet-5": (0.003, 0.015),
    "claude-haiku-4-5": (0.0008, 0.004),
    "gemma-2-2b-it": (0.0, 0.0),        # local inference, no per-token $ cost
    "phi-4-mini-instruct": (0.0, 0.0),  # local inference, no per-token $ cost
}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = COST_TABLE.get(model, (0.0, 0.0))
    return round((input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate, 6)


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    provider: str


class LLMProvider(ABC):
    name: str = "base"
    model: str = "base"

    @abstractmethod
    def complete(self, system: str, prompt: str) -> LLMResult:
        raise NotImplementedError


class MockProvider(LLMProvider):
    """Deterministic, offline, zero-cost. Answers by extracting and
    lightly reformatting the retrieved context rather than "generating"
    free-form prose -- this keeps eval scores reproducible and keeps the
    default `make chat` usable without any credentials."""

    name = "mock"
    model = "mock"

    def complete(self, system: str, prompt: str) -> LLMResult:
        start = time.time()
        context_match = re.search(r"CONTEXT:\n(.*?)\nQUESTION:", prompt, re.DOTALL)
        question_match = re.search(r"QUESTION:\n(.*)", prompt, re.DOTALL)
        context = context_match.group(1).strip() if context_match else ""
        question = question_match.group(1).strip() if question_match else prompt

        snippets = [line.strip() for line in context.splitlines() if line.strip()]
        top_snippets = snippets[:3]
        if top_snippets:
            body = " ".join(top_snippets)
            answer = f"Based on the retrieved context: {body}"
        else:
            answer = "I couldn't find grounded context in the two PIDs or the delta report to answer this."

        latency_ms = (time.time() - start) * 1000
        in_tok = estimate_tokens(system + prompt)
        out_tok = estimate_tokens(answer)
        return LLMResult(
            text=answer,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(self.model, in_tok, out_tok),
            latency_ms=latency_ms,
            provider=self.name,
        )


class HFLocalProvider(LLMProvider):

    name = "hf_local"

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model = model_name or os.environ.get("LLM_MODEL_NAME", "google/gemma-2-2b-it")
        self.device = device or os.environ.get("LLM_DEVICE", "cpu")
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError(
                "hf_local provider requires `torch` and `transformers`. "
                "Install with: pip install torch transformers accelerate "
                "(see requirements-local-llm.txt)."
            ) from e
        self._pipe = pipeline(
            "text-generation",
            model=self.model,
            device_map=self.device,
        )

    def complete(self, system: str, prompt: str) -> LLMResult:
        self._load()
        start = time.time()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        out = self._pipe(messages, max_new_tokens=400, do_sample=False)
        text = out[0]["generated_text"]
        if isinstance(text, list):  # chat-style pipelines return the message list back
            text = text[-1]["content"]
        latency_ms = (time.time() - start) * 1000
        in_tok = estimate_tokens(system + prompt)
        out_tok = estimate_tokens(text)
        return LLMResult(
            text=text,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(self.model, in_tok, out_tok),
            latency_ms=latency_ms,
            provider=self.name,
        )


class AnthropicProvider(LLMProvider):

    name = "anthropic"

    def __init__(self, model_name: str | None = None):
        self.model = model_name or os.environ.get("LLM_MODEL_NAME", "claude-sonnet-5")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")

    def complete(self, system: str, prompt: str) -> LLMResult:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).")
        import requests

        start = time.time()
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 800,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        in_tok = usage.get("input_tokens", estimate_tokens(system + prompt))
        out_tok = usage.get("output_tokens", estimate_tokens(text))
        latency_ms = (time.time() - start) * 1000
        return LLMResult(
            text=text,
            model=self.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=estimate_cost(self.model, in_tok, out_tok),
            latency_ms=latency_ms,
            provider=self.name,
        )


def get_provider(provider_name: str | None = None) -> LLMProvider:
    name = provider_name or os.environ.get("LLM_PROVIDER", "mock")
    if name == "mock":
        return MockProvider()
    if name == "hf_local":
        return HFLocalProvider()
    if name == "anthropic":
        return AnthropicProvider()
    raise ValueError(f"Unknown LLM_PROVIDER: {name!r}. Expected mock | hf_local | anthropic.")
