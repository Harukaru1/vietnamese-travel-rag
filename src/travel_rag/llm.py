"""Minimal Gemini REST client with retries and an on-disk response cache.

* Retries on 429 (rate limit) and 5xx (e.g. 503 "model overloaded") with
  exponential backoff plus jitter; other 4xx errors fail immediately because
  retrying a bad request cannot succeed.
* temperature=0 and a cache keyed by (model, request) make evaluation runs
  repeatable and avoid paying twice for the same call.
* The API key is read from GEMINI_API_KEY. In the cloud sandbox a proxy injects
  the real key into the x-goog-api-key header, so a placeholder is enough there.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from travel_rag.config import DATA_DIR

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODEL = "gemini-flash-lite-latest"
RETRYABLE = {429, 500, 502, 503, 504}
CACHE_DIR = DATA_DIR / "llm_cache"


class LLMError(RuntimeError):
    pass


@dataclass
class GeminiClient:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_retries: int = 6
    timeout: int = 90
    use_cache: bool = True
    cache_dir: Path = CACHE_DIR
    stats: dict = field(default_factory=lambda: {"calls": 0, "cache_hits": 0, "retries": 0})

    def _key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "injected-by-proxy")

    def generate(self, prompt: str, system: str | None = None,
                 response_schema: dict | None = None) -> str:
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if response_schema:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = response_schema

        cache_path = self._cache_path(body)
        if self.use_cache and cache_path.exists():
            self.stats["cache_hits"] += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))["text"]

        text = self._post(body)
        if self.use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"model": self.model, "text": text}, ensure_ascii=False),
                                  encoding="utf-8")
        return text

    def generate_json(self, prompt: str, system: str | None, response_schema: dict) -> dict:
        return json.loads(self.generate(prompt, system, response_schema))

    def _post(self, body: dict) -> str:
        url = API_URL.format(model=self.model)
        headers = {"x-goog-api-key": self._key(), "Content-Type": "application/json"}
        for attempt in range(self.max_retries + 1):
            self.stats["calls"] += 1
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as exc:  # network error: retry
                status, detail = None, str(exc)
            else:
                if resp.status_code == 200:
                    return self._extract_text(resp.json())
                status, detail = resp.status_code, resp.text[:300]
                if status not in RETRYABLE:
                    raise LLMError(f"HTTP {status}: {detail}")
            if attempt == self.max_retries:
                raise LLMError(f"gave up after {attempt + 1} attempts (last: {status} {detail})")
            self.stats["retries"] += 1
            time.sleep(min(60, 2 ** attempt) + random.uniform(0, 1))
        raise AssertionError("unreachable")

    @staticmethod
    def _extract_text(data: dict) -> str:
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            reason = data.get("promptFeedback") or data.get("candidates", [{}])[0].get("finishReason")
            raise LLMError(f"no text in response ({reason})") from exc
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def _cache_path(self, body: dict) -> Path:
        digest = hashlib.sha256(json.dumps([self.model, body], sort_keys=True,
                                           ensure_ascii=False).encode()).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"
