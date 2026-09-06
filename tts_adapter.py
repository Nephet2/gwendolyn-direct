from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
import requests


@dataclass(frozen=True)
class TTSBackend:
    key: str
    label: str
    url: str
    model: str
    voice: str
    sample_rate: int = 24000
    response_format: str = "pcm"
    enabled: bool = True


class TTSAdapter:
    """Small backend-neutral TTS client.

    Gwendolyn owns queueing, barge-in, playback and PCM edge treatment. This
    adapter owns only the synthesis request. Any local engine can participate
    if its bridge accepts POST /v1/audio/speech and returns PCM or WAV bytes.
    """

    def __init__(self, config: Dict[str, Any], session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()

    def backend_key(self) -> str:
        return str(self.config.get("tts_backend") or "f5")

    def backend(self, key: Optional[str] = None) -> TTSBackend:
        key = key or self.backend_key()
        backends = self.config.get("tts_backends") or {}
        raw = dict(backends.get(key) or {})

        # v0.8 compatibility: if no structured F5 entry exists, use the legacy keys.
        if key == "f5" and not raw:
            raw = {
                "label": "F5 baseline",
                "url": self.config.get("tts_url", "http://127.0.0.1:8765/v1/audio/speech"),
                "model": self.config.get("tts_model", "tts-1"),
                "voice": self.config.get("tts_voice", "p239"),
                "sample_rate": self.config.get("tts_sample_rate", 24000),
                "response_format": "pcm",
                "enabled": True,
            }
        if not raw:
            raise KeyError(f"Unknown TTS backend: {key}")

        return TTSBackend(
            key=key,
            label=str(raw.get("label") or key),
            url=str(raw["url"]),
            model=str(raw.get("model") or "tts-1"),
            voice=str(raw.get("voice") or "default"),
            sample_rate=int(raw.get("sample_rate") or 24000),
            response_format=str(raw.get("response_format") or "pcm"),
            enabled=bool(raw.get("enabled", True)),
        )

    def backend_choices(self):
        result = []
        for key, raw in (self.config.get("tts_backends") or {}).items():
            if bool(raw.get("enabled", True)):
                result.append((key, str(raw.get("label") or key)))
        if not result:
            result.append(("f5", "F5 baseline"))
        return result

    def request(self, text: str, voice_id: str, chunk_index: int, *, timeout=(3, 120)):
        backend = self.backend()
        if not backend.enabled:
            raise RuntimeError(f"TTS backend {backend.label} is disabled")
        payload = {
            "model": backend.model,
            "voice": backend.voice,
            "input": text,
            "response_format": backend.response_format,
        }
        headers = {
            "Connection": "close",
            "X-Gwendolyn-Voice": voice_id,
            "X-Gwendolyn-Chunk": str(chunk_index),
            "X-Gwendolyn-Backend": backend.key,
        }
        return self.session.post(backend.url, json=payload, stream=True, timeout=timeout, headers=headers)

    def healthy(self, timeout: float = 1.2) -> bool:
        backend = self.backend()
        try:
            root = backend.url.rsplit("/v1/", 1)[0]
            r = self.session.get(root, timeout=timeout)
            # A local bridge may answer 404/405 at root but still prove the process is alive.
            return r.status_code < 500
        except Exception:
            return False
