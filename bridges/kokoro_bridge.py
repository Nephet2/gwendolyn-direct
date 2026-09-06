from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import soundfile as sf
from kokoro import KPipeline

HOST = "127.0.0.1"
PORT = 8770
SAMPLE_RATE = 24000
PIPELINES = {}


def pipeline_for(voice: str):
    # Kokoro convention: af/am = American, bf/bm = British.
    lang = "b" if (voice or "").lower().startswith(("bf_", "bm_")) else "a"
    if lang not in PIPELINES:
        PIPELINES[lang] = KPipeline(lang_code=lang)
    return PIPELINES[lang]


def synthesize(text: str, voice: str) -> bytes:
    pipe = pipeline_for(voice)
    audio_parts = []
    for _graphemes, _phonemes, audio in pipe(text, voice=voice, speed=1.0):
        audio_parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
    if not audio_parts:
        raise RuntimeError("Kokoro returned no audio")
    audio = np.concatenate(audio_parts)
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ok": True, "engine": "kokoro", "sample_rate": SAMPLE_RATE}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = str(req.get("input") or "").strip()
            voice = str(req.get("voice") or "af_heart")
            if not text:
                raise ValueError("input is empty")
            wav = synthesize(text, voice)
            self.send_response(200); self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav))); self.end_headers(); self.wfile.write(wav)
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("[Kokoro] " + fmt % args, flush=True)


if __name__ == "__main__":
    print(f"Kokoro bridge: http://{HOST}:{PORT}/v1/audio/speech")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
