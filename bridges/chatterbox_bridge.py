from __future__ import annotations

import io
import json
import os
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS

HOST = "127.0.0.1"
PORT = 8771
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = ROOT / "voices" / "gwendolyn_reference.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        print(f"Loading Chatterbox Turbo on {DEVICE}...", flush=True)
        MODEL = ChatterboxTurboTTS.from_pretrained(device=DEVICE)
        print(f"Chatterbox ready at {MODEL.sr} Hz", flush=True)
    return MODEL


def synthesize(text: str, reference: str) -> bytes:
    model = get_model()
    ref = Path(reference) if reference else DEFAULT_REFERENCE
    if not ref.is_absolute():
        ref = ROOT / ref
    if not ref.exists():
        raise FileNotFoundError(
            f"Chatterbox Turbo needs a reference WAV. Put one at {DEFAULT_REFERENCE} "
            "or set the backend voice field to a WAV path."
        )
    wav = model.generate(text, audio_prompt_path=str(ref))
    if model.sr != 24000:
        wav = ta.functional.resample(wav, model.sr, 24000)
        sr = 24000
    else:
        sr = model.sr
    # Chatterbox returns floating-point PCM, normally shaped [1, samples].
    # Avoid torchaudio.save(BytesIO), which is brittle across newer Torch/Torchaudio
    # versions used for RTX 50-series support. Encode a standard PCM16 WAV directly.
    pcm = wav.detach().float().cpu()
    if pcm.ndim == 2:
        if pcm.shape[0] == 1:
            pcm = pcm.squeeze(0)
        elif pcm.shape[1] == 1:
            pcm = pcm.squeeze(1)
        else:
            # torch audio convention is [channels, time]; interleave for WAV.
            pcm = pcm.transpose(0, 1).contiguous()
    pcm = pcm.clamp(-1.0, 1.0)
    pcm16 = (pcm * 32767.0).round().to(torch.int16).contiguous()

    channels = 1 if pcm16.ndim == 1 else int(pcm16.shape[1])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.numpy().tobytes())
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"ok": True, "engine": "chatterbox-turbo", "device": DEVICE}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/audio/speech":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(length) or b"{}")
            text = str(req.get("input") or "").strip()
            reference = str(req.get("voice") or "")
            if not text:
                raise ValueError("input is empty")
            wav = synthesize(text, reference)
            self.send_response(200); self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav))); self.end_headers(); self.wfile.write(wav)
        except Exception as e:
            print("[Chatterbox] synthesis/response error:", flush=True)
            traceback.print_exc()
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, fmt, *args):
        print("[Chatterbox] " + fmt % args, flush=True)


if __name__ == "__main__":
    # Pay model-load and first-generation costs during startup rather than on
    # Gwendolyn's first conversational reply. The HTTP port opens only once warm.
    print("Chatterbox startup warm-up beginning...", flush=True)
    try:
        get_model()
        if DEFAULT_REFERENCE.exists():
            _ = synthesize("Ready.", str(DEFAULT_REFERENCE))
            print("Chatterbox warm-up complete.", flush=True)
        else:
            print(f"Chatterbox warm-up skipped: reference WAV not found at {DEFAULT_REFERENCE}", flush=True)
    except Exception:
        print("Chatterbox warm-up failed; bridge will still start for diagnostics.", flush=True)
        traceback.print_exc()
    print(f"Chatterbox bridge: http://{HOST}:{PORT}/v1/audio/speech", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
