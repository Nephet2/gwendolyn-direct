from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent


def mark(ok: bool, label: str, detail: str) -> None:
    print(f"[{'OK' if ok else '!!'}] {label}: {detail}")


def read_config() -> dict:
    path = ROOT / "config.json"
    if not path.exists():
        path = ROOT / "config.example.json"
    return json.loads(path.read_text(encoding="utf-8"))


def get_json(url: str, timeout: float = 1.5) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Gwendolyn-Setup-Check"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def tcp_open(url: str, timeout: float = 1.0) -> bool:
    parsed = urlsplit(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    with socket.create_connection((parsed.hostname or "127.0.0.1", port), timeout=timeout):
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a Gwendolyn Direct installation without sending prompts or controls.")
    parser.add_argument("--offline", action="store_true", help="check files and Python packages only")
    args = parser.parse_args()

    print("Gwendolyn Direct setup check\n")
    hard_failure = False
    python_ok = sys.version_info >= (3, 11)
    mark(python_ok, "Python", sys.version.split()[0])
    hard_failure |= not python_ok

    for name in ("gwendolyn_direct.py", "config.example.json", "director_profile.example.json", "director_voices.example.json"):
        ok = (ROOT / name).is_file()
        mark(ok, name, "present" if ok else "missing")
        hard_failure |= not ok

    for module in ("requests", "numpy", "sounddevice", "faster_whisper"):
        ok = importlib.util.find_spec(module) is not None
        mark(ok, f"Python package {module}", "installed" if ok else "missing; run SETUP_GWENDOLYN.bat")
        hard_failure |= not ok

    if args.offline:
        return 1 if hard_failure else 0

    cfg = read_config()
    backend = str(cfg.get("llm_backend") or "ollama").lower()
    if backend == "openai":
        base = str(cfg.get("openai_base") or "http://127.0.0.1:8091/v1").rstrip("/")
        url = base + "/models"
        expected = str(cfg.get("openai_model") or "")
    else:
        base = str(cfg.get("ollama_base") or "http://127.0.0.1:11434").rstrip("/")
        url = base + "/api/tags"
        expected = str(cfg.get("ollama_model") or "")
    try:
        payload = get_json(url)
        ids = []
        for item in payload.get("data", payload.get("models", [])):
            if isinstance(item, dict):
                ids.append(str(item.get("id") or item.get("model") or item.get("name") or ""))
        found = not expected or expected in ids or any(x.startswith(expected + ":") for x in ids)
        mark(found, "Model server", f"reachable at {base}; configured model {expected!r}" + (" found" if found else " was not listed"))
    except Exception as exc:
        mark(False, "Model server", f"not reachable at {base} ({type(exc).__name__})")

    vector_base = str(cfg.get("vector_base") or "http://127.0.0.1:11436").rstrip("/")
    try:
        state = get_json(vector_base + "/v1/state")
        mark(isinstance(state, dict), "Vector Director API", f"reachable at {vector_base}")
    except Exception as exc:
        mark(False, "Vector Director API", f"not reachable at {vector_base} ({type(exc).__name__})")

    tts_key = str(cfg.get("tts_backend") or "kokoro")
    tts = dict((cfg.get("tts_backends") or {}).get(tts_key) or {})
    if not tts or not bool(tts.get("enabled", True)):
        mark(True, "Speech output", "disabled or not configured")
    else:
        tts_url = str(tts.get("url") or "")
        try:
            mark(tcp_open(tts_url), "Speech output", f"{tts_key} bridge is listening")
        except Exception:
            mark(False, "Speech output", f"{tts_key} bridge is not running; text chat still works")

    print("\nThe network checks above are read-only. No model prompt or Vector command was sent.")
    return 1 if hard_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
