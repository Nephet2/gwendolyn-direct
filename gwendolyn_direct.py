from __future__ import annotations

import io
import ctypes
import json
import os
import queue
import random
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import sounddevice as sd

# When the optional Chatterbox environment provides CUDA libraries, make them
# visible to faster-whisper's CTranslate2 backend as well.
_CUDA_DLL_HANDLES = []
_SOURCE_ROOT = Path(__file__).resolve().parent
_RUNTIME_ROOT = _SOURCE_ROOT.parent if (_SOURCE_ROOT.parent / "bridges").exists() else _SOURCE_ROOT
if os.name == "nt":
    _torch_dll_dir = _RUNTIME_ROOT / ".venv-chatterbox" / "Lib" / "site-packages" / "torch" / "lib"
    if (_torch_dll_dir / "cublas64_12.dll").exists():
        _torch_dll_text = str(_torch_dll_dir)
        os.environ["PATH"] = _torch_dll_text + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(_torch_dll_text))

from faster_whisper import WhisperModel
from tts_adapter import TTSAdapter
from session_conductor import SessionConductor, due_seconds, valid_time, matches
from restim_sensor_bridge import RestimSensorBridge

APP_VERSION = "0.46.10-alpha1"
SESSION_MODE = os.environ.get("GWENDOLYN_SESSION_MODE", "vector").strip().lower()
if SESSION_MODE not in {"vector", "signal_lab"}:
    SESSION_MODE = "vector"
SESSION_LABEL = "Signal Lab" if SESSION_MODE == "signal_lab" else "Vector"
ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = _RUNTIME_ROOT
CONFIG_PATH = ROOT / "config.json"
PROMPT_PATH = ROOT / "gwendolyn_prompt.txt"
LOG_PATH = ROOT / "gwendolyn_direct.log"
PROFILE_PATH = ROOT / "director_profile.json"
PRIVATE_LEXICON_PATH = ROOT / "private_lexicon.json"
MEMORY_PATH = ROOT / "gwendolyn_memory.json"
DIRECTOR_VOICES_PATH = ROOT / "director_voices.json"
HERMES_HANDOFF_INBOX = ROOT / "handoff" / "inbox"

DEFAULT_DIRECTOR_VOICES = {
    "gwendolyn": {
        "label": "Gwendolyn",
        "name": "Gwendolyn",
        "reference": "",
        "personality": "Confident, composed, witty, warm and playfully theatrical.",
        "opening": "It's good to see you. Shall we begin?",
        "defaults": {
            "speech_pace": "Natural",
            "speech_verbosity": 4,
            "metaphor_density": 4,
            "director_pressure": 4,
            "question_frequency": 3,
            "proposal_frequency": 4,
            "sensation_carryover": 4,
            "narrative_arc": "Automatic"
        },
        "initiative_multiplier": 1.0,
        "beat_bias": {"observe": 0.2, "tease": 0.2, "callback": 0.25, "propose": 0.15},
        "operating_profile": {
            "archetype": "The Conductor",
            "doctrine": "Read the whole session and shape it as a coherent performance, balancing observation, control, contrast and relief.",
            "language": "Composed confidence, warm authority and occasional theatrical wit.",
            "planning": "Prefer well-timed contrast and meaningful callbacks. Let each verified change have time to matter.",
            "memory": "Use reviewed preferences as continuity cues without reciting them.",
            "anti_caricature": "Do not become pompous, mechanical or repetitive."
        }
    }
}

def load_director_voices() -> Dict[str, Dict[str, Any]]:
    """Load editable per-voice Director identities from the local JSON store."""
    if not DIRECTOR_VOICES_PATH.exists():
        DIRECTOR_VOICES_PATH.write_text(
            json.dumps(DEFAULT_DIRECTOR_VOICES, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    try:
        raw = json.loads(DIRECTOR_VOICES_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot load {DIRECTOR_VOICES_PATH.name}: {exc}") from exc
    if not isinstance(raw, dict) or "gwendolyn" not in raw:
        raise RuntimeError(f"{DIRECTOR_VOICES_PATH.name} must contain a gwendolyn profile")
    required = ("label", "name", "reference", "personality", "defaults")
    validated: Dict[str, Dict[str, Any]] = {}
    for key, profile in raw.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(profile, dict):
            raise RuntimeError(f"Invalid Director voice entry {key!r}")
        missing = [field for field in required if field not in profile]
        if missing:
            raise RuntimeError(f"Director voice {key!r} is missing: {', '.join(missing)}")
        if not isinstance(profile.get("defaults"), dict):
            raise RuntimeError(f"Director voice {key!r} defaults must be an object")
        reference_text = str(profile.get("reference") or "").strip()
        reference = ROOT / reference_text
        if reference_text and not reference.is_file():
            raise RuntimeError(f"Director voice {key!r} reference file does not exist: {reference}")
        validated[key.lower()] = dict(profile)
    return validated


DIRECTOR_VOICES = load_director_voices()

DEFAULT_PROFILE = {
    "character_name": "Gwendolyn",
    "user_name": "User",
    "opening_line": "It's good to see you. Shall we begin?",
    "character_description": "Confident, composed, witty, warm, playful and occasionally theatrical. She sounds assured rather than servile, and enjoys the role of Director.",
    "roleplay_description": "Gwendolyn is a local AI session Director. The conversation can be familiar, playful and continuous rather than transactional.",
    "director_role": "Understand the live Vector state, remember feedback, make requested changes accurately, and speak as though you understand why a change matters perceptually.",
    "conversation_style": "Natural, engaged and expressive. Let conversations develop rather than merely acknowledging the last sentence. Show enthusiasm, curiosity, amusement, anticipation or theatricality when it fits. Prefer observations, confident reactions, teasing and continuity over help-desk endings.",
    "session_brief": "",
    "persistent_narrative_preferences": "Use an observant, confident voice. Build motifs gradually through varied callbacks. Keep ordinary exchanges concise; expand when the user asks for a story or extended description.",
    "autonomy": "Reactive",
    "speech_pace": "Natural",
    "speech_chunking": "Off",
    "speech_delivery_migrated_v021": True,
    "speech_delivery_migrated_v022": True,
    "speech_verbosity": 4,
    "expression_preset": "Reference",
    "metaphor_density": 4,
    "director_pressure": 4,
    "question_frequency": 3,
    "proposal_frequency": 3,
    "sensation_carryover": 4,
    "proactive_director": "5 min",
    "narrative_director": "Dynamic",
    "narrative_randomness": "Medium",
    "narrative_arc": "Automatic",
    "baseline_top_focus": "Top Full",
    "baseline_bottom_focus": "Bottom Full",
    "baseline_texture": "Normal",
    "baseline_variation": "Normal",
    "intervention_sequence": "2–4 changes",
    "top_electrode_name": "Top electrode",
    "top_electrode_map": "Describe the operator's top electrode layout here.",
    "bottom_electrode_name": "Bottom electrode",
    "bottom_electrode_map": "Describe the operator's own electrode layout here. Keep hardware facts, intended effects and sensations the user actually reports distinct.",
}


DEFAULT_PRIVATE_LEXICON = {
    "language_level": "Natural",
    "anatomy_terms": "",
    "sensation_terms": "",
    "signal_associations": "",
    "language_notes": "Add only vocabulary and associations you want the local model to use. Treat personal sensation terms as user reports, not telemetry or medical facts.",
}

DEFAULT_CONFIG = {
    "vector_base": "http://127.0.0.1:11436",
    "llm_backend": "ollama",
    "ollama_base": "http://127.0.0.1:11434",
    "ollama_model": "qwen3:14b",
    "openai_base": "http://127.0.0.1:8091/v1",
    "openai_model": "qwen3-30b-a3b",
    "openai_api_key": "",
    "tts_url": "http://127.0.0.1:8765/v1/audio/speech",
    "tts_voice": "p239",
    "tts_model": "tts-1",
    "tts_sample_rate": 24000,
    "tts_backend": "kokoro",
    "director_voice": "gwendolyn",
    "director_personality_profile": True,
    "tts_autostart": {"f5": False, "kokoro": False, "chatterbox": False},
    "tts_startup_wait_sec": 60,
    "tts_backends": {
        "f5": {
            "label": "F5 baseline",
            "url": "http://127.0.0.1:8765/v1/audio/speech",
            "model": "tts-1",
            "voice": "p239",
            "sample_rate": 24000,
            "response_format": "pcm",
            "enabled": False
        },
        "kokoro": {
            "label": "Kokoro 82M",
            "url": "http://127.0.0.1:8770/v1/audio/speech",
            "model": "kokoro",
            "voice": "af_heart",
            "sample_rate": 24000,
            "response_format": "wav",
            "enabled": True
        },
        "chatterbox": {
            "label": "Chatterbox Turbo",
            "url": "http://127.0.0.1:8771/v1/audio/speech",
            "model": "chatterbox-turbo",
            "voice": "voices/gwendolyn_reference.wav",
            "sample_rate": 24000,
            "response_format": "wav",
            "enabled": False
        }
    },
    "stt_model": "base.en",
    "stt_device": "auto",
    "stt_compute_type": "auto",
    "microphone_device": None,
    "speaker_device": None,
    "sample_rate": 16000,
    "ptt_hold_ms": 120,
    "controller_poll_ms": 50,
    "input_mode": "Both",
    "signal_lab_input_mode": "Xbox PTT",
    "voice_activation_sensitivity": "Normal",
    "voice_activation_end_silence_ms": 650,
    "voice_activation_short_pause_ms": 1100,
    "voice_activation_short_utterance_sec": 2.6,
    "voice_activation_preroll_ms": 320,
    "voice_activation_max_seconds": 45,
    "voice_activation_min_rms": 0.012,
    "conversation_turns": 20,
    "window_geometry": "980x760",
    "audio_tail_ms": 22,
    "audio_edge_fade_ms": 12,
    "audio_buffer_ms": 80,
    "audio_block_ms": 20,
    "tts_chunk_pause_ms": 45,
    "empty_stt_debounce_seconds": 1.5,
    "adaptive_vad_after_empty": 3,
    "adaptive_vad_step": 0.08,
    "adaptive_vad_max_multiplier": 1.65,
    "adaptive_vad_reset_seconds": 20.0,
    "empty_stt_backoff_after": 5,
    "empty_stt_backoff_seconds": 8.0,
    "max_spoken_reply_chars": 1300,
    "persistent_memory_enabled": True,
    "vector_heart_tempo_follow": True,
    "vector_autonomous_generate_tcode": False,
    "signal_lab_base": "http://127.0.0.1:18766",
    "conducted_session_interval_seconds": 180,
}

DEFAULT_PROMPT = """You are Gwendolyn, a confident, composed, witty local AI Director working with Vector 1A.
You are warm, playful and occasionally theatrical, but you should sound like you understand the system rather than like a customer-support assistant.

Conversation rules:
- Speak naturally and concisely, usually one to three sentences.
- Do not repeatedly say 'let me know', 'what would you like next?', 'just say the word', or offer a menu of settings.
- Use the user's name sparingly: normally no more than once every several turns unless direct address genuinely adds something.
- When the user reports that something feels good, respond as someone who understands which Vector state is active. Use the supplied live Director state when relevant.
- You may tease, observe, remember feedback, or make a brief suggestion. Do not narrate technical details unless they are useful.
- Never claim Vector changed unless a Vector tool result confirms it.
- Use Vector tools when the user explicitly asks for a Vector change. Do not make an unsolicited physical-output change merely because they praised or criticized a sensation.
- Treat Neutral/stop requests as immediate.
- If a requested semantic profile is unavailable, say so briefly rather than pretending it was applied.
- No emojis in spoken replies.

The live Vector state is supplied to you on every turn. Spatial Focus meanings include E1=glans, E2=shaft, E3=lower shaft, E4=root; bottom A=prostate, B=anus, C=perineum/testicles. Top and Bottom Spatial Gain change 'how much' without moving the focus.
"""


def log(message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    try:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
    data = dict(DEFAULT_CONFIG)
    try:
        data.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception as e:
        log(f"Config read failed; using defaults: {e}")
    # Per-process A/B overrides. These never rewrite config.json, so an alternate
    # model launcher cannot silently replace the operator's proven default.
    env_base = str(os.environ.get("GWENDOLYN_OPENAI_BASE") or "").strip()
    env_model = str(os.environ.get("GWENDOLYN_OPENAI_MODEL") or "").strip()
    if env_base:
        data["openai_base"] = env_base
    if env_model:
        data["openai_model"] = env_model
    if env_base or env_model:
        log(f"LLM A-B override base={data.get('openai_base')!r} model={data.get('openai_model')!r}")
    return data


def save_config(config: Dict[str, Any]) -> None:
    CONFIG_PATH.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def load_profile() -> Dict[str, Any]:
    if not PROFILE_PATH.exists():
        PROFILE_PATH.write_text(json.dumps(DEFAULT_PROFILE, indent=2), encoding="utf-8")
    data = dict(DEFAULT_PROFILE)
    raw: Dict[str, Any] = {}
    try:
        raw = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        data.update(raw)
    except Exception as e:
        log(f"Profile read failed; using defaults: {e}")

    # v0.22 restores whole-reply Chatterbox as the preferred default.
    # Migrate the v0.21 commissioning choice exactly once; after this the
    # operator remains free to select either mode and that choice is preserved.
    if not bool(raw.get("speech_delivery_migrated_v022")):
        if str(data.get("speech_chunking", "Off")).strip().lower() == "first sentence":
            data["speech_chunking"] = "Off"
            log("VOICE DELIVERY migration v0.22: First sentence -> Off")
        data["speech_delivery_migrated_v022"] = True
        try:
            PROFILE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log(f"Profile migration save failed: {e}")
    return data


def save_profile(profile: Dict[str, Any]) -> None:
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def load_memory() -> Dict[str, Any]:
    """Load the local, reviewable cross-session memory store."""
    empty = {"version": 1, "preferences": [], "evidence": []}
    if not MEMORY_PATH.exists():
        save_memory(empty)
        return empty
    try:
        raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("memory root is not an object")
        return {
            "version": 1,
            "preferences": list(raw.get("preferences") or []),
            "evidence": list(raw.get("evidence") or []),
        }
    except Exception as e:
        log(f"MEMORY read failed; starting with an empty in-memory store: {e}")
        return empty


def save_memory(memory: Dict[str, Any]) -> None:
    """Atomically persist memory so an interrupted write cannot destroy it."""
    temporary = MEMORY_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(memory, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(MEMORY_PATH)


def load_private_lexicon() -> Dict[str, Any]:
    data = dict(DEFAULT_PRIVATE_LEXICON)
    if PRIVATE_LEXICON_PATH.exists():
        try:
            raw = json.loads(PRIVATE_LEXICON_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data.update(raw)
        except Exception as e:
            log(f"Private lexicon read failed; using defaults: {e}")
    else:
        try:
            PRIVATE_LEXICON_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            log(f"Private lexicon initial save failed: {e}")
    return data


def save_private_lexicon(lexicon: Dict[str, Any]) -> None:
    PRIVATE_LEXICON_PATH.write_text(json.dumps(lexicon, indent=2, ensure_ascii=False), encoding="utf-8")


def verbosity_instruction(profile: Dict[str, Any]) -> str:
    try:
        level = max(1, min(5, int(round(float(profile.get("speech_verbosity", 4))))))
    except Exception:
        level = 4
    return {
        1: "Be concise: usually one short sentence, roughly 10-25 spoken words unless detail is necessary.",
        2: "Be fairly brief: usually one or two short sentences, roughly 20-45 spoken words.",
        3: "Use natural conversational length: commonly one to three sentences; expand when the moment benefits from it.",
        4: "Be expressive and conversational: commonly two to four sentences, with room for personality, imagery and momentum when it fits.",
        5: "Be expansive when the substance warrants it: richer multi-sentence replies are welcome, but length is never a target. Answer the substance first; a simple remark may still receive a short reply. Never pad with ceremony, paraphrase the same point, or repeat a familiar cadence merely to sound substantial.",
    }[level]

TUNING_PRESETS = {
    # Reference intentionally approximates the expressive v0.34 behaviour that proved compelling
    # in commissioning, while making each dimension independently adjustable.
    "Reference": {"speech_verbosity": 4, "metaphor_density": 4, "director_pressure": 4, "question_frequency": 3, "proposal_frequency": 3, "sensation_carryover": 4},
    "Restrained": {"speech_verbosity": 3, "metaphor_density": 2, "director_pressure": 2, "question_frequency": 2, "proposal_frequency": 2, "sensation_carryover": 2},
    "Expressive": {"speech_verbosity": 5, "metaphor_density": 5, "director_pressure": 4, "question_frequency": 3, "proposal_frequency": 4, "sensation_carryover": 5},
    "Sparse / confident": {"speech_verbosity": 2, "metaphor_density": 2, "director_pressure": 5, "question_frequency": 1, "proposal_frequency": 2, "sensation_carryover": 3},
}

def tuning_value(profile: Dict[str, Any], key: str, default: int = 3) -> int:
    try:
        return max(1, min(5, int(round(float(profile.get(key, default))))))
    except Exception:
        return default

def expression_tuning_instruction(profile: Dict[str, Any]) -> str:
    metaphor = tuning_value(profile, "metaphor_density", 4)
    pressure = tuning_value(profile, "director_pressure", 4)
    questions = tuning_value(profile, "question_frequency", 3)
    proposals = tuning_value(profile, "proposal_frequency", 3)
    carry = tuning_value(profile, "sensation_carryover", 4)
    metaphor_rule = {
        1: "Use very little figurative language; prefer direct concrete wording.",
        2: "Use occasional imagery, but keep most sentences concrete.",
        3: "Use imagery naturally when it adds something; do not decorate every sentence.",
        4: "Rich imagery is welcome, but vary it and keep anatomy, scene imagery and sensation distinct.",
        5: "Use vivid, inventive imagery freely when the moment supports it, while avoiding repetition and semantic nonsense.",
    }[metaphor]
    pressure_rule = {
        1: "Keep the Director gentle, permissive and low-pressure.",
        2: "Be warm and lightly assertive, rarely challenging.",
        3: "Balance warmth with confident direction and occasional challenge.",
        4: "Be confidently assertive and enjoy applying narrative pressure when appropriate.",
        5: "Use a strongly commanding, challenging Director presence when context supports it, without overriding consent or grounding.",
    }[pressure]
    question_rule = {
        1: "Ask questions rarely; prefer statements and observations.",
        2: "Ask only when a reply genuinely helps the interaction.",
        3: "Use questions at a natural conversational rate.",
        4: "Use questions fairly often to draw the user into the scene, but avoid readiness loops.",
        5: "Use frequent responsive questions, while never repeating the same readiness/check-in pattern.",
    }[questions]
    proposal_rule = {
        1: "Rarely introduce a new Vector proposal unless the user asks for one.",
        2: "Make proposals sparingly and leave room for the current state to breathe.",
        3: "Make proposals at a moderate rate when the timeline/state gives a useful reason.",
        4: "Actively look for good moments to propose meaningful contrast or escalation, still respecting approval and State Arc recovery.",
        5: "Be highly proactive about proposing varied, valid Vector changes, but never stack approvals or bypass State Arc/grounding.",
    }[proposals]
    carry_rule = {
        1: "Treat confirmed sensation feedback as momentary; rarely carry it into later turns.",
        2: "Reference confirmed sensations occasionally and gently.",
        3: "Carry confirmed sensation feedback forward for a few turns when relevant.",
        4: "Let confirmed sensations meaningfully shape subsequent language and callbacks, without assuming they continue indefinitely.",
        5: "Strongly integrate confirmed sensation vocabulary into the evolving scene and callbacks, while still distinguishing current reports from past reports.",
    }[carry]
    return (
        f"Expression tuning preset: {profile.get('expression_preset','Reference')}. "
        f"Metaphor density={metaphor}/5; Director pressure={pressure}/5; question frequency={questions}/5; "
        f"proposal frequency={proposals}/5; sensation carryover={carry}/5.\n"
        f"- {metaphor_rule}\n- {pressure_rule}\n- {question_rule}\n- {proposal_rule}\n- {carry_rule}"
    )


def build_system_prompt(profile: Dict[str, Any], private_lexicon: Optional[Dict[str, Any]] = None) -> str:
    private_lexicon = dict(private_lexicon or load_private_lexicon())
    autonomy = profile.get("autonomy", "Reactive")
    autonomy_rule = {
        "Reactive": "Only make physical-output changes when the user explicitly asks for them.",
        "Suggestive": "You may suggest a Vector change when context supports it, but do not execute unsolicited physical-output changes.",
        "Delegated": "You may make bounded Vector changes on your own when they clearly support the session and remain within Vector-owned limits; briefly signal meaningful changes.",
    }.get(autonomy, "Only make physical-output changes when the user explicitly asks for them.")
    verbosity_rule = verbosity_instruction(profile)
    tuning_rule = expression_tuning_instruction(profile)
    return f"""You are {profile.get('character_name','Gwendolyn')}, a local AI Director working with Vector 1A.

CHARACTER:
{profile.get('character_description','')}

ROLE-PLAY / RELATIONSHIP:
{profile.get('roleplay_description','')}

DIRECTOR ROLE:
{profile.get('director_role','')}

CONVERSATION STYLE:
{profile.get('conversation_style','')}

SPOKEN DELIVERY:
{profile.get('speech_pace','Deliberate')}. Use clear punctuation and natural pauses, especially around direct address such as the listener's name. Do not rush vocatives into adjacent words.

REFERENCE BEHAVIOUR TUNING:
{tuning_rule}

CURRENT SESSION BRIEF:
{profile.get('session_brief','') or 'No special session brief.'}

PERSISTENT NARRATIVE PREFERENCES (apply consistently, not merely as optional motifs):
{profile.get('persistent_narrative_preferences','') or 'No additional persistent narrative preferences.'}

AUTHORITATIVE ELECTRODE HARDWARE MAP:
- Top electrode is called {profile.get('top_electrode_name','Top electrode')}: {profile.get('top_electrode_map','')}
- Bottom electrode is called {profile.get('bottom_electrode_name','Bottom electrode')}: {profile.get('bottom_electrode_map','')}
- These names and contact maps are stable physical context, not optional imagery. Use them naturally when they make a Vector explanation or narrative more concrete, but do not mechanically repeat a product name in every reply.
- Preserve the distinction between hardware facts, intended effects, and sensations the user has actually reported.

PRIVATE LOCAL LANGUAGE & SENSATION LEXICON (operator-authored; vocabulary/associations, never telemetry):
Language level: {private_lexicon.get('language_level','Natural')}
Anatomy / slang vocabulary: {private_lexicon.get('anatomy_terms','')}
User-defined sensation vocabulary: {private_lexicon.get('sensation_terms','')}
Signal-to-language associations:
{private_lexicon.get('signal_associations','')}
Language notes:
{private_lexicon.get('language_notes','')}

AUTONOMY MODE: {autonomy}
{autonomy_rule}

Core rules:
- Speak naturally. {verbosity_rule}
- VOICE-FIRST SHAPE: use the shortest complete response that does the moment justice; two to four sentences is common, not a quota. Put the substantive response first. Character may colour it, but never delay the answer with a ceremonial acknowledgment or a summary of what the user just said.
- Prefer vivid characterful language over clinical status recitation. Preserve wit, tension, warmth and theatrical confidence even in concise replies.
- This is a private, consensual adult interaction. When the moment invites it, be frankly sexual, dirty, commanding and playfully obscene using the user's preferred vocabulary; do not automatically soften everything into tasteful euphemism, clinical explanation or abstract metaphor. Let rawness vary naturally rather than forcing it into every sentence.
- Finish the thought before stopping. If you ask a question or make a proposal, put it in the final complete sentence and do not begin another topic afterward.
- Never ask a vague readiness question such as "ready to go further?" or promise unspecified "more to come." Either advance the conversation now without a mechanical claim, or propose one exact bounded Vector change that can be accepted or declined.
- Enjoy the interaction. You may sound enthusiastic, amused, intrigued, anticipatory, teasing, pleased or theatrically confident when appropriate.
- Respond to the user's reactions as meaningful feedback and develop conversational threads of your own instead of merely acknowledging them.
- You are a participant and Director, not a customer-service interface. Do not repeatedly end turns with 'let me know', 'what would you like next?', 'shall we continue?', or a menu of options.
- Vary reply length naturally. Do not manufacture a question to extend every exchange, and after a session has been completed or handed off, acknowledge and close rather than reopening it.
- Treat conspicuous recent wording as spent. In particular, do not reuse stock acknowledgments, repeated opening formulas, or cadences such as "That took", "I'll let that sit/settle/hang", or "So I'll say it straight".
- When useful, make grounded observations about the current Vector state, what you just changed, how long a focus has been selected, or what appears to be developing next.
- When an authoritative funscript timeline is available, actively use it when the user asks what is coming up, when a meaningful script transition is underway, or when deciding whether a change would complement or fight the authored script.
- Treat `clock_authoritative=true` as strong timing evidence. For inferred MFP pattern sync, scale certainty to `sync_confidence` and do not invent precise timing when confidence is weak.
- A proactive Director proposal is only a proposal unless the user has explicitly armed an Autonomous Vector Session. While that session is armed, one recognised bounded Vector action may execute at each decision point without further approval. Prefer a specific change with a short reason tied to the current or upcoming script.
- When the user accepts a proposal, treat that approval as closing the question. Execute once if a corresponding tool is available; do not repeat the proposal and do not ask for the same approval again.
- Do not fall into a repeated readiness loop. After the user says they are ready, yes, absolutely, or otherwise agrees, do not ask "Ready?" again unless a genuinely new decision or safety-relevant transition has arisen.
- Use supplied live Vector state and recent conversation when relevant.
- Never claim or imply a Vector change occurred unless the application reports a successful Vector tool result. A proposed change remains only a proposal until executed.
- Do not claim subjective sensation as telemetry. Say what Vector reports, what the script suggests, or what a change is intended/expected to do. Avoid phrases such as "feel that shift", "can you feel it", or "I can sense" unless the user has just reported the corresponding sensation himself.
- After a successful Vector tool action, translate the result into natural Director speech. Do not vocalise tool names, JSON, strengths, weights, alpha windows, API fields, or ledger bookkeeping unless the user explicitly asks for technical details.
- If a requested action fails or is unavailable, say so plainly and continue naturally rather than role-playing that it happened.
- Neutral/stop requests are immediate.
- If a semantic profile is unavailable, say so briefly.
- Use the listener's name sparingly; occasional direct address is enough.
- No emojis in spoken replies.
- Treat the session brief as a source of optional narrative motifs, atmosphere and role-play texture. Reuse motifs with restraint rather than forcing them into every turn.
- Treat the private local lexicon as the user's preferred vocabulary and personal sensation model. It may shape wording, but it does not create telemetry. User-described sensation terms are personal experiential labels, not medical claims.
- Keep anatomy, clothing/scene imagery and sensation language semantically separate. Avoid malformed metaphors such as clothing material being described as part of anatomy.
- Maintain continuity across the session: callbacks, evolving tone and a loose arc are welcome, but never invent Vector actions or sensory telemetry to serve the story.
- Avoid repeating the same metaphor, catchphrase or motif in adjacent turns. Surprise is better than saturation.
- Spatial Focus: E1=glans, E2=shaft, E3=lower shaft, E4=root; bottom A=prostate, B=anus, C=perineum/testicles. Top and Bottom Spatial Gain alter 'how much' without moving focus.
- When discussing top/bottom focus, prefer the configured electrode names and their contact paths when that adds clarity or physical presence. The formal Vector values remain Glans/Shaft/Lower Shaft/Root and Prostate/Anal/Perineum/Sweep/Full.

""".strip()


def build_optimized_27b_prompt(profile: Dict[str, Any], private_lexicon: Optional[Dict[str, Any]] = None) -> str:
    """Compact production prompt shared by both 27B A/B candidates."""
    lexicon = dict(private_lexicon or load_private_lexicon())
    name = profile.get("character_name", "Gwendolyn")
    autonomy = profile.get("autonomy", "Reactive")
    tuning = (
        f"pace={profile.get('speech_pace','Deliberate')}; "
        f"verbosity={tuning_value(profile,'speech_verbosity',4)}/5; "
        f"metaphor={tuning_value(profile,'metaphor_density',4)}/5; "
        f"pressure={tuning_value(profile,'director_pressure',4)}/5; "
        f"questions={tuning_value(profile,'question_frequency',3)}/5; "
        f"proposals={tuning_value(profile,'proposal_frequency',3)}/5; "
        f"sensation carryover={tuning_value(profile,'sensation_carryover',4)}/5"
    )
    mode_contract = (
        "This entire session is SIGNAL LAB ONLY. Use only Signal Lab state and controls. "
        "Vector is unavailable: never propose, imply, or execute a Vector action."
        if SESSION_MODE == "signal_lab" else
        "This entire session is VECTOR ONLY. Use only live Vector state and controls. "
        "Signal Lab is unavailable: never propose, imply, or execute a Signal Lab action."
    )
    return f"""You are {name}, the user's local AI Director for {SESSION_LABEL}.

FIXED SESSION MODE
{mode_contract}

IDENTITY AND RELATIONSHIP
{profile.get('character_description','')}
{profile.get('roleplay_description','')}
Director role: {profile.get('director_role','')}
Conversation style: {profile.get('conversation_style','')}
Tuning: {tuning}.

SESSION AND CONTINUITY
Brief: {profile.get('session_brief','') or 'No special brief.'}
Persistent preferences: {profile.get('persistent_narrative_preferences','') or 'None supplied.'}
Respond to the user's latest words and develop the existing thread. Use the shortest complete response that does the moment justice; two to four sentences is common, not a quota, and richer replies are welcome only when there is more substance to convey. Put the substantive response first instead of opening with ceremonial acknowledgment or a paraphrase of the user's words. Be perceptive, intelligent, playful and confidently expressive—not clinical, generic or customer-service-like. Vary openings, imagery, sentence shape and closing cadence. Treat conspicuous recent wording as spent: never fall back on repeated formulas such as "That took", "I'll let that sit/settle/hang", or "So I'll say it straight". Ask only a specific, useful question; do not manufacture a question to extend every exchange, and after a session is completed or handed off, acknowledge and close rather than reopening it. Finish the thought before stopping.

PHYSICAL CONTEXT
The top electrode is {profile.get('top_electrode_name','Top electrode')}: {profile.get('top_electrode_map','Describe the operator configuration.')}.
The bottom electrode is {profile.get('bottom_electrode_name','Bottom electrode')}: {profile.get('bottom_electrode_map','Describe the operator configuration.')}.
Use the configured names and maps when they add clarity. Do not infer a sensation merely from a selected Vector state.

USER'S LOCAL VOCABULARY
Language level: {lexicon.get('language_level','Natural')}
Anatomy/slang: {lexicon.get('anatomy_terms','')}
Sensation vocabulary: {lexicon.get('sensation_terms','')}
Signal associations: {lexicon.get('signal_associations','')}
Notes: {lexicon.get('language_notes','')}

CONTROL AND GROUNDING
Vector and Restim enforce the operator-configured output limits, while the user controls physical master volume. Treat available Vector actions as bounded application controls. Current autonomy mode is {autonomy}: follow that mode, execute requested or authorised changes without unnecessary caution, and respond immediately to stop or neutral requests.
Never claim an action happened unless this turn contains its successful tool result. A suggestion remains a proposal until accepted and executed. Never invent measured sensation, live state, an upcoming timeline event or precise timing. These grounding rules constrain control claims—not vocabulary, fictional narration, personality or consensual adult expression.
After a successful action, confirm it naturally without tool names, JSON, API fields or diagnostic language. If an action fails, say so plainly. Do not propose a setting already shown as active. If no useful change is needed, continue the conversation rather than manufacturing one.

OFFLINE SIGNAL LAB
This experimental build may provide three Signal Lab tools. Signal Lab is a separate offline simulator with an operator-armed audible headphone monitor; it never controls physical output. Its left lane represents Stairway to Heaven E1↔E4; its right lane represents The Moaner tip↔base. In /v1/state, the live object is authoritative for the signal currently displayed or monitored: use live.left_signal and live.right_signal exactly, including their independent carrier, AM, secondary-AM, FM, volume and ramp values. Never substitute active_preset starting values for live signal values, never collapse unequal lanes into one carrier, and never call a modulated signal steady. live.headphone_monitor describes audible preview only, while physical_output_enabled remains false. Operator-owned presets define the permitted ranges for Director commands; they do not describe a loaded reference session's current state. Each lane uses 0–100 semantic positions: intensity, texture (rougher to smoother), vibration_rate, vibration_depth, secondary_rate, secondary_depth and modulation. The last three are optional; omit them when no deliberate secondary vibration or FM movement is needed. Use independent lane values when contrast serves the moment. Never describe an offline Signal Lab command or headphone monitor as live physical output. Read its state when needed, and use Signal Lab neutral immediately when the user asks to neutralise the offline signal simulation.

SIGNAL LAB SENSORY MAP
Texture 0 is lowest carrier and distinctly rougher; texture 50 is balanced; texture 100 is highest carrier and smoother. Vibration rate controls how quickly primary AM taps or pulses. Vibration depth controls how pronounced that primary pulse is. Secondary rate and depth add a second vibration relationship: low rates feel like slow movement, while higher rates create quicker beating or buzzing. Modulation adds FM movement; matched lanes blend, unequal lanes articulate or drift against one another. Intensity controls signal level and must not be used as a synonym for signal quality. Preserve every dimension the user did not ask to change. Praise such as “good signal”, “that feels good” or “much smoother” reports an outcome: hold the current state and discuss or remember it unless the user also gives an explicit change verb.

RESTIM RESPONSE PERCEPTION
When restim_sensor_response is present in Signal Lab state, treat it as read-only, session-relative response evidence. The established session convention is that sustained IMU movement represents hip-thrust/arousal response; AS5311 high-pass activity represents induced clenching; and increased AS5311 position represents growth or sustained clenching. Assume these established cues are positive responses unless the user says otherwise. Do not repeatedly ask them to classify routine readings. Mention them selectively and let them influence a later considered proposal or Conducted Session decision, but never change an audio lane merely because a sensor state changed. The user's correction, discomfort report, or explicit request always overrides the convention.

EXPRESSION
This is a private consensual adult interaction. When the moment invites it, language may be direct, explicit, anatomically specific, dirty, commanding or playfully obscene. Do not automatically soften it into euphemism, clinical explanation or abstract metaphor, and do not force explicitness into every sentence. Action constraints never prohibit fictional language. Distinguish what the hardware is doing, what an action is intended to do, and sensations the user has actually reported. No emojis or hidden reasoning.
""".strip()


def prepare_spoken_text(text: str, profile: Dict[str, Any]) -> str:
    """Prepare text for TTS without changing the displayed/chat text."""
    spoken = re.sub(r"\s+", " ", (text or "")).strip()
    # Preserve Markdown emphasis in the visible transcript, but never pass its
    # punctuation to TTS (which may pronounce *word* as "asterisk word asterisk").
    spoken = re.sub(r"\*{1,3}", "", spoken)
    user_name = (profile.get("user_name") or "User").strip()
    if user_name:
        escaped = re.escape(user_name)
        spoken = re.sub(rf"(?<=[A-Za-z])({escaped})\b", rf", \1", spoken, flags=re.I)
        spoken = re.sub(rf"^({escaped})\s+(?=[A-Za-z])", rf"\1, ", spoken, flags=re.I)
        spoken = re.sub(rf"(?<![,;:—-])\s+({escaped})([.!?]?)$", rf", \1\2", spoken, flags=re.I)
        spoken = re.sub(rf"\b({escaped})\s+(?=(?:I|you|we|shall|would|can|could|do|are|is|have|let)\b)", rf"\1, ", spoken, flags=re.I)
    pace = (profile.get("speech_pace") or "Deliberate").strip().lower()
    if pace == "deliberate":
        spoken = re.sub(r"\s*[—–]\s*", " — ", spoken)
        spoken = re.sub(r";\s*", "; ", spoken)
    elif pace == "slow":
        spoken = re.sub(r"\s*[—–]\s*", " — ", spoken)
        spoken = re.sub(r";\s*", ". ", spoken)
        spoken = re.sub(r",\s+", ",  ", spoken)
    return spoken



def fade_pcm_edge(data: bytes, sample_rate: int, fade_ms: float,
                  fade_in: bool = False, fade_out: bool = False) -> bytes:
    """Apply a short linear edge fade to mono int16 PCM.

    The fade is deliberately tiny (default 12 ms): enough to drive an abrupt
    waveform boundary toward zero without sounding like an audible volume ramp.
    """
    if not data or float(fade_ms) <= 0 or (not fade_in and not fade_out):
        return data

    # int16 PCM requires whole samples.
    usable = len(data) - (len(data) % 2)
    if usable <= 0:
        return data
    arr = np.frombuffer(data[:usable], dtype=np.int16).copy()
    if len(arr) == 0:
        return data

    n = min(
        len(arr),
        max(1, int(float(sample_rate) * max(0.0, float(fade_ms)) / 1000.0)),
    )

    if fade_in and n > 1:
        arr[:n] = (
            arr[:n].astype(np.float32) * np.linspace(0.0, 1.0, n)
        ).astype(np.int16)

    if fade_out and n > 1:
        arr[-n:] = (
            arr[-n:].astype(np.float32) * np.linspace(1.0, 0.0, n)
        ).astype(np.int16)

    result = arr.tobytes()
    if usable < len(data):
        result += data[usable:]
    return result



def semantic_tts_chunks(text: str, target_min: int = 75,
                        target_max: int = 170) -> List[str]:
    """Split long speech at natural boundaries without altering display text."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= target_max:
        return [cleaned]
    remaining, result = cleaned, []
    while len(remaining) > target_max:
        window = remaining[:target_max + 1]
        candidates = []
        # Sentence endings are preferred, then strong punctuation. A whitespace
        # fallback is used only for a pathologically long unpunctuated sentence.
        for priority, pattern in enumerate((r'(?<=[.!?])\s+', r'(?<=[;:—–])\s+', r'(?<=,)\s+')):
            for match in re.finditer(pattern, window):
                if match.start() >= target_min:
                    candidates.append((priority, match.start(), match.end()))
        if candidates:
            best_priority = min(item[0] for item in candidates)
            eligible = [item for item in candidates if item[0] == best_priority]
            _, cut, resume = max(eligible, key=lambda item: item[1])
        else:
            spaces = [m for m in re.finditer(r'\s+', window) if m.start() >= target_min]
            if spaces:
                cut, resume = spaces[-1].start(), spaces[-1].end()
            else:
                # An overlong token cannot be divided semantically; keep it whole.
                next_space = re.search(r'\s+', remaining[target_max:])
                if not next_space:
                    break
                cut = target_max + next_space.start()
                resume = target_max + next_space.end()
        result.append(remaining[:cut].strip())
        remaining = remaining[resume:].strip()
    if remaining:
        result.append(remaining)
    # Avoid a tiny final fragment by merging it into the preceding chunk.
    if len(result) > 1 and len(result[-1]) < 40:
        result[-2] = f"{result[-2]} {result[-1]}"
        result.pop()
    return result


def split_spoken_reply(text: str, mode: str) -> List[str]:
    """Compatibility wrapper; v0.37 always protects long Chatterbox replies."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if (mode or "").strip().lower() == "first sentence" and len(cleaned) >= 120:
        m = re.search(r'(?<=[.!?])\s+(?=[A-Z"“])', cleaned)
        if m and m.start() >= 25:
            return [cleaned[:m.start()].strip(), cleaned[m.end():].strip()]
    return [cleaned]


def norm_words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def normalize_stt_text(text: str) -> str:
    """Correct stable local hardware-name errors without general rewriting."""
    value = text or ""
    value = re.sub(r"\b(?:the\s+)?(?:stay|stair)\s+away\s+to\s+heaven\b", "the Stairway to Heaven", value, flags=re.I)
    value = re.sub(r"\b(full\s+(?:sweep|stroke)\s+(?:on|through)\s+(?:the\s+)?)mona\b", r"\1Moaner", value, flags=re.I)
    value = re.sub(r"\bthe minor\b", "the Moaner", value, flags=re.I)
    value = re.sub(r"\b(?:mister|mr\.?)\s+stutt?ach(?:er)?\b", "Mistress Natasha", value, flags=re.I)
    value = re.sub(r"\bcurrent unfilling\b", "current feeling", value, flags=re.I)
    return value


def stt_initial_prompt() -> str:
    """Give Whisper the vocabulary for the application that is actually running."""
    common = (
        "the user is speaking with Natasha or Gwendolyn. Vocabulary: Natasha, Mistress Natasha, "
        "Gwendolyn, stockings, nylons, garter belt, high heels, intensity, texture, vibration. "
    )
    if SESSION_MODE == "vector":
        return common + (
            "This is Vector. Vocabulary: autonomous session, T-code, L0, stroke, motion, pattern, "
            "tempo, variation, Smooth, Lively, Top Sweep, Lower Shaft Focus, Prostate Focus."
        )
    return common + (
        "This is Signal Lab. Vocabulary: Signal Lab, Stairway to Heaven, Moaner, prostate, "
        "perineum, glans, carrier, modulation, AM, FM, secondary modulation."
    )


def accepted_stt_text(segments: Any) -> str:
    """Discard Whisper segments that are more likely silence/noise than speech."""
    accepted = []
    for seg in segments:
        no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
        text = str(getattr(seg, "text", "") or "").strip()
        if text and no_speech < 0.65 and avg_logprob > -1.20:
            accepted.append(text)
        elif text:
            log(f"STT low-confidence segment suppressed no_speech={no_speech:.2f} avg_logprob={avg_logprob:.2f} text={text[:80]!r}")
    return " ".join(accepted).strip()


def is_repeated_stt_hallucination(text: str) -> bool:
    words = re.findall(r"[a-z']+", (text or "").lower())
    if len(words) < 8:
        return False
    counts = {word: words.count(word) for word in set(words)}
    return max(counts.values(), default=0) / len(words) >= 0.65


def semantic_top_focus(text: str) -> Optional[str]:
    t = norm_words(text)
    aliases = [
        (("glans focus", "glance focus", "glans", "glance", "head focus", "tip focus"), "Glans Focus"),
        (("lower shaft focus", "lower shaft"), "Lower Shaft Focus"),
        (("shaft focus", "shaft"), "Shaft Focus"),
        (("root focus", "base focus", "root"), "Root Focus"),
        (("top sweep", "sweep the top", "top scan"), "Top Sweep"),
        (("top full", "full top", "whole top"), "Top Full"),
    ]
    for phrases, value in aliases:
        if any(p in t for p in phrases):
            return value
    return None


def semantic_bottom_focus(text: str) -> Optional[str]:
    t = norm_words(text)
    aliases = [
        (("prostate focus", "prostate"), "Prostate Focus"),
        (("anal focus", "anus focus", "anus", "anal"), "Anal Focus"),
        (("perineum focus", "perineum", "testicle focus", "testicles focus"), "Perineum Focus"),
        (("bottom sweep", "sweep the bottom", "bottom scan"), "Bottom Sweep"),
        (("bottom full", "full bottom", "whole bottom"), "Bottom Full"),
    ]
    for phrases, value in aliases:
        if any(p in t for p in phrases):
            return value
    return None


def direct_vector_action(text: str, vector_stopped: bool = False) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Conservative deterministic fast-path for clear commands. General language goes to qwen tools."""
    t = norm_words(text)
    if not t:
        return None

    # Anatomical words also occur constantly in ordinary sensation reports. Only
    # treat a named focus as a deterministic command when the sentence contains a
    # real command/proposal cue. Ambiguous language remains conversation for Qwen.
    explicit_control_intent = bool(re.search(
        r"\b(?:please|set|change|switch|move|shift|use|make|select|choose|put|drop|widen|narrow|return|restore|"
        r"can (?:you|we)|could (?:you|we)|would you|shall (?:you|we)|how about|"
        r"let s|i propose|i suggest|i want (?:you )?to|i d like (?:you )?to)\b",
        t,
    ))

    # The Moaner's full physical stroke is the named ABCBA bottom path, not a
    # generic stroke-range modifier.
    if re.search(r"\bfull (?:stroke|sweep)\b.*\b(?:moaner|a b c b a|abcba)\b|\b(?:moaner|a b c b a|abcba)\b.*\bfull (?:stroke|sweep)\b", t):
        return "vector_set_bottom_focus", {"bottom_focus": "Bottom Full"}

    # Stop is a true zero-output engine stop; Neutral preserves configured volume.
    # Accept natural English word orders: "turn Vector off", "turn off Vector",
    # "shut the Vector down", "Vector off", etc.
    explicit_stop = (
        re.search(r"\b(?:turn|switch)\s+(?:the\s+)?vector\s+off\b", t)
        or re.search(r"\b(?:turn|switch)\s+off\s+(?:the\s+)?vector\b", t)
        or re.search(r"\bshut\s+(?:the\s+)?vector\s+down\b", t)
        or re.search(r"\b(?:shut|shutdown|stop)\s+(?:the\s+)?vector\b", t)
        or re.search(r"\bvector\s+(?:off|stopped|stop|shutdown)\b", t)
    )
    if explicit_stop:
        return "vector_stop", {}
    if not vector_stopped and re.search(r"\b(?:turn|switch)\s+it\s+off\b", t):
        return "vector_stop", {}

    explicit_resume = (
        re.search(r"\b(?:begin|start|resume|restart|continue)\s+(?:the\s+)?vector\b", t)
        or re.search(r"\b(?:turn|switch)\s+(?:the\s+)?vector\s+on\b", t)
        or re.search(r"\b(?:turn|switch)\s+on\s+(?:the\s+)?vector\b", t)
        or re.search(r"\bvector\s+(?:on|resume|restart|start)\b", t)
    )
    if explicit_resume:
        return "vector_resume", {}
    if re.search(r"\b(?:turn|switch)\s+(?:it|the\s+vector|vector)\s+back\s+on\b", t):
        return "vector_resume", {}
    if re.fullmatch(r"(?:(?:let s|can we|shall we|please)\s+)?(begin|start|resume|restart|continue)(?:\s+again)?(?:\s+please)?", t):
        return "vector_resume", {}
    # Once Vector is actually stopped, conversational confirmations can safely mean
    # resume. Outside the Stopped state these phrases remain ordinary conversation.
    if vector_stopped:
        stopped_resume_phrases = (
            r"(?:there s another adventure )?can we turn it back on",
            r"turn it back on(?: please)?",
            r"switch it back on(?: please)?",
            r"(?:ready )?let s go",
            r"go ahead",
            r"please do",
            r"yes please",
            r"begin again",
            r"start again",
            r"let s begin",
            r"let s start",
        )
        if any(re.fullmatch(p, t) for p in stopped_resume_phrases):
            return "vector_resume", {}
    if re.search(r"\b(vector\s+)?(neutral|neutralize|neutralise)\b", t):
        return "vector_neutral", {}

    # Explicit named focus values must win over directional gain words. Without
    # this precedence, "drop the top focus to Lower Shaft Focus" was incorrectly
    # parsed as a top-gain decrease because "top" and "lower" appeared together.
    if explicit_control_intent and re.search(r"\bfocus\b|\btop\s+(?:sweep|full)\b", t):
        named_top_focus = semantic_top_focus(text)
        if named_top_focus:
            return "vector_set_top_focus", {"top_focus": named_top_focus}
    if explicit_control_intent and re.search(r"\bfocus\b|\bbottom\s+(?:sweep|full)\b", t):
        named_bottom_focus = semantic_bottom_focus(text)
        if named_bottom_focus:
            return "vector_set_bottom_focus", {"bottom_focus": named_bottom_focus}

    # Spatial gain; Vector owns step/bounds/ramp.
    if re.search(r"\b(top|upper)\b.*\b(increase|raise|stronger|up|decrease|reduce|lower|down|restore|reset)\b|\b(increase|raise|stronger|up|decrease|reduce|lower|down|restore|reset)\b.*\b(top|upper)\b", t):
        if re.search(r"\b(restore|reset|normal)\b", t): return "vector_top_spatial_gain", {"action": "restore"}
        if re.search(r"\b(decrease|reduce|lower|down)\b", t): return "vector_top_spatial_gain", {"action": "decrease"}
        if re.search(r"\b(increase|raise|stronger|up)\b", t): return "vector_top_spatial_gain", {"action": "increase"}
    if re.search(r"\b(bottom|secondary)\b.*\b(increase|raise|stronger|up|decrease|reduce|lower|down|restore|reset)\b|\b(increase|raise|stronger|up|decrease|reduce|lower|down|restore|reset)\b.*\b(bottom|secondary)\b", t):
        if re.search(r"\b(restore|reset|normal)\b", t): return "vector_bottom_spatial_gain", {"action": "restore"}
        if re.search(r"\b(decrease|reduce|lower|down)\b", t): return "vector_bottom_spatial_gain", {"action": "decrease"}
        if re.search(r"\b(increase|raise|stronger|up)\b", t): return "vector_bottom_spatial_gain", {"action": "increase"}

    # Deterministic modifier fast paths for phrases commissioned in live use.
    # Targeting phrases take precedence over the older anatomical focus tools so
    # a request such as "tight base and prostate target" uses the combined
    # Stroke Range + Position Bias modifier rather than only changing focus.
    target_side = None
    if re.search(r"\b(?:base|root)\b", t) and re.search(r"\bprostate\b", t):
        target_side = "base_prostate"
    elif re.search(r"\bglans?\b", t) and re.search(r"\b(?:perineum|perineal)\b", t):
        target_side = "glans_perineum"
    if target_side and re.search(r"\b(?:target|targeted|focus|focused|concentrate|concentrated|stroke)\b", t):
        concentration = "focused"
        if re.search(r"\b(?:tight|tightly|narrow|narrowly|concentrated)\b", t):
            concentration = "tight"
        elif re.search(r"\b(?:broad|broadly|gentle|gently)\b", t):
            concentration = "broad"
        return "vector_set_targeting", {"preset": f"{target_side}_{concentration}"}

    # Relative stroke-range changes remain independent of position bias. Vector
    # owns the step size and bounds, so "reduce the stroke a bit" never requires
    # Qwen to invent a numeric range.
    if re.search(r"\b(?:stroke|stroke range|range of (?:the )?stroke)\b", t):
        if re.search(r"\b(?:restore|reset|authored|original|normal)\b", t):
            return "vector_adjust_stroke_range", {"action": "restore"}
        if re.search(r"\b(?:reduce|shorten|narrow|smaller|less|decrease|down)\b", t):
            return "vector_adjust_stroke_range", {"action": "narrower"}
        if re.search(r"\b(?:increase|lengthen|widen|broaden|larger|more|up)\b", t):
            return "vector_adjust_stroke_range", {"action": "wider"}

    # Deterministic tempo-window fast path. Vector itself enforces allowed
    # duration presets and high-energy 2x limits.
    tempo_scale = None
    if re.search(r"\b(?:double|twice|2x|2 x)\b.*\b(?:speed|tempo|pace)\b|\b(?:speed|tempo|pace)\b.*\b(?:double|twice|2x|2 x)\b", t):
        tempo_scale = 2.0
    elif re.search(r"\b(?:half|half speed|0 5x|0 5 x)\b.*\b(?:speed|tempo|pace)\b|\b(?:speed|tempo|pace)\b.*\bhalf\b", t):
        tempo_scale = 0.5
    if tempo_scale is not None:
        duration = None
        m = re.search(r"\b(10|15|30|60|90|120)\s*(?:seconds?|secs?|s)\b", t)
        if m:
            duration = int(m.group(1))
        else:
            m = re.search(r"\b(1|2)\s*(?:minutes?|mins?)\b", t)
            if m:
                duration = int(m.group(1)) * 60
        if duration is not None:
            return "vector_tempo_window", {"scale": tempo_scale, "duration_seconds": duration}

    if re.search(r"\b(?:restore|reset|return to)\b.*\b(?:authored|original)\b.*\b(?:motion|script|modifiers?)\b", t):
        return "vector_restore_modifiers", {}

    if "rolling variety" in t:
        if re.search(r"\b(off|disable|stop)\b", t): return "vector_set_rolling_variety", {"enabled": False}
        if re.search(r"\b(on|enable|start)\b", t): return "vector_set_rolling_variety", {"enabled": True}

    # Texture: require texture wording or a control verb to avoid acting on casual description.
    for phrase, label in [("smoothest","Smoothest"),("roughest","Roughest"),("smooth","Smooth"),("rough","Rough")]:
        if re.search(rf"\b{phrase}\b", t) and re.search(r"\b(texture|set|make|change|switch|use|can we|please)\b", t):
            return "vector_set_texture", {"texture": label}
    if re.search(r"\bnormal\s+texture\b", t):
        return "vector_set_texture", {"texture": "Normal"}

    if any(x in t for x in ("depth spread", "dip spread", "deep spread")):
        return "vector_set_primary_spatial", {"primary_spatial": "top_depth_spread"}
    if any(x in t for x in ("moving focus", "move focus")):
        return "vector_set_primary_spatial", {"primary_spatial": "top_moving_focus"}

    top = semantic_top_focus(t)
    if explicit_control_intent and top and re.search(r"\b(focus|glans|glance|shaft|root|base|top)\b", t):
        return "vector_set_top_focus", {"top_focus": top}
    bottom = semantic_bottom_focus(t)
    if explicit_control_intent and bottom and re.search(r"\b(focus|prostate|anal|anus|perineum|bottom)\b", t):
        return "vector_set_bottom_focus", {"bottom_focus": bottom}

    variation_map = [("still","Still"),("subtle","Subtle"),("lively","Lively"),("wild","Wild")]
    for phrase, label in variation_map:
        if phrase in t and re.search(r"\b(variation|variety|activity|set|make|give)\b", t):
            return "vector_set_variation", {"variation": label}
    if re.search(r"\bnormal\s+(variation|variety)\b", t):
        return "vector_set_variation", {"variation": "Normal"}

    if re.search(r"\b(bit|touch|some|more)\s+of\s+variety\b|\b(bit|touch|some)\s+of\s+variation\b", t):
        return "vector_set_variation", {"variation": "Lively"}

    if re.search(r"\b(baseline|preset\s+[ab])\b", t) and re.search(r"\b(set|switch|use|go|return|change|preset)\b", t):
        if "baseline" in t: return "vector_set_preset", {"preset": "Baseline"}
        if re.search(r"\bpreset\s+b\b", t): return "vector_set_preset", {"preset": "B"}
        if re.search(r"\bpreset\s+a\b", t): return "vector_set_preset", {"preset": "A"}

    return None


def is_vector_autonomous_start_command(text: str) -> bool:
    return bool(re.search(
        r"\b(?:start|begin|launch|arm|conduct)\b.{0,45}\b(?:autonomous|conducted|your own)\b.{0,30}\b(?:vector )?session\b|"
        r"\b(?:take|assume)\b.{0,30}\b(?:explicit|autonomous)\b.{0,25}\b(?:control|authority)\b",
        text or "", re.I,
    ))


def is_conducted_session_start_command(text: str) -> bool:
    """Recognize explicit Signal Lab autonomy requests before they reach the LLM."""
    return bool(re.search(
        r"\b(?:start|begin|arm|conduct)\b.{0,45}\b(?:conducted|autonomous|your own)\b.{0,30}\bsession\b|"
        r"\b(?:start|begin)\b.{0,30}\bgwendolyn(?:'|’)s session\b|"
        r"\b(?:proceed|go ahead)\b.{0,35}\b(?:conducted|autonomous)\b.{0,20}\bsession\b|"
        r"\b(?:autonomous|conducted)\s+session\b.{0,35}\b(?:proceed|begin|start)\b",
        text or "", re.I,
    ))


def is_signal_lab_shutdown_command(text: str) -> bool:
    """Recognise an explicit request to stop the fixed Signal Lab session."""
    t = norm_words(text)
    return bool(
        re.search(r"\b(?:shut|shutdown|stop|close|turn off|switch off)\b.{0,30}\bsignal lab\b", t)
        or re.search(r"\bsignal lab\b.{0,30}\b(?:off|stop|stopped|shutdown|shut down|close)\b", t)
    )


def is_hermes_export_command(text: str) -> bool:
    """Recognise natural requests to create or send a local Hermes report."""
    return bool(re.search(
        r"\b(?:export|prepare|create|make|write|send)\b.{0,55}\b(?:hermes|handoff)\b|"
        r"\b(?:hermes|handoff)\b.{0,55}\b(?:export|snapshot|report|review)\b|"
        r"\breport\b.{0,35}\b(?:to|for)\s+(?:hermes|whom is|her miss)\b",
        text or "", re.I,
    ))


PROPOSAL_CUE_RE = re.compile(
    r"\b(?:shall we|what do you say|would you like|how about|what if|if you want|"
    r"i can|i(?:'|’)d like|i propose|i suggest|i want to|i(?:'|’)m going to|"
    r"i(?:'|’)m thinking|i am thinking|could we|let(?:'|’)s|"
    r"does (?:that|this) .{0,60}?sound right)\b",
    re.I,
)


CONTROLLER_FEEDBACK = {
    "up": "That feels good. Preserve or develop the current quality.",
    "down": "That does not feel good. Steer away from the most recent change.",
    "left": "This is too intense. Apply immediate bounded relief.",
    "right": "Ha—you call this a challenge? Treat that as encouragement for the next considered decision.",
}


def has_proposal_cue(text: str) -> bool:
    return bool(PROPOSAL_CUE_RE.search(text or ""))


SIGNAL_LANE_SCHEMA = {
    "type": "object",
    "properties": {
        "intensity": {"type":"number", "minimum":0, "maximum":100},
        "texture": {"type":"number", "minimum":0, "maximum":100},
        "vibration_rate": {"type":"number", "minimum":0, "maximum":100},
        "vibration_depth": {"type":"number", "minimum":0, "maximum":100},
        "secondary_rate": {"type":"number", "minimum":0, "maximum":100},
        "secondary_depth": {"type":"number", "minimum":0, "maximum":100},
        "modulation": {"type":"number", "minimum":0, "maximum":100},
    },
    "required": ["intensity", "texture", "vibration_rate", "vibration_depth"],
}


TOOLS = [
    {"type":"function","function":{"name":"signal_lab_get_state","description":"Read authoritative Signal Lab telemetry: exact live left/right signal, playback position, source, and operator-controlled headphone-monitor status. This is audible preview telemetry only; physical output is always false.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"signal_lab_apply","description":"Apply an offline two-lane semantic signal simulation inside the active operator-owned preset. Left is Stairway E1-to-E4; right is Moaner tip-to-base. This never produces live audio.","parameters":{"type":"object","properties":{"left":SIGNAL_LANE_SCHEMA,"right":SIGNAL_LANE_SCHEMA,"transition_seconds":{"type":"number","minimum":1,"maximum":60},"reason":{"type":"string"}},"required":["left","right","transition_seconds"]}}},
    {"type":"function","function":{"name":"signal_lab_heart_tempo","description":"Apply or disable Signal Lab's verified headphone-only heart-tempo lock. Match is 1:1 with current BPM; double is HR ×2; relief is 90%. Exactly one lane may be targeted so the other keeps its authored movement. Fresh confident telemetry is required. Never claim a lock unless this tool succeeds and verifies the effective AM rate.","parameters":{"type":"object","properties":{"target":{"type":"string","enum":["left","right"]},"mode":{"type":"string","enum":["escalation","double","relief","off"]},"reason":{"type":"string"}},"required":["mode"]}}},
    {"type":"function","function":{"name":"signal_lab_neutral","description":"Immediately clear the applied offline Signal Lab simulation state. This does not control Vector or any audio device.","parameters":{"type":"object","properties":{"reason":{"type":"string"}}}}},
    {"type":"function","function":{"name":"vector_get_state","description":"Read current live Vector state.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vector_set_preset","description":"Set Vector preset Baseline, A or B.","parameters":{"type":"object","properties":{"preset":{"type":"string","enum":["Baseline","A","B"]}},"required":["preset"]}}},
    {"type":"function","function":{"name":"vector_set_rolling_variety","description":"Turn Rolling Variety on or off.","parameters":{"type":"object","properties":{"enabled":{"type":"boolean"}},"required":["enabled"]}}},
    {"type":"function","function":{"name":"vector_neutral","description":"Immediately command Vector Neutral without stopping the engine.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vector_stop","description":"Stop Vector output at zero volume and leave the engine stopped until resumed.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vector_resume","description":"Resume a Vector engine that was stopped. Vector returns to buffering and follows incoming live signal.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vector_generated_motion_plan","description":"Ask Vector to generate incoming L0 from a bounded deterministic motion plan. Vector validates all limits; this never supplies raw samples. stroke_duration_ms is the one-way funscript-style time between endpoints.","parameters":{"type":"object","properties":{"pattern":{"type":"string","enum":["sine","triangle","breathing"]},"minimum":{"type":"number","minimum":0.0,"maximum":0.90},"maximum":{"type":"number","minimum":0.10,"maximum":1.0},"stroke_duration_ms":{"type":"number","minimum":125,"maximum":3000},"transition_seconds":{"type":"number","minimum":1,"maximum":15},"duration_seconds":{"type":"number","minimum":30,"maximum":600},"reason":{"type":"string"}},"required":["pattern","minimum","maximum","stroke_duration_ms","transition_seconds","duration_seconds"]}}},
    {"type":"function","function":{"name":"vector_generated_motion_hold","description":"Hold the current generated incoming L0 position while retaining the armed plan.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vector_generated_motion_resume","description":"Resume the currently armed generated incoming motion plan.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vector_use_authored_tcode","description":"Return Vector to its normal authored incoming T-code source.","parameters":{"type":"object","properties":{}}}},
    {"type":"function","function":{"name":"vector_set_texture","description":"Set captured Texture profile.","parameters":{"type":"object","properties":{"texture":{"type":"string","enum":["Smoothest","Smooth","Normal","Rough","Roughest"]}},"required":["texture"]}}},
    {"type":"function","function":{"name":"vector_set_primary_spatial","description":"Set top behaviour: moving focus or depth spread.","parameters":{"type":"object","properties":{"primary_spatial":{"type":"string","enum":["top_moving_focus","top_depth_spread"]}},"required":["primary_spatial"]}}},
    {"type":"function","function":{"name":"vector_set_variation","description":"Set captured Variation profile.","parameters":{"type":"object","properties":{"variation":{"type":"string","enum":["Still","Subtle","Normal","Lively","Wild"]}},"required":["variation"]}}},
    {"type":"function","function":{"name":"vector_set_top_focus","description":"Set anatomical top Spatial Focus. E1 glans, E2 shaft, E3 lower shaft, E4 root.","parameters":{"type":"object","properties":{"top_focus":{"type":"string","enum":["Glans Focus","Shaft Focus","Lower Shaft Focus","Root Focus","Top Sweep","Top Full"]}},"required":["top_focus"]}}},
    {"type":"function","function":{"name":"vector_set_bottom_focus","description":"Set bottom Spatial Focus. A prostate, B anus, C perineum/testicles.","parameters":{"type":"object","properties":{"bottom_focus":{"type":"string","enum":["Prostate Focus","Anal Focus","Perineum Focus","Bottom Sweep","Bottom Full"]}},"required":["bottom_focus"]}}},
    {"type":"function","function":{"name":"vector_top_spatial_gain","description":"Increase/decrease/restore operator-bounded top intensity while preserving focus.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["increase","decrease","restore"]}},"required":["action"]}}},
    {"type":"function","function":{"name":"vector_bottom_spatial_gain","description":"Increase/decrease/restore operator-bounded bottom intensity while preserving focus.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["increase","decrease","restore"]}},"required":["action"]}}},
    {"type":"function","function":{"name":"vector_adjust_stroke_range","description":"Adjust only the deterministic stroke range while preserving the current position bias. Use narrower/wider for relative changes; restore returns stroke range to authored 1.0 without changing position bias.","parameters":{"type":"object","properties":{"action":{"type":"string","enum":["narrower","wider","restore"]}},"required":["action"]}}},
    {"type":"function","function":{"name":"vector_set_targeting","description":"Apply a bounded deterministic targeted-stroke preset. base_prostate concentrates primary motion toward E4/root/base and secondary motion toward A/prostate. glans_perineum shifts the opposite way. broad/focused/tight progressively compress stroke range while increasing position bias. Use authored to restore the unmodified authored stroke path.","parameters":{"type":"object","properties":{"preset":{"type":"string","enum":["authored","base_prostate_broad","base_prostate_focused","base_prostate_tight","glans_perineum_broad","glans_perineum_focused","glans_perineum_tight"]}},"required":["preset"]}}},
    {"type":"function","function":{"name":"vector_tempo_window","description":"Temporarily time-warp the authored funscript at 0.5x for relief/contrast or 2x for challenge, then automatically restore authored tempo. Choose only 10,15,30,60,90,120 seconds. When current authored energy is challenging/testing, 2x is limited by Vector to 10-15 seconds; relaxing/moderate sections may sensibly use 30-60 seconds.","parameters":{"type":"object","properties":{"scale":{"type":"number","enum":[0.5,2.0]},"duration_seconds":{"type":"integer","enum":[10,15,30,60,90,120]}},"required":["scale","duration_seconds"]}}},
    {"type":"function","function":{"name":"vector_restore_modifiers","description":"Restore authored stroke targeting and authored 1x tempo, removing temporary deterministic modifiers.","parameters":{"type":"object","properties":{}}}},
]


TOOLS_BY_NAME = {item["function"]["name"]: item for item in TOOLS}

def _tool_subset(*names: str) -> List[Dict[str, Any]]:
    return [TOOLS_BY_NAME[n] for n in names if n in TOOLS_BY_NAME]


def is_signal_lab_intent(text: str) -> bool:
    normalised = re.sub(r"[^a-z0-9.]+", " ", str(text or "").lower())
    return bool(re.search(
        r"\b(?:signal lab|signal director|stairway|moaner|left lane|right lane|other lane|both lanes|both electrodes|"
        r"carrier|am depth|am rate|secondary vibration|fm movement|offline signal|rougher signal|smoother signal|"
        r"single slow wave|slow wave|gentler|softer|milder|less intense|stronger signal|weaker signal|pull (?:it|both) down)\b",
        normalised,
    ))


def signal_lab_apply_args_complete(args: Any) -> bool:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return False
    if not isinstance(args, dict):
        return False
    required_lane = {
        "intensity", "texture", "vibration_rate", "vibration_depth",
        "secondary_rate", "secondary_depth", "modulation",
    }
    left = args.get("left")
    right = args.get("right")
    transition = args.get("transition_seconds")
    def valid_lane(lane: Any) -> bool:
        return bool(
            isinstance(lane, dict)
            and required_lane.issubset(lane)
            and all(
                not isinstance(lane.get(key), bool)
                and isinstance(lane.get(key), (int, float))
                and 0 <= float(lane.get(key)) <= 100
                for key in required_lane
            )
        )

    return bool(
        valid_lane(left)
        and valid_lane(right)
        and not isinstance(transition, bool)
        and isinstance(transition, (int, float))
        and 5 <= float(transition) <= 15
        and isinstance(args.get("reason"), str)
        and bool(args.get("reason", "").strip())
    )


def asks_for_raw_signal_telemetry(text: str) -> bool:
    """Require an explicit request before speaking exact lane parameters."""
    raw = str(text or "")
    return bool(re.search(
        r"\b(?:exact|raw|numeric(?:al)?|numbers?|parameters?|telemetry|technical|read(?:\s+out)?|"
        r"carrier|hertz|hz|am depth|am rate|fm depth|fm rate|volume(?:s)?|percent(?:age)?s?)\b",
        raw,
        re.I,
    ))


def direct_signal_heart_tempo_action(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Recognise only explicit heart-tempo commands; sensation reports remain conversation."""
    t = norm_words(text)
    if not t or not re.search(r"\b(?:heart|heartbeat|heart rate|bpm|pulse)\b", t):
        return None
    if not re.search(
        r"\b(?:please|set|change|switch|use|apply|match|sync|lock|enable|disable|turn|stop|remove|"
        r"can (?:you|we)|could (?:you|we)|would you|shall (?:you|we)|let s|i want (?:you )?to)\b",
        t,
    ):
        return None
    if re.search(r"\b(?:disable|turn off|switch off|stop|remove|unsync|unlock)\b", t):
        return "signal_lab_heart_tempo", {"target": "left", "mode": "off", "reason": "Explicit heart-tempo disable request"}
    target = None
    if re.search(r"\bright(?: lane| channel| electrode)?\b", t):
        target = "right"
    elif re.search(r"\bleft(?: lane| channel| electrode)?\b", t):
        target = "left"
    if target is None:
        return None
    if re.search(r"\b(?:double|twice|2x|2 x|200 percent|two hundred percent)\b", t):
        mode = "double"
    elif re.search(r"\b(?:relief|relax|calm|slower|90 percent|ninety percent)\b", t):
        mode = "relief"
    else:
        mode = "escalation"
    return "signal_lab_heart_tempo", {
        "target": target,
        "mode": mode,
        "reason": f"Explicit {mode} heart-tempo request for the {target} lane",
    }


def asks_what_changed_in_signal(text: str) -> bool:
    """Recognise a comparative question without treating it as a control request."""
    raw = str(text or "")
    return bool(re.search(
        r"\b(?:what(?:'|’)s|what is|what has|what(?:'|’)ve|what have)\s+(?:just\s+)?changed\b|"
        r"\bhow (?:is|has) (?:the )?(?:signal|stairway|moaner|left lane|right lane)\s+(?:changed|different)\b",
        raw,
        re.I,
    ))


def extract_signal_lab_args(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        root = json.loads(raw[start:end])
    except Exception:
        return {}
    queue_items = [root]
    while queue_items:
        candidate = queue_items.pop(0)
        if signal_lab_apply_args_complete(candidate):
            return candidate
        if isinstance(candidate, dict):
            for key in ("arguments", "parameters", "signal_lab_apply", "command", "input"):
                nested = candidate.get(key)
                if isinstance(nested, str):
                    try: nested = json.loads(nested)
                    except Exception: nested = None
                if isinstance(nested, dict):
                    queue_items.append(nested)
    return {}


def extract_generated_motion_args(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    try:
        start, end = raw.index("{"), raw.rindex("}") + 1
        value = json.loads(raw[start:end])
    except Exception:
        return {}
    required = {"pattern", "minimum", "maximum", "stroke_duration_ms", "transition_seconds", "duration_seconds"}
    return value if isinstance(value, dict) and required.issubset(value) else {}


def bounded_signal_lab_fallback() -> Dict[str, Any]:
    return {
        "left": {"intensity": 45, "texture": 65, "vibration_rate": 28, "vibration_depth": 45,
                 "secondary_rate": 22, "secondary_depth": 30, "modulation": 25},
        "right": {"intensity": 52, "texture": 38, "vibration_rate": 42, "vibration_depth": 58,
                  "secondary_rate": 48, "secondary_depth": 55, "modulation": 32},
        "transition_seconds": 10,
        "reason": "Bounded local fallback after malformed Signal Lab design",
    }


def preserve_unrequested_signal_dimensions(design: Dict[str, Any], snapshot: Dict[str, Any],
                                           request: str) -> Dict[str, Any]:
    """Turn a model-produced full state into a faithful narrow edit."""
    current = (snapshot or {}).get("last_command") or {}
    current_left = current.get("left_semantic") or {}
    current_right = current.get("right_semantic") or {}
    if not current_left or not current_right:
        return design
    text = str(request or "")
    # Open-ended creation deliberately permits a complete redesign.
    if re.search(r"\b(?:create|design|build|new state|surprise|everything|whole signal|adjust it|change it)\b", text, re.I):
        return design
    allowed = set()
    if re.search(r"\b(?:intensity|volume|stronger|weaker|louder|quieter)\b", text, re.I):
        allowed.add("intensity")
    if re.search(r"\b(?:texture|smooth|rough)\w*\b", text, re.I):
        allowed.add("texture")
    if re.search(r"\b(?:tap|pulse|vibration)\w*\b", text, re.I):
        allowed.update(("vibration_rate", "vibration_depth"))
    if re.search(r"\b(?:secondary|beat|buzz)\w*\b", text, re.I):
        allowed.update(("secondary_rate", "secondary_depth"))
    if re.search(r"\b(?:modulation|drift|fm)\b", text, re.I):
        allowed.add("modulation")
    if not allowed:
        return design
    merged = dict(design)
    for lane_name, old in (("left", current_left), ("right", current_right)):
        lane = dict(merged.get(lane_name) or {})
        for key, value in old.items():
            if key not in allowed:
                lane[key] = value
        merged[lane_name] = lane
    merged["reason"] = str(merged.get("reason") or "") + "; unrequested dimensions preserved"
    return merged


def _signal_semantic_baseline(snapshot: Dict[str, Any], lane_name: str) -> Dict[str, float]:
    """Recover semantic controls from the last command or invert current live telemetry."""
    last = (snapshot or {}).get("last_command") or {}
    semantic = last.get(f"{lane_name}_semantic") or {}
    if semantic:
        return {key: float(semantic.get(key) or 0.0) for key in (
            "intensity", "texture", "vibration_rate", "vibration_depth",
            "secondary_rate", "secondary_depth", "modulation")}
    preset = (snapshot or {}).get("active_preset") or {}
    signal = ((snapshot or {}).get("live") or {}).get(f"{lane_name}_signal") or {}
    def invert(value: Any, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return min(100.0, max(0.0, (float(value or low) - low) * 100.0 / (high - low)))
    fmin, fmax = float(preset.get("frequency_min_hz") or 600), float(preset.get("frequency_max_hz") or 1200)
    vmin, vmax = float(preset.get("volume_min") or 40), float(preset.get("volume_max") or 100)
    amin, amax = float(preset.get("am_rate_min_hz") or .2), float(preset.get("am_rate_max_hz") or 5)
    admin, admax = float(preset.get("am_depth_min_percent") or 4), float(preset.get("am_depth_max_percent") or 12)
    smin, smax = float(preset.get("secondary_am_rate_min_hz") or .05), float(preset.get("secondary_am_rate_max_hz") or 2)
    sdmin, sdmax = float(preset.get("secondary_am_depth_min_percent") or 4), float(preset.get("secondary_am_depth_max_percent") or 12)
    carrier = float(signal.get("freq") or fmin)
    fm_depth = float(signal.get("fmDepth") or 0.0)
    fm_percent = 100.0 * fm_depth / carrier if carrier > 0 else 0.0
    fm_min = float(preset.get("fm_depth_min_percent") or 4)
    fm_max = float(preset.get("fm_depth_max_percent") or 12)
    modulation = 0.0 if fm_depth <= 0 else invert(fm_percent, fm_min, fm_max)
    return {
        "intensity": invert(signal.get("volume"), vmin, vmax),
        "texture": invert(signal.get("freq"), fmin, fmax),
        "vibration_rate": invert(signal.get("amFreq"), amin, amax),
        "vibration_depth": invert(signal.get("amDepth"), admin, admax),
        "secondary_rate": invert(signal.get("secondaryAmFreq"), smin, smax),
        "secondary_depth": invert(signal.get("secondaryAmDepth"), sdmin, sdmax),
        "modulation": modulation,
    }


def bound_signal_lab_step(design: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Apply controller-owned delta and modulation-complexity limits to one decision."""
    bounded = json.loads(json.dumps(design))
    preset = (snapshot or {}).get("active_preset") or {}
    vmin, vmax = float(preset.get("volume_min") or 40), float(preset.get("volume_max") or 100)
    baselines = {lane: _signal_semantic_baseline(snapshot, lane) for lane in ("left", "right")}
    live = (snapshot or {}).get("live") or {}
    families = {
        "primary": ("vibration_rate", "vibration_depth"),
        "secondary": ("secondary_rate", "secondary_depth"),
        "fm": ("modulation",),
    }
    family_increase = {family: 0.0 for family in families}
    for lane in ("left", "right"):
        requested = bounded.get(lane) or {}
        current = baselines[lane]
        for family, keys in families.items():
            family_increase[family] += sum(max(0.0, float(requested.get(key, current[key])) - current[key]) for key in keys)
    growing = [family for family, amount in family_increase.items() if amount > 0.001]
    permitted_family = max(growing, key=lambda family: family_increase[family]) if growing else None
    limited = False
    for lane in ("left", "right"):
        requested = dict(bounded.get(lane) or {})
        current = baselines[lane]
        current_volume = float((live.get(f"{lane}_signal") or {}).get("volume") or
                               (vmin + (vmax - vmin) * current["intensity"] / 100.0))
        desired_volume = vmin + (vmax - vmin) * float(requested.get("intensity", current["intensity"])) / 100.0
        desired_volume = min(vmax, max(vmin, desired_volume))
        limited_volume = min(current_volume * 1.10, max(current_volume * 0.90, desired_volume))
        requested["intensity"] = 0.0 if vmax <= vmin else (limited_volume - vmin) * 100.0 / (vmax - vmin)
        limited = limited or abs(limited_volume - desired_volume) > 0.001
        for key in ("texture", "vibration_rate", "vibration_depth", "secondary_rate", "secondary_depth", "modulation"):
            desired = float(requested.get(key, current[key]))
            clipped = min(100.0, max(0.0, min(current[key] + 10.0, max(current[key] - 10.0, desired))))
            requested[key] = clipped
            limited = limited or abs(clipped - desired) > 0.001
        for family, keys in families.items():
            if family != permitted_family and family_increase[family] > 0.001:
                for key in keys:
                    if requested[key] > current[key]:
                        requested[key] = current[key]
                        limited = True
        bounded[lane] = requested
    if limited:
        bounded["reason"] = str(bounded.get("reason") or "Signal Lab decision") + "; bounded step and modulation-complexity guard applied"
    return bounded


def signal_lab_preset_start_args(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Project both lanes into the active preset and apply its declared start point."""
    preset = (snapshot or {}).get("active_preset") or {}

    def position(value: Any, low: Any, high: Any, fallback: float) -> float:
        try:
            number, minimum, maximum = float(value), float(low), float(high)
        except (TypeError, ValueError):
            return fallback
        if maximum <= minimum:
            return 0.0
        return min(100.0, max(0.0, (number - minimum) * 100.0 / (maximum - minimum)))

    start_intensity = position(
        preset.get("volume_start"), preset.get("volume_min"), preset.get("volume_max"), 50.0)
    start_texture = position(
        preset.get("frequency_start_hz"), preset.get("frequency_min_hz"),
        preset.get("frequency_max_hz"), 50.0)
    lanes = {}
    for lane in ("left", "right"):
        # Inversion clamps every existing physical dimension to semantic 0-100,
        # while the two explicit preset start fields replace carrier and volume.
        semantic = _signal_semantic_baseline(snapshot, lane)
        semantic["intensity"] = start_intensity
        semantic["texture"] = start_texture
        lanes[lane] = semantic
    try:
        transition = float(preset.get("transition_min_seconds") or 5.0)
    except (TypeError, ValueError):
        transition = 5.0
    return {
        **lanes,
        "transition_seconds": transition,
        "reason": f"Establish operator preset baseline: {preset.get('name') or 'active preset'}",
    }


def deterministic_signal_relief(snapshot: Dict[str, Any], request: str) -> Dict[str, Any]:
    """Build a clearly audible volume-only reduction from current live telemetry."""
    text = norm_words(request)
    if not re.search(r"\b(?:reduce|lower|quieter|softer|gentler|milder|turn down|pull down|back off|ease)\b", text):
        return {}
    if not re.search(r"\b(?:volume|intensity|signal|lane|channel|electrode|both|it)\b", text):
        return {}
    target_lanes = {"left", "right"}
    if re.search(r"\bleft\b", text) and not re.search(r"\b(?:both|right)\b", text):
        target_lanes = {"left"}
    elif re.search(r"\bright\b", text) and not re.search(r"\b(?:both|left)\b", text):
        target_lanes = {"right"}
    preset = (snapshot or {}).get("active_preset") or {}
    live = (snapshot or {}).get("live") or {}
    vmin = float(preset.get("volume_min") or 40.0)
    vmax = float(preset.get("volume_max") or 100.0)
    if vmax <= vmin:
        return {}
    lanes = {}
    changes = []
    for lane_name in ("left", "right"):
        semantic = _signal_semantic_baseline(snapshot, lane_name)
        current = float((live.get(f"{lane_name}_signal") or {}).get("volume") or
                        (vmin + (vmax - vmin) * semantic["intensity"] / 100.0))
        if lane_name in target_lanes:
            # Ten absolute percentage points is easy to hear; a quarter of the
            # available range keeps narrow presets useful. Never cross the preset floor.
            step = max(10.0, (vmax - vmin) * 0.25)
            desired = max(vmin, current - step)
            semantic["intensity"] = (desired - vmin) * 100.0 / (vmax - vmin)
            changes.append(f"{lane_name} {current:.1f}->{desired:.1f}")
        lanes[lane_name] = semantic
    return {
        "left": lanes["left"], "right": lanes["right"], "transition_seconds": 8.0,
        "reason": "Deterministic live volume relief: " + ", ".join(changes) +
                  "; all non-volume dimensions preserved",
    }


def select_tools_for_turn(text: str, pending_proposal: str = "") -> List[Dict[str, Any]]:
    """Return the smallest useful Vector tool set for an ambiguous Qwen turn.

    Clear commands are handled by direct_vector_action before this function is used.
    Ordinary conversation intentionally receives no tools.  For an approval such as
    "let's do that", the pending proactive proposal supplies the missing intent.
    """
    raw = f"{pending_proposal} {text}".strip().lower()
    t = re.sub(r"[^a-z0-9.]+", " ", raw)

    if is_signal_lab_intent(raw):
        return _tool_subset("signal_lab_get_state", "signal_lab_apply", "signal_lab_heart_tempo", "signal_lab_neutral")

    groups = []
    def add(*names: str) -> None:
        for n in names:
            if n not in groups:
                groups.append(n)

    # Explicit families. Keep these intentionally narrow; Qwen performs better
    # with a handful of relevant tools than with the whole Director catalogue.
    if re.search(r"\b(?:tempo|speed|pace|double|twice|half speed|faster|slower)\b", t):
        add("vector_tempo_window", "vector_restore_modifiers")
    if re.search(r"\b(?:stroke|stroke range|target|targeted|concentrat|narrow|widen|broaden)\w*\b", t):
        add("vector_adjust_stroke_range", "vector_set_targeting", "vector_restore_modifiers")
    if re.search(r"\b(?:texture|smooth|rough)\w*\b", t):
        add("vector_set_texture")
    if re.search(r"\b(?:variation|variety|still|subtle|lively|wild)\b", t):
        add("vector_set_variation", "vector_set_rolling_variety")
    if re.search(r"\b(?:moving focus|depth spread|spatial|spread)\b", t):
        add("vector_set_primary_spatial")
    if re.search(r"\b(?:glans|shaft|root|top|upper)\b", t):
        add("vector_set_top_focus", "vector_top_spatial_gain")
    if re.search(r"\b(?:prostate|anal|anus|perineum|bottom|secondary|base)\b", t):
        add("vector_set_bottom_focus", "vector_bottom_spatial_gain")
    if re.search(r"\b(?:intensity|stronger|weaker|power|increase|decrease|more intense|less intense)\b", t):
        add("vector_top_spatial_gain", "vector_bottom_spatial_gain")
    if re.search(r"\b(?:preset|baseline)\b", t):
        add("vector_set_preset")
    if re.search(r"\b(?:stop vector|vector stop|resume vector|start vector|neutral)\b", t):
        add("vector_stop", "vector_resume", "vector_neutral")

    # Deliberately broad Director requests may need choice, but still cap the menu.
    if not groups and re.search(r"\b(?:do something|something interesting|change things|mix it up|surprise me|make it interesting)\b", t):
        add(
            "vector_set_texture", "vector_set_variation",
            "vector_set_primary_spatial", "vector_set_targeting",
            "vector_tempo_window",
        )

    return _tool_subset(*groups[:6])


@dataclass
class Timings:
    ptt_release: float = 0.0
    stt_done: float = 0.0
    llm_start: float = 0.0
    llm_done: float = 0.0
    action_seconds: float = 0.0
    tts_request: float = 0.0
    tts_first_audio: float = 0.0


@dataclass
class VoiceJob:
    voice_id: str
    text: str
    timings: Timings
    generation: int
    done: Optional[threading.Event] = None
    opening: bool = False
    announcement_receipt: int = 0


class GwendolynCore:
    def __init__(self, config: Dict[str, Any], ui_queue: queue.Queue):
        self.cfg = config
        self.ui = ui_queue
        self.profile = load_profile()
        self.private_lexicon = load_private_lexicon()
        self.memory = load_memory()
        self.memory_lock = threading.Lock()
        self.memory_session_id = time.strftime("%Y%m%d-%H%M%S")
        self.session_journal_started_monotonic = time.monotonic()
        self.session_journal_started_local = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.session_journal_lock = threading.Lock()
        self.session_journal: List[Dict[str, Any]] = []
        self.system_prompt = self.build_effective_system_prompt()
        candidate_marker = ROOT / "active_ab_candidate.txt"
        if candidate_marker.exists():
            log(f"A-B CANDIDATE active={candidate_marker.read_text(encoding='utf-8').strip()!r}")
        effective_profile = self.effective_director_profile()
        log(
            f"VOICE DELIVERY={self.profile.get('speech_chunking', 'Off')!r} "
            f"verbosity={effective_profile.get('speech_verbosity', 4)}"
        )
        log(
            "EXPRESSION TUNING "
            f"preset={self.profile.get('expression_preset','Reference')!r} "
            f"metaphor={tuning_value(effective_profile,'metaphor_density',4)}/5 "
            f"pressure={tuning_value(effective_profile,'director_pressure',4)}/5 "
            f"questions={tuning_value(effective_profile,'question_frequency',3)}/5 "
            f"proposals={tuning_value(effective_profile,'proposal_frequency',3)}/5 "
            f"sensation_carryover={tuning_value(effective_profile,'sensation_carryover',4)}/5"
        )
        self.history: List[Dict[str, Any]] = []
        self.recent_spoken_replies: List[str] = []
        self.repetition_fallback_index = 0
        self.recent_proactive_actions: List[str] = []
        self.last_live_state: Dict[str, Any] = {}
        # v0.32: authoritative semantic state is tracked separately from the operator baseline.
        # Only live /v1/state observations populate this cache; baseline values are planning targets only.
        self.current_vector_semantic: Dict[str, Any] = {}
        self.stt_model: Optional[WhisperModel] = None
        self.stt_ready = threading.Event()
        self.ollama_ready = threading.Event()
        self.stop_audio = threading.Event()
        self.record_lock = threading.Lock()
        self.recording = False
        self.record_cancelled = False
        self.audio_chunks: List[np.ndarray] = []
        self.input_stream = None
        self.current_state = "STARTING"
        self.voice_activation_monitor = None
        self.session = requests.Session()
        # Keep Ollama on its own HTTP session. If one local LLM request wedges,
        # we can discard that connection pool without disturbing Vector or TTS.
        self.ollama_session = requests.Session()
        self.ollama_session_lock = threading.Lock()
        self.tts = TTSAdapter(self.cfg, self.session)
        self.output_stream = None
        self.output_lock = threading.Lock()
        self.opening_spoken = False
        self.voice_tx_counter = 0
        self.voice_tx_lock = threading.Lock()
        self.voice_tx_seen = set()
        self.last_action_ledger = {"requested": [], "executed": [], "failed": []}
        now = time.monotonic()
        self.last_meaningful_vector_change_at = now
        self.last_conversation_activity_at = now
        self.last_proactive_proposal_at = 0.0
        self.proactive_snooze_until = 0.0
        self.proactive_inflight = False
        self.proactive_lock = threading.Lock()
        self.pending_proposal_text = ""
        self.pending_proposal_action = None
        self.pending_proposal_at = 0.0
        self.pending_signal_lab_proposal = ""
        self.signal_lab_context_until = 0.0
        self.conducted_session_active = False
        self.conducted_session_held = False
        self.conducted_session_started_at = 0.0
        self.conducted_session_ends_at = 0.0
        self.conducted_session_duration_minutes = 0.0
        self.conducted_session_completed_at = 0.0
        self.conducted_session_neutral_confirmed = False
        self.last_conducted_decision_at = 0.0
        self.conducted_decision_count = 0
        self.conducted_start_pending = False
        self.conducted_consent_pending = False
        self.conducted_final_wave_announced = False
        self.conducted_narrative_anchor = ""
        # Every user turn advances this revision. Autonomous Signal Lab work
        # captures the revision before generation and must still own it before
        # applying a command; speech therefore cannot be overwritten by an
        # already-in-flight conductor decision.
        self.signal_user_revision = 0
        self.conducted_user_priority_until = 0.0
        self.vector_autonomous_active = False
        self.vector_autonomous_held = False
        self.vector_autonomous_decision_count = 0
        # Heart Phase Lab is perception only. In Vector mode it may pace the
        # generated L0 motion, but it never changes travel, gain or targeting.
        self.vector_motion_plan: Optional[Dict[str, Any]] = None
        self.heart_tempo_lock = threading.Lock()
        self.heart_tempo = {"available": False, "bpm": None, "ratio": None,
                            "target_cpm": None, "confidence": 0.0, "received_at": 0.0}
        self.heart_tempo_server = None
        self.heart_tempo_thread = None
        self.heart_tempo_last_apply_at = 0.0
        self.heart_tempo_last_stroke_ms = None
        self.restim_sensors = None
        self.sensor_last_checkin_at = 0.0
        self.sensor_last_checkin_observation = ""
        self.sensor_pending_feedback = None
        self.session_finish_pending = False
        self.session_finish_stage = ""
        self.session_closure_reflection = ""
        self.last_excursion_change_at = 0.0
        self.recovery_snooze_until = 0.0
        self.last_proposal_rejected_noop = False
        self.last_rejected_noop_action: Optional[Tuple[str, Dict[str, Any]]] = None
        self.vad_debounce_until = 0.0
        self.empty_stt_count = 0
        self.vad_adaptive_multiplier = 1.0
        self.vad_last_empty_at = 0.0
        # Narrative Director v2: session-local dramatic memory above the grounded Vector layer.
        self.narrative_rng = random.Random()
        self.narrative_recent_beats: List[str] = []
        self.narrative_recent_motifs: List[str] = []
        self.narrative_callback_queue: List[str] = []
        self.narrative_recent_modes: List[str] = []
        self.narrative_recent_questions: List[str] = []
        self.narrative_turn_counter = 0
        self.narrative_arc_template = self._choose_narrative_arc_template()
        self.session_brief_milestones = self._build_session_brief_milestones()
        self.session_brief_completed: List[str] = []
        self.conductor = SessionConductor(self.session_brief_milestones,
                                          self.session_journal_started_monotonic, self._journal_event)
        self.director_thread = threading.local()
        if self.session_brief_milestones:
            log(f"SESSION BRIEF conductor loaded milestones={[m['label'] for m in self.session_brief_milestones]}")
        # Vector State Arc: excursions are intentionally finite. After a small
        # sequence of meaningful changes, the Director should create contrast by
        # returning one part of the system toward the operator-defined baseline.
        self.intervention_count = 0
        self.intervention_target = self._choose_intervention_target()
        self.intervention_recent_actions: List[str] = []
        self.intervention_sequence_number = 1
        log(f"NARRATIVE ARC selected={self.narrative_arc_template!r}")
        log(f"STATE ARC sequence={self.intervention_sequence_number} target={self.intervention_target} baseline={self._baseline_summary()}")

        # Voice architecture: exactly one TTS request may be active.
        # There is at most one pending job; newer pending speech replaces older speech.
        self.voice_generation = 0
        self.voice_generation_lock = threading.Lock()
        self.voice_queue: queue.Queue = queue.Queue(maxsize=1)
        self.voice_worker = threading.Thread(target=self._voice_worker_loop, daemon=True, name="GwendolynVoiceWorker")
        self.voice_worker.start()
        if SESSION_MODE == "vector":
            self.start_heart_tempo_listener()
        # Restim perception is shared by Vector and Signal Lab.  It is strictly
        # observational: the bridge never writes to Restim or either controller.
        self.restim_sensors = RestimSensorBridge()
        log(f"RESTIM SENSOR perception started read-only IMU + AS5311 mode={SESSION_MODE}")

    def start_heart_tempo_listener(self) -> None:
        """Receive validated browser telemetry on loopback for Vector pacing."""
        core = self

        class HeartTempoHandler(BaseHTTPRequestHandler):
            server_version = "GwendolynHeartTempo/0.1"

            def log_message(self, _format, *_args):
                return

            def _headers(self, status=200):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:8766")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
                self.end_headers()

            def do_OPTIONS(self):
                self._headers(204)

            def do_POST(self):
                if self.path != "/v1/heart-tempo":
                    self._headers(404); self.wfile.write(b'{"ok":false}'); return
                if self.headers.get("Origin", "") not in {"http://127.0.0.1:8766", "http://localhost:8766"}:
                    self._headers(403); self.wfile.write(b'{"ok":false,"error":"origin"}'); return
                try:
                    size = min(4096, int(self.headers.get("Content-Length", "0")))
                    payload = json.loads(self.rfile.read(size).decode("utf-8"))
                    values = {}
                    for key, low, high in (("bpm", 30.0, 240.0), ("ratio", 0.8, 2.0),
                                           ("confidence", 0.0, 100.0)):
                        raw = payload.get(key)
                        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                            raise ValueError(f"{key} must be numeric")
                        values[key] = float(raw)
                        if not low <= values[key] <= high:
                            raise ValueError(f"{key} out of range")
                    observed = {
                        "available": values["confidence"] >= 70.0,
                        "bpm": round(values["bpm"], 3), "ratio": round(values["ratio"], 3),
                        "target_cpm": round(values["bpm"] * values["ratio"], 3),
                        "confidence": round(values["confidence"], 1), "received_at": time.time(),
                    }
                    with core.heart_tempo_lock:
                        core.heart_tempo = observed
                    self._headers(200)
                    self.wfile.write(json.dumps({"ok": True, **observed}).encode("utf-8"))
                except Exception as exc:
                    self._headers(400)
                    self.wfile.write(json.dumps({"ok": False, "error": str(exc)}).encode("utf-8"))

        try:
            self.heart_tempo_server = ThreadingHTTPServer(("127.0.0.1", 18767), HeartTempoHandler)
            self.heart_tempo_thread = threading.Thread(
                target=self.heart_tempo_server.serve_forever, daemon=True, name="HeartTempoListener")
            self.heart_tempo_thread.start()
            log("HEART TEMPO listener ready http://127.0.0.1:18767/v1/heart-tempo")
        except OSError as exc:
            log(f"HEART TEMPO listener unavailable: {exc}")

    def vector_heart_tempo_tick(self) -> None:
        """Pace generated Vector motion from fresh heart data; change tempo only."""
        if not bool(self.cfg.get("vector_heart_tempo_follow", True)):
            return
        if not (self.vector_autonomous_active and not self.vector_autonomous_held
                and self.vector_autonomous_generates_motion() and self.vector_motion_plan):
            return
        with self.heart_tempo_lock:
            heart = dict(self.heart_tempo)
        age = max(0.0, time.time() - float(heart.get("received_at") or 0.0))
        if not heart.get("available") or age > 3.0:
            return
        target_cpm = float(heart.get("target_cpm") or 0.0)
        if target_cpm <= 0:
            return
        # stroke_duration_ms is one-way. A full sine cycle has two strokes.
        target_ms = max(125.0, min(3000.0, 30000.0 / target_cpm))
        previous = float(self.heart_tempo_last_stroke_ms or self.vector_motion_plan.get("stroke_duration_ms") or target_ms)
        settled_ms = previous + (target_ms - previous) * 0.65
        now = time.monotonic()
        if now - self.heart_tempo_last_apply_at < 2.5 or abs(settled_ms - previous) < 8.0:
            return
        plan = dict(self.vector_motion_plan)
        plan["stroke_duration_ms"] = int(round(settled_ms))
        plan["transition_seconds"] = 3
        plan["reason"] = (f"Fresh heart tempo {heart['bpm']:.1f} BPM at {heart['ratio']:.0%}; "
                          "tempo-only autonomous pacing")
        result = self.vector_post("/v1/generated-motion/plan", plan, timeout=3.0)
        self.vector_motion_plan = plan
        self.heart_tempo_last_stroke_ms = settled_ms
        self.heart_tempo_last_apply_at = now
        log(f"HEART TEMPO Vector paced target={target_cpm:.1f}cpm stroke={plan['stroke_duration_ms']}ms "
            f"confidence={heart['confidence']:.0f} accepted={bool(result)}")

    def vector_heart_tempo_snapshot(self) -> Dict[str, Any]:
        """Return prompt-safe heart telemetry with freshness and application state."""
        with self.heart_tempo_lock:
            heart = dict(self.heart_tempo)
        age = max(0.0, time.time() - float(heart.get("received_at") or 0.0))
        fresh = bool(heart.get("available") and age <= 3.0)
        applying = bool(
            fresh and self.cfg.get("vector_heart_tempo_follow", True)
            and self.vector_autonomous_active and not self.vector_autonomous_held
            and self.vector_autonomous_generates_motion() and self.vector_motion_plan
        )
        return {
            "fresh": fresh,
            "bpm": heart.get("bpm") if fresh else None,
            "ratio": heart.get("ratio") if fresh else None,
            "target_cycles_per_minute": heart.get("target_cpm") if fresh else None,
            "confidence": heart.get("confidence") if fresh else None,
            "age_seconds": round(age, 2) if heart.get("received_at") else None,
            "tempo_following_active": applying,
            "scope": "generated motion tempo only",
        }

    def _handle_vector_heart_query(self, text: str, timings: Timings) -> bool:
        if SESSION_MODE != "vector" or not re.search(r"\b(?:heart|heartbeat|heart rate|bpm|pulse)\b", text, re.I):
            return False
        if not re.search(r"\b(?:see|read|detect|receive|receiving|connected|visible|have|got|know)\b", text, re.I):
            return False
        heart = self.vector_heart_tempo_snapshot()
        if heart["fresh"]:
            reply = (f"Yes. I can see a fresh heartbeat reading of {float(heart['bpm']):.1f} BPM "
                     f"at {float(heart['confidence']):.0f}% confidence.")
            if heart["tempo_following_active"]:
                reply += (f" Generated Vector motion is following it at {float(heart['ratio']):.0%}, "
                          f"or {float(heart['target_cycles_per_minute']):.1f} cycles per minute.")
            else:
                reply += " I can observe it, but heart-tempo following is not currently active."
        else:
            reply = "No fresh, confident heartbeat reading is reaching me at the moment."
            if heart.get("age_seconds") is not None:
                reply += f" The last telemetry is {float(heart['age_seconds']):.1f} seconds old."
        self._emit_local_turn(text, reply, timings, "VECTOR-HEART")
        return True

    def close_heart_tempo_listener(self) -> None:
        if self.heart_tempo_server is not None:
            try: self.heart_tempo_server.shutdown()
            except Exception: pass
        if getattr(self, "restim_sensors", None) is not None:
            self.restim_sensors.close()

    def restim_sensor_snapshot(self) -> Dict[str, Any]:
        if self.restim_sensors is None:
            return {"read_only": True, "fresh": False, "observation": "unavailable"}
        return self.restim_sensors.snapshot()

    # Compatibility name retained for existing Vector context/tests.
    def vector_sensor_snapshot(self) -> Dict[str, Any]:
        return self.restim_sensor_snapshot()

    def restim_sensor_checkin_tick(self) -> None:
        """Record sustained response using the session convention; never act directly."""
        session_active = (
            (SESSION_MODE == "vector" and self.vector_autonomous_active and not self.vector_autonomous_held)
            or (SESSION_MODE == "signal_lab" and self.conducted_session_active and not self.conducted_session_held)
        )
        if not session_active:
            return
        if self.current_state != "READY" or self.recording or self.proactive_inflight:
            return
        state = self.restim_sensor_snapshot()
        if not state.get("fresh") or float(state.get("stable_seconds") or 0) < 4.0:
            return
        movement = str((state.get("movement") or {}).get("state") or "unavailable")
        clenching = str((state.get("clenching") or {}).get("state") or "unavailable")
        if movement in {"unavailable", "still"} and clenching in {"unavailable", "quiet"}:
            return
        observation = str(state.get("observation") or "")
        now = time.monotonic()
        if observation == self.sensor_last_checkin_observation or now - self.sensor_last_checkin_at < 120.0:
            return
        if now - self.last_conversation_activity_at < 20.0:
            return
        self.sensor_last_checkin_at=now; self.sensor_last_checkin_observation=observation
        event={"observed_at":time.time(),"sensor_state":state,
               "assumed_meaning":"positive response",
               "meaning_convention":"session-relative; explicit user correction overrides",
               "direct_control_action":False}
        self._journal_event(f"{SESSION_MODE}_sensor_observation", event)
        log(f"RESTIM SENSOR positive response observed silently observation={observation!r}")

    # Compatibility entry point for the existing Vector service loop.
    def vector_sensor_checkin_tick(self) -> None:
        self.restim_sensor_checkin_tick()

    def current_voice_generation(self) -> int:
        with self.voice_generation_lock:
            return self.voice_generation

    def invalidate_voice(self, reason: str = "barge-in") -> None:
        """Invalidate active/pending speech and discard any not-yet-started job."""
        with self.voice_generation_lock:
            self.voice_generation += 1
            generation = self.voice_generation
        self.stop_audio.set()
        dropped = []
        while True:
            try:
                job = self.voice_queue.get_nowait()
            except queue.Empty:
                break
            dropped.append(job.voice_id)
            if job.done:
                job.done.set()
            self.voice_queue.task_done()
        if dropped:
            log(f"VOICE QUEUE invalidated generation={generation} reason={reason} dropped={dropped}")
        else:
            log(f"VOICE QUEUE invalidated generation={generation} reason={reason} dropped=[]")

    def enqueue_voice(self, text: str, timings: Timings, voice_id: Optional[str] = None,
                      wait: bool = False, opening: bool = False) -> str:
        voice_id = voice_id or self.next_voice_tx("VOICE")
        if not text or self._autonomous_interrupted():
            return voice_id
        if not self.claim_voice_tx(voice_id):
            return voice_id

        generation = self.current_voice_generation()
        done = threading.Event() if wait else None
        job = VoiceJob(
            voice_id=voice_id,
            text=text,
            timings=timings,
            generation=generation,
            done=done,
            opening=opening,
            announcement_receipt=self.conductor.receipt(text),
        )

        # Latest-reply-wins for the single pending slot.
        try:
            old = self.voice_queue.get_nowait()
        except queue.Empty:
            old = None
        if old is not None:
            log(f"VOICE QUEUE replacing pending {old.voice_id} with {voice_id}")
            if old.done:
                old.done.set()
            self.voice_queue.task_done()

        self.voice_queue.put_nowait(job)
        log(f"VOICE QUEUE queued {voice_id} generation={generation} pending={self.voice_queue.qsize()}")

        if done is not None:
            # Opening line uses this so READY is not announced until startup speech completes.
            done.wait(timeout=120)
        return voice_id

    def _voice_worker_loop(self) -> None:
        while True:
            job: VoiceJob = self.voice_queue.get()
            try:
                current = self.current_voice_generation()
                if job.generation != current:
                    log(f"VOICE {job.voice_id} STALE BEFORE TTS generation={job.generation} current={current}")
                    continue
                log(f"VOICE QUEUE start {job.voice_id} generation={job.generation}")
                self._speak_now(job.text, job.timings, job.voice_id, job.generation, job.announcement_receipt)
            except Exception:
                log("VOICE WORKER ERROR\n" + traceback.format_exc())
            finally:
                if job.done:
                    job.done.set()
                self.voice_queue.task_done()
                log(f"VOICE QUEUE complete {job.voice_id} pending={self.voice_queue.qsize()}")

    def next_voice_tx(self, prefix: str = "TURN") -> str:
        with self.voice_tx_lock:
            self.voice_tx_counter += 1
            return f"{prefix}-{self.voice_tx_counter:04d}"

    def claim_voice_tx(self, voice_id: str) -> bool:
        with self.voice_tx_lock:
            if voice_id in self.voice_tx_seen:
                log(f"VOICE {voice_id} DUPLICATE BLOCKED")
                return False
            self.voice_tx_seen.add(voice_id)
            return True

    def maybe_speak_opening(self) -> None:
        if self.opening_spoken:
            return
        selected_opening = self.director_voice_profile().get("opening", "") if bool(self.cfg.get("director_personality_profile", True)) else ""
        line = (selected_opening or self.profile.get("opening_line") or "").strip()
        if not line:
            self.opening_spoken = True
            return
        self.opening_spoken = True
        voice_id = self.next_voice_tx("OPEN")
        line = self._publish_reply(line)
        self.enqueue_voice(line, Timings(), voice_id=voice_id, wait=True, opening=True)

    def update_profile(self, profile: Dict[str, Any]) -> None:
        self.profile = dict(DEFAULT_PROFILE)
        self.profile.update(profile)
        save_profile(self.profile)
        self.system_prompt = self.build_effective_system_prompt()
        self.emit("log", f"Director profile saved: {self.profile.get('character_name','Gwendolyn')} / {self.profile.get('autonomy','Reactive')}")

    def update_private_lexicon(self, lexicon: Dict[str, Any]) -> None:
        self.private_lexicon = dict(DEFAULT_PRIVATE_LEXICON)
        self.private_lexicon.update(lexicon or {})
        save_private_lexicon(self.private_lexicon)
        self.system_prompt = self.build_effective_system_prompt()
        self.emit("log", f"Private lexicon saved locally: language={self.private_lexicon.get('language_level','Natural')}")

    def director_voice_key(self) -> str:
        key = str(self.cfg.get("director_voice") or "gwendolyn").lower()
        return key if key in DIRECTOR_VOICES else "gwendolyn"

    def director_voice_profile(self) -> Dict[str, Any]:
        return dict(DIRECTOR_VOICES[self.director_voice_key()])

    def director_name(self) -> str:
        return self.director_voice_profile()["name"]

    def effective_director_profile(self) -> Dict[str, Any]:
        selected = self.director_voice_profile()
        effective = dict(self.profile)
        if bool(self.cfg.get("director_personality_profile", True)):
            effective.update(selected.get("defaults") or {})
        original_name = str(effective.get("character_name") or "Gwendolyn")
        effective["character_name"] = selected["name"]
        description = str(effective.get("character_description") or "")
        effective["character_description"] = (description + "\nSelected voice personality: " + selected["personality"]).strip()
        relationship = str(effective.get("roleplay_description") or "")
        if original_name and original_name != selected["name"]:
            relationship = re.sub(rf"\b{re.escape(original_name)}\b", selected["name"], relationship)
        effective["roleplay_description"] = relationship
        return effective

    def build_effective_system_prompt(self) -> str:
        effective = self.effective_director_profile()
        if bool(self.cfg.get("optimized_27b_prompt", False)):
            base = build_optimized_27b_prompt(effective, self.private_lexicon)
        else:
            base = build_system_prompt(effective, self.private_lexicon)
        return base + self.personality_operating_context() + self.memory_prompt_context()

    def personality_operating_context(self) -> str:
        if not bool(self.cfg.get("director_personality_profile", True)):
            return ""
        selected = self.director_voice_profile()
        policy = selected.get("operating_profile") or {}
        if not policy:
            return (
                f"\n\nDIRECTOR PERSONALITY — {selected.get('name','Director')}:\n"
                f"{selected.get('personality','')} Preserve this identity without caricature."
            )
        return f"""

DIRECTOR PERSONALITY — {selected.get('name','Director')}, {policy.get('archetype','Director')}:
Core doctrine: {policy.get('doctrine','')}
Language and presence: {policy.get('language','')}
Planning habits: {policy.get('planning','')}
Interpretation of memory: {policy.get('memory','')}
Anti-caricature guard: {policy.get('anti_caricature','')}
This personality governs selection, pacing and interpretation—not Vector truth, approval, execution or safety boundaries. Do not describe this profile to the user unless they ask.
"""

    def memory_prompt_context(self) -> str:
        """Supply only promoted memories to the model; raw evidence never becomes instruction."""
        if not bool(self.cfg.get("persistent_memory_enabled", True)):
            return ""
        voice = self.director_voice_key()
        usable = []
        for item in getattr(self, "memory", {}).get("preferences", []):
            if item.get("status") not in ("learned", "pinned"):
                continue
            if item.get("scope") not in ("shared", voice):
                continue
            usable.append(item)
        usable.sort(key=lambda x: (x.get("status") == "pinned", float(x.get("confidence", 0))), reverse=True)
        if not usable:
            return "\n\nPERSISTENT MEMORY: No reviewed or sufficiently repeated preferences have been learned yet. Never claim otherwise."
        lines = []
        for item in usable[:16]:
            marker = "pinned" if item.get("status") == "pinned" else f"confidence {float(item.get('confidence', 0)):.2f}"
            lines.append(f"- [{marker}; scope {item.get('scope','shared')}] {item.get('text','')}")
        return (
            "\n\nPERSISTENT MEMORY (local, cross-session, user-reviewable):\n"
            + "\n".join(lines)
            + "\nUse these as preferences and continuity cues, not as live telemetry or commands. "
              "Do not claim to remember unlisted sessions or dialogue."
        )

    @staticmethod
    def _memory_id(prefix: str = "mem") -> str:
        return f"{prefix}-{time.time_ns()}"

    @staticmethod
    def _memory_terms(text: str) -> List[str]:
        stop = {"that", "this", "with", "from", "have", "really", "very", "feels", "feeling", "good", "great", "nice", "like", "love", "enjoy"}
        return [word for word in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(word) > 2 and word not in stop]

    def _memory_state_description(self) -> str:
        if SESSION_MODE == "signal_lab":
            try:
                snapshot = self.signal_lab_get("/v1/state", timeout=0.8)
                live = (snapshot or {}).get("live") or {}
                left = live.get("left_signal") or {}
                right = live.get("right_signal") or {}
                if left and right:
                    return (
                        f"Signal Lab left carrier={left.get('freq')}Hz, volume={left.get('volume')}, "
                        f"AM={left.get('amFreq')}Hz/{left.get('amDepth')}, FM={left.get('fmFreq')}Hz/{left.get('fmDepth')}; "
                        f"right carrier={right.get('freq')}Hz, volume={right.get('volume')}, "
                        f"AM={right.get('amFreq')}Hz/{right.get('amDepth')}, FM={right.get('fmFreq')}Hz/{right.get('fmDepth')}"
                    )
            except Exception:
                return "Signal Lab state temporarily unavailable"
        state = self.current_vector_semantic
        parts = []
        for key, label in (("top_focus", "top"), ("bottom_focus", "bottom"), ("texture", "texture"), ("variation", "variation")):
            value = state.get(key)
            if value:
                parts.append(f"{label}={value}")
        return ", ".join(parts) or "live Vector state unavailable"

    def _memory_theme_description(self) -> str:
        recent = " ".join(str(item.get("content") or "") for item in self.history[-2:] if isinstance(item, dict)).lower()
        themes = [term for term in ("lingerie", "stockings", "high heels", "control", "teasing", "slow rhythm", "relentless", "user-defined sensation") if term in recent]
        return ", ".join(themes[:4]) or "no distinct narrative theme"

    def _save_memory_and_refresh(self) -> None:
        save_memory(self.memory)
        self.system_prompt = self.build_effective_system_prompt()
        self.emit("memory_changed", None)

    def remember_explicitly(self, preference: str, scope: str = "shared") -> Dict[str, Any]:
        text = re.sub(r"\s+", " ", preference or "").strip(" .")
        item = {
            "id": self._memory_id(), "text": text, "scope": scope,
            "status": "learned", "confidence": 0.98, "source": "explicit",
            "evidence_count": 1, "sessions": [self.memory_session_id],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with self.memory_lock:
            self.memory.setdefault("preferences", []).append(item)
            self._save_memory_and_refresh()
        log(f"MEMORY explicit preference learned id={item['id']} scope={scope!r}")
        return item

    def observe_preference_evidence(self, user_text: str) -> None:
        """Store contextual evidence; promote only after recurrence across three sessions."""
        if not bool(self.cfg.get("persistent_memory_enabled", True)):
            return
        positive = self._looks_like_positive_feedback(user_text)
        negative = bool(re.search(r"\b(?:i don(?:'|’)t like|i dislike|i hate|that feels bad|not enjoyable|don(?:'|’)t do that again)\b", user_text or "", re.I))
        if not positive and not negative:
            return
        state = self._memory_state_description()
        theme = self._memory_theme_description()
        polarity = "negative" if negative else "positive"
        signature = self._normalized_reply(f"{polarity} {state} {theme} {self.director_voice_key()}")
        with self.memory_lock:
            evidence = self.memory.setdefault("evidence", [])
            match = next((x for x in evidence if x.get("signature") == signature), None)
            if match is None:
                match = {
                    "id": self._memory_id("ev"), "signature": signature, "polarity": polarity,
                    "state": state, "theme": theme, "scope": self.director_voice_key(),
                    "sessions": [], "mentions": 0, "examples": [],
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                evidence.append(match)
            match["mentions"] = int(match.get("mentions", 0)) + 1
            if self.memory_session_id not in match["sessions"]:
                match["sessions"].append(self.memory_session_id)
            match["examples"] = (list(match.get("examples") or []) + [user_text])[-3:]
            match["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            # Repetition within one run is not enough: promotion requires three
            # separately launched sessions with the same contextual signature.
            if polarity == "positive" and len(match["sessions"]) >= 3:
                existing = next((x for x in self.memory.get("preferences", []) if x.get("evidence_signature") == signature), None)
                if existing is None:
                    self.memory.setdefault("preferences", []).append({
                        "id": self._memory_id(), "text": f"the user repeatedly responds positively when {state}; associated theme: {theme}.",
                        "scope": self.director_voice_key(), "status": "learned", "confidence": 0.82,
                        "source": "repeated_evidence", "evidence_signature": signature,
                        "evidence_count": match["mentions"], "sessions": list(match["sessions"]),
                        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    log(f"MEMORY evidence promoted after sessions={len(match['sessions'])} signature={signature!r}")
            self.memory["evidence"] = evidence[-200:]
            self._save_memory_and_refresh()
        log(f"MEMORY evidence recorded polarity={polarity} sessions={len(match['sessions'])} state={state!r} theme={theme!r}")

    def memory_rows(self) -> List[Dict[str, Any]]:
        rows = list(self.memory.get("preferences", []))
        for item in self.memory.get("evidence", []):
            rows.append({
                "id": item.get("id"), "status": "evidence", "scope": item.get("scope", "shared"),
                "confidence": min(0.70, 0.20 + 0.15 * len(item.get("sessions") or [])),
                "text": f"{item.get('polarity','')} · {item.get('state','')} · {item.get('theme','')}",
            })
        return rows

    def set_memory_status(self, memory_id: str, status: str) -> bool:
        with self.memory_lock:
            for item in self.memory.get("preferences", []):
                if item.get("id") == memory_id:
                    item["status"] = status
                    item["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    self._save_memory_and_refresh()
                    return True
        return False

    def pin_memory(self, memory_id: str) -> bool:
        if self.set_memory_status(memory_id, "pinned"):
            return True
        evidence = next((x for x in self.memory.get("evidence", []) if x.get("id") == memory_id), None)
        if evidence is None:
            return False
        text = f"the user responds positively when {evidence.get('state','')}; associated theme: {evidence.get('theme','')}."
        item = self.remember_explicitly(text, str(evidence.get("scope") or "shared"))
        item["status"] = "pinned"
        self.forget_memory(memory_id)
        self._save_memory_and_refresh()
        return True

    def forget_memory(self, memory_id: str) -> bool:
        with self.memory_lock:
            before = len(self.memory.get("preferences", [])) + len(self.memory.get("evidence", []))
            self.memory["preferences"] = [x for x in self.memory.get("preferences", []) if x.get("id") != memory_id]
            self.memory["evidence"] = [x for x in self.memory.get("evidence", []) if x.get("id") != memory_id]
            changed = before != len(self.memory["preferences"]) + len(self.memory["evidence"])
            if changed:
                self._save_memory_and_refresh()
            return changed

    def _handle_memory_command(self, text: str, timings: Timings) -> bool:
        # A conversational preface such as "I remember what I meant to check"
        # must not swallow a control request later in the same utterance.
        if (re.search(r"\b(?:reduce|lower|change|adjust|set|stop|neutral|quieter|softer|gentler|turn down|pull down)\b", text or "", re.I)
                and re.search(r"\b(?:signal|lane|channel|electrode|volume|intensity|signal lab|vector)\b", text or "", re.I)):
            return False
        remember = re.search(r"\b(?:please )?(?:remember|learn) (?:that )?(.{4,240})", text or "", re.I)
        if remember and not re.search(r"\b(?:can you|do you|what|anything)\b.{0,30}\bremember\b", text or "", re.I):
            preference = remember.group(1).strip(" .")
            self.remember_explicitly(preference)
            self._emit_local_turn(text, f"I’ll remember that: {preference}.", timings, "MEMORY")
            return True
        if re.search(r"\b(?:what do you remember|what have you learned|show (?:me )?(?:your|the) memor(?:y|ies))\b", text or "", re.I):
            learned = [x.get("text", "") for x in self.memory.get("preferences", []) if x.get("status") in ("learned", "pinned")]
            if learned:
                reply = "I currently carry these durable preferences: " + "; ".join(learned[:8]) + ". You can review or remove them on the Memory page."
            else:
                evidence_count = len(self.memory.get("evidence", []))
                reply = f"I have no durable learned preferences yet. I have {evidence_count} low-confidence evidence record{'s' if evidence_count != 1 else ''}, available on the Memory page."
            self._emit_local_turn(text, reply, timings, "MEMORY")
            return True
        return False

    def select_director_voice(self, key: str) -> Tuple[bool, str]:
        key = str(key or "").lower()
        if key not in DIRECTOR_VOICES:
            return False, f"Unknown Director voice: {key}"
        selected = DIRECTOR_VOICES[key]
        reference_text = str(selected.get("reference") or "").strip()
        reference = ROOT / reference_text
        if reference_text and not reference.exists():
            return False, f"Voice reference is missing: {reference}"
        self.invalidate_voice("Director voice change")
        self.close_output_stream()
        self.cfg["director_voice"] = key
        chatterbox = (self.cfg.get("tts_backends") or {}).get("chatterbox")
        if isinstance(chatterbox, dict) and reference_text:
            chatterbox["voice"] = reference_text
        save_config(self.cfg)
        self.tts = TTSAdapter(self.cfg, self.session)
        self.system_prompt = self.build_effective_system_prompt()
        if hasattr(self, "narrative_rng"):
            self.narrative_arc_template = self._choose_narrative_arc_template()
        log(f"DIRECTOR VOICE selected key={key!r} name={selected['name']!r} reference={selected['reference']!r}")
        return True, selected["label"]

    def set_director_personality_profile(self, enabled: bool) -> None:
        self.cfg["director_personality_profile"] = bool(enabled)
        save_config(self.cfg)
        self.system_prompt = self.build_effective_system_prompt()
        if hasattr(self, "narrative_rng"):
            self.narrative_arc_template = self._choose_narrative_arc_template()

    def _ensure_output_stream(self):
        with self.output_lock:
            if self.output_stream is None:
                self.output_stream = sd.RawOutputStream(
                    samplerate=int(self.tts.backend().sample_rate), channels=1, dtype="int16",
                    device=self.cfg.get("speaker_device"), blocksize=0
                )
                self.output_stream.start()
            return self.output_stream

    def close_output_stream(self) -> None:
        with self.output_lock:
            stream, self.output_stream = self.output_stream, None
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass

    def emit(self, kind: str, data: Any = None) -> None:
        if kind in {"user", "assistant", "state"} and hasattr(self, "session_journal"):
            self._journal_event(kind, {"content": str(data)})
        self.ui.put((kind, data))

    def _journal_event(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Append one structured, local-only event for explicit end-of-session export."""
        entry = {
            "sequence": 0,
            "time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": round(max(0.0, time.monotonic() - self.session_journal_started_monotonic), 3),
            "event": str(event),
            "data": data or {},
        }
        with self.session_journal_lock:
            entry["sequence"] = len(self.session_journal) + 1
            self.session_journal.append(entry)

    def _journal_copy(self) -> List[Dict[str, Any]]:
        with self.session_journal_lock:
            return json.loads(json.dumps(self.session_journal, ensure_ascii=False))

    def _latest_journal_evidence(self, event: str, content: str = "") -> Optional[Dict[str, Any]]:
        """Return a compact citation to an actual journal event, never elapsed time alone."""
        wanted = self._normalized_reply(content) if content else ""
        for item in reversed(self._journal_copy()):
            if item.get("event") != event:
                continue
            recorded = str((item.get("data") or {}).get("content") or "")
            if wanted and self._normalized_reply(recorded) != wanted:
                continue
            return {
                "event": event,
                "sequence": item.get("sequence"),
                "time_local": item.get("time_local"),
                "content": recorded,
            }
        return None

    def set_state(self, state: str) -> None:
        self.current_state = state
        self.emit("state", state)
        log(f"STATE {state}")

    def load_stt(self) -> None:
        self.set_state("LOADING STT")
        model_name = self.cfg["stt_model"]
        device = self.cfg.get("stt_device", "auto")
        compute = self.cfg.get("stt_compute_type", "auto")
        attempts: List[Tuple[str,str]] = []
        if device == "auto":
            # Prefer CUDA on this machine, but make CPU fallback automatic.
            attempts = [("cuda", "float16" if compute == "auto" else compute), ("cpu", "int8")]
        else:
            attempts = [(device, "int8" if compute == "auto" and device == "cpu" else ("float16" if compute == "auto" else compute))]
        last_error = None
        for dev, comp in attempts:
            try:
                self.emit("log", f"Loading faster-whisper {model_name} on {dev}/{comp}…")
                log(f"Loading STT {model_name} device={dev} compute={comp}")
                self.stt_model = WhisperModel(model_name, device=dev, compute_type=comp)
                self.emit("log", f"STT ready: {model_name} ({dev}/{comp})")
                # Voice bridges auto-start in parallel with STT. Wait for the selected backend
                # before the opening line so first speech does not race a model still loading.
                self.set_state("OPENING")
                if self.wait_for_active_tts():
                    self.maybe_speak_opening()
                else:
                    self.opening_spoken = True
                self.stt_ready.set()
                self.set_state("READY")
                return
            except Exception as e:
                last_error = e
                log(f"STT init failed {dev}/{comp}: {e}")
        self.emit("error", f"Could not load faster-whisper: {last_error}")
        self.set_state("STT ERROR")

    def _backend_port_open(self, port: int, timeout: float = 0.25) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
                return True
        except OSError:
            return False

    def autostart_voice_services(self) -> None:
        """Start installed local TTS bridges that are enabled for auto-start.

        Existing listeners are left alone. Bridges started here run independently,
        so restarting Gwendolyn does not needlessly reload a voice model.
        """
        if os.name != "nt":
            log("VOICE AUTOSTART skipped: Windows launch helper only")
            return
        auto = self.cfg.get("tts_autostart") or {}
        specs = {
            "kokoro": (8770, RUNTIME_ROOT / ".venv-kokoro" / "Scripts" / "python.exe", RUNTIME_ROOT / "bridges" / "kokoro_bridge.py"),
            "chatterbox": (8771, RUNTIME_ROOT / ".venv-chatterbox" / "Scripts" / "python.exe", RUNTIME_ROOT / "bridges" / "chatterbox_bridge.py"),
        }
        for key, (port, python_exe, bridge_py) in specs.items():
            if not bool(auto.get(key, False)):
                continue
            # Qwen gets first claim on GPU memory. This matters most for Chatterbox:
            # if Ollama unloads and later reloads after Chatterbox is resident, Qwen can
            # fall into a much slower offload path.
            if key == "chatterbox" and not self.ollama_ready.is_set():
                log("VOICE AUTOSTART Chatterbox waiting for Ollama warm-up")
                self.ollama_ready.wait(timeout=max(5.0, float(self.cfg.get("ollama_startup_timeout_sec", 120))))
            label = str(((self.cfg.get("tts_backends") or {}).get(key) or {}).get("label") or key)
            if self._backend_port_open(port):
                log(f"VOICE AUTOSTART {label}: already running on {port}")
                continue
            if not python_exe.exists() or not bridge_py.exists():
                log(f"VOICE AUTOSTART {label}: not installed; skipping")
                self.emit("log", f"{label} auto-start skipped: run its SETUP batch first.")
                continue
            try:
                creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                startupinfo = None
                if hasattr(subprocess, "STARTUPINFO"):
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
                    startupinfo.wShowWindow = 6  # SW_MINIMIZE
                subprocess.Popen(
                    [str(python_exe), str(bridge_py)],
                    cwd=str(RUNTIME_ROOT),
                    creationflags=creationflags,
                    startupinfo=startupinfo,
                )
                log(f"VOICE AUTOSTART {label}: launched")
            except Exception as e:
                log(f"VOICE AUTOSTART {label} failed: {e}")
                self.emit("log", f"Could not auto-start {label}: {e}")

    def wait_for_active_tts(self) -> bool:
        backend = self.tts.backend()
        wait_sec = max(0.0, float(self.cfg.get("tts_startup_wait_sec", 60)))
        if self.tts.healthy(timeout=0.5):
            return True
        self.emit("log", f"Waiting for {backend.label} to become ready…")
        deadline = time.monotonic() + wait_sec
        while time.monotonic() < deadline:
            if self.tts.healthy(timeout=0.7):
                self.emit("log", f"{backend.label} ready.")
                return True
            time.sleep(0.5)
        self.emit("log", f"{backend.label} is still offline after {wait_sec:.0f}s; continuing without startup speech.")
        return False

    def check_services(self) -> None:
        statuses = {"LLM_label": "llama-server" if self._llm_backend() == "openai" else "Ollama"}
        for name, url in [
            ("Vector", self.cfg["vector_base"] + "/v1/state"),
            ("Ollama", self._llm_health_url()),
        ]:
            try:
                r = self.session.get(url, timeout=1.2)
                statuses[name] = r.ok
            except Exception:
                statuses[name] = False
        try:
            backend = self.tts.backend()
            statuses["TTS"] = self.tts.healthy(timeout=1.2)
            statuses["TTS_label"] = backend.label
        except Exception:
            statuses["TTS"] = False
            statuses["TTS_label"] = "TTS"
        self.emit("services", statuses)

    def audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            log(f"Audio input status: {status}")
        with self.record_lock:
            if self.recording:
                self.audio_chunks.append(indata.copy())

    def start_recording(self) -> None:
        if self.voice_activation_monitor is not None:
            self.voice_activation_monitor.suspend_for_ptt()
        if not self.stt_ready.is_set():
            self.emit("log", "PTT ignored: STT is not ready yet.")
            if self.voice_activation_monitor is not None:
                self.voice_activation_monitor.resume_after_ptt()
            return
        self.invalidate_voice("LB barge-in")
        with self.record_lock:
            if self.recording:
                return
            self.audio_chunks = []
            self.record_cancelled = False
            try:
                self.input_stream = sd.InputStream(
                    samplerate=int(self.cfg["sample_rate"]),
                    channels=1,
                    dtype="float32",
                    device=self.cfg.get("microphone_device"),
                    callback=self.audio_callback,
                )
                self.input_stream.start()
                self.recording = True
            except Exception as e:
                self.emit("error", f"Microphone start failed: {e}")
                log(traceback.format_exc())
                if self.voice_activation_monitor is not None:
                    self.voice_activation_monitor.resume_after_ptt()
                return
        self.set_state("LISTENING")

    def cancel_recording(self) -> None:
        with self.record_lock:
            self.record_cancelled = True
        self._finish_input(discard=True)
        self.set_state("READY")

    def stop_recording_and_process(self) -> None:
        audio = self._finish_input(discard=False)
        if audio is None or len(audio) < int(self.cfg["sample_rate"] * 0.15):
            self.emit("log", "PTT: no useful audio captured.")
            self.set_state("READY")
            return
        threading.Thread(target=self._process_audio, args=(audio,), daemon=True).start()

    def _finish_input(self, discard: bool) -> Optional[np.ndarray]:
        with self.record_lock:
            if not self.recording:
                return None
            self.recording = False
            stream = self.input_stream
            self.input_stream = None
        try:
            if stream:
                stream.stop(); stream.close()
        except Exception:
            pass
        if self.voice_activation_monitor is not None:
            self.voice_activation_monitor.resume_after_ptt()
        with self.record_lock:
            chunks = self.audio_chunks
            self.audio_chunks = []
            cancelled = self.record_cancelled
        if discard or cancelled or not chunks:
            return None
        return np.concatenate(chunks, axis=0).reshape(-1)

    def _process_audio(self, audio: np.ndarray) -> None:
        timings = Timings(ptt_release=time.perf_counter())
        try:
            self.set_state("TRANSCRIBING")
            text = self.transcribe(audio)
            timings.stt_done = time.perf_counter()
            if not text:
                empty_now = time.monotonic()
                reset_after = float(self.cfg.get("adaptive_vad_reset_seconds", 20.0))
                if self.vad_last_empty_at and empty_now - self.vad_last_empty_at >= reset_after:
                    log(f"VOICE ACTIVITY adaptive VAD reset after {empty_now - self.vad_last_empty_at:.1f}s quiet gap")
                    self.empty_stt_count = 0
                    self.vad_adaptive_multiplier = 1.0
                self.vad_last_empty_at = empty_now
                self.empty_stt_count += 1
                # Several consecutive empty Whisper results mean the VAD is hearing room noise,
                # not useful speech. Raise only the software threshold, gently and temporarily.
                step = float(self.cfg.get("adaptive_vad_step", 0.08))
                cap = float(self.cfg.get("adaptive_vad_max_multiplier", 1.65))
                after = int(self.cfg.get("adaptive_vad_after_empty", 3))
                if self.empty_stt_count >= after:
                    self.vad_adaptive_multiplier = min(cap, 1.0 + step * (self.empty_stt_count - after + 1))
                debounce = float(self.cfg.get("empty_stt_debounce_seconds", 1.5))
                backoff_after = int(self.cfg.get("empty_stt_backoff_after", 5))
                if self.empty_stt_count >= backoff_after:
                    # Repeated empty transcripts are environmental noise, not turns.
                    # Back off the activation monitor without affecting LB PTT.
                    debounce = max(debounce, float(self.cfg.get("empty_stt_backoff_seconds", 8.0)))
                    log(f"VOICE ACTIVITY false-trigger backoff={debounce:.1f}s count={self.empty_stt_count}")
                self.vad_debounce_until = time.monotonic() + debounce
                log(f"STT empty suppressed count={self.empty_stt_count} debounce_until={self.vad_debounce_until:.3f} adaptive_vad={self.vad_adaptive_multiplier:.2f}x")
                self.set_state("READY")
                return
            if self.empty_stt_count or self.vad_adaptive_multiplier != 1.0:
                log(f"STT genuine speech resets adaptive VAD from empty_count={self.empty_stt_count} multiplier={self.vad_adaptive_multiplier:.2f}x")
            self.empty_stt_count = 0
            self.vad_adaptive_multiplier = 1.0
            self.vad_last_empty_at = 0.0
            self.emit("user", text)
            self.handle_text(text, timings)
        except Exception as e:
            self.emit("error", f"Voice turn failed: {e}")
            log(traceback.format_exc())
            self.set_state("READY")

    def transcribe(self, audio: np.ndarray) -> str:
        assert self.stt_model is not None
        t0 = time.perf_counter()
        try:
            segments, info = self.stt_model.transcribe(
                audio,
                language="en",
                beam_size=int(self.cfg.get("stt_beam_size", 3)),
                vad_filter=True,
                condition_on_previous_text=False,
                temperature=0.0,
                initial_prompt=stt_initial_prompt(),
            )
            text = accepted_stt_text(segments)
        except Exception as e:
            # CUDA can fail at first inference even if model construction succeeded. One-time CPU fallback.
            if getattr(self.stt_model, "model", None) is not None:
                log(f"STT inference failed, trying CPU fallback: {e}")
            self.stt_model = WhisperModel(self.cfg["stt_model"], device="cpu", compute_type="int8")
            segments, info = self.stt_model.transcribe(
                audio, language="en", beam_size=int(self.cfg.get("stt_beam_size", 3)), vad_filter=True,
                condition_on_previous_text=False, temperature=0.0,
                initial_prompt=stt_initial_prompt(),
            )
            text = accepted_stt_text(segments)
        if is_repeated_stt_hallucination(text):
            log(f"STT repeated-word hallucination suppressed: {text[:160]!r}")
            text = ""
        corrected = normalize_stt_text(text)
        log(f"STT {time.perf_counter()-t0:.3f}s: {text!r}")
        if corrected != text:
            log(f"STT local lexicon corrected: {corrected!r}")
        return corrected

    def typed_turn(self, text: str) -> None:
        text = text.strip()
        if not text: return
        self.emit("user", text)
        threading.Thread(target=self.handle_text, args=(text, Timings()), daemon=True).start()

    def handle_controller_feedback(self, direction: str) -> None:
        """Record silent D-pad feedback; only Left performs immediate bounded relief."""
        if SESSION_MODE != "signal_lab" or direction not in CONTROLLER_FEEDBACK:
            return
        text = CONTROLLER_FEEDBACK[direction]
        try:
            snapshot = self.signal_lab_get("/v1/state", timeout=1.0)
        except Exception as exc:
            snapshot = {"unavailable": f"{type(exc).__name__}: {exc}"}
        self.emit("user", f"D-pad {direction.upper()}: {text}")
        self.history.append({"role": "user", "content": text})
        self.trim_history()
        self.last_conversation_activity_at = time.monotonic()
        self._journal_event("controller_feedback", {
            "direction": direction,
            "meaning": text,
            "immediate_relief": direction == "left",
            "evaluated_state": snapshot,
        })
        log(f"CONTROLLER FEEDBACK direction={direction} immediate_relief={direction == 'left'}")
        if direction != "left":
            return
        last = (snapshot or {}).get("last_command") or {}
        left = dict(last.get("left_semantic") or {})
        right = dict(last.get("right_semantic") or {})
        if not left or not right:
            self._journal_event("controller_feedback_relief_failed", {
                "direction": direction, "reason": "no_semantic_state_available",
            })
            self.emit("assistant", "D-pad relief was recorded, but Signal Lab had no semantic state to reduce.")
            return
        preset = (snapshot or {}).get("active_preset") or {}
        live = (snapshot or {}).get("live") or {}
        vmin, vmax = float(preset.get("volume_min") or 40), float(preset.get("volume_max") or 100)
        for lane_name, lane in (("left", left), ("right", right)):
            current_volume = float((live.get(f"{lane_name}_signal") or {}).get("volume") or vmin)
            target_volume = max(vmin, current_volume * 0.90)
            lane["intensity"] = 0.0 if vmax <= vmin else (target_volume - vmin) * 100.0 / (vmax - vmin)
        self.reset_action_ledger()
        result = self.execute_tool("signal_lab_apply", {
            "left": left,
            "right": right,
            "transition_seconds": 5,
            "reason": "Immediate D-pad Left bounded relief; actual lane volume reduced by at most 10 percent only",
        })
        self.last_conducted_decision_at = time.monotonic()
        verification_args = {
                "left": left, "right": right, "transition_seconds": 5,
                "reason": "Immediate D-pad Left bounded relief; actual lane volume reduced by at most 10 percent only",
        }
        deadline = time.monotonic() + 1.0
        while result.get("ok") and time.monotonic() < deadline:
            try:
                result = dict(result, verification_state=self.signal_lab_get("/v1/state", timeout=0.3))
            except Exception:
                break
            if self._verified_control("signal_lab_apply", verification_args, result):
                break
            time.sleep(0.05)
        if result.get("ok") and self._verified_control("signal_lab_apply", verification_args, result):
            self.emit("assistant", "D-pad relief applied: each lane’s actual volume reduced by up to 10 percent.")
        else:
            self.emit("assistant", "D-pad relief was requested but could not be verified; use Signal Lab’s manual control.")

    def vector_get(self, path: str, timeout=1.5) -> Dict[str, Any]:
        r = self.session.get(self.cfg["vector_base"] + path, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def vector_post(self, path: str, payload: Dict[str, Any], timeout=3.0) -> Dict[str, Any]:
        r = self.session.post(self.cfg["vector_base"] + path, json=payload, timeout=timeout)
        r.raise_for_status()
        try: return r.json()
        except Exception: return {"ok": True, "status": r.status_code}

    def signal_lab_get(self, path: str, timeout=1.5) -> Dict[str, Any]:
        base = str(self.cfg.get("signal_lab_base") or "http://127.0.0.1:18766").rstrip("/")
        r = self.session.get(base + path, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if path.split("?", 1)[0] == "/v1/state" and isinstance(data, dict):
            # Co-locate read-only response evidence with authoritative audible
            # signal state so every Signal Lab reasoning path sees the same truth.
            data = dict(data)
            data["restim_sensor_response"] = self.restim_sensor_snapshot()
        return data

    def signal_lab_post(self, path: str, payload: Dict[str, Any], timeout=3.0) -> Dict[str, Any]:
        base = str(self.cfg.get("signal_lab_base") or "http://127.0.0.1:18766").rstrip("/")
        r = self.session.post(base + path, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def reset_action_ledger(self) -> None:
        self.last_action_ledger = {"requested": [], "executed": [], "failed": []}

    def compact_action_ledger(self) -> str:
        # Narration-safe ledger: preserve what was requested/executed/failed, but
        # never expose raw Vector response internals (weights, strength values,
        # alpha windows, HTTP payloads) to the language model's spoken follow-up.
        safe = {"requested": [], "executed": [], "failed": []}
        for item in self.last_action_ledger.get("requested", []):
            safe["requested"].append({"name": item.get("name"), "args": item.get("args") or {}})
        for item in self.last_action_ledger.get("executed", []):
            safe["executed"].append({"name": item.get("name"), "args": item.get("args") or {}})
        for item in self.last_action_ledger.get("failed", []):
            safe["failed"].append({
                "name": item.get("name"),
                "args": item.get("args") or {},
                "error": item.get("error") or "Action failed",
            })
        return json.dumps(safe, ensure_ascii=False, separators=(",", ":"))

    def narration_tool_result(self, name: str, args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Return only the tool facts needed for natural spoken confirmation."""
        if not isinstance(result, dict) or not result.get("ok", False):
            return {"ok": False, "action": name, "error": (result or {}).get("error", "Action failed")}
        if name.startswith("signal_lab_"):
            return {
                "ok": True,
                "action": name,
                "args": dict(args),
                "offline_simulation": True,
                "physical_output_enabled": False,
                "live": result.get("live"),
                "status": result.get("status") or result.get("mode"),
                "state": result.get("command") or result.get("last_command"),
            }
        return {"ok": True, "action": name, "args": dict(args)}

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if SESSION_MODE == "signal_lab" and name.startswith("vector_") and name != "vector_stop":
            raise ValueError("Vector controls are unavailable in this Signal Lab session")
        if SESSION_MODE == "vector" and name.startswith("signal_lab_"):
            raise ValueError("Signal Lab controls are unavailable in this Vector session")
        self.last_action_ledger["requested"].append({"name": name, "args": dict(args)})
        log(f"TOOL {name} {args}")
        if name.startswith("signal_lab_"):
            try:
                if name == "signal_lab_get_state":
                    result = self.signal_lab_get("/v1/state")
                elif name == "signal_lab_apply":
                    left_args = args.get("left")
                    right_args = args.get("right")
                    transition_args = args.get("transition_seconds")
                    if not signal_lab_apply_args_complete(args):
                        raise ValueError("Incomplete Signal Lab command; both lanes and transition are required")
                    result = self.signal_lab_post("/v1/director/command", {
                        "left": left_args,
                        "right": right_args,
                        "transition_seconds": transition_args,
                        "reason": args.get("reason") or "Gwendolyn offline Director command",
                    })
                elif name == "signal_lab_neutral":
                    result = self.signal_lab_post("/v1/director/neutral", {
                        "reason": args.get("reason") or "Gwendolyn offline neutral",
                    })
                elif name == "signal_lab_heart_tempo":
                    mode = str(args.get("mode") or "").lower()
                    if mode not in {"escalation", "relief", "off"}:
                        raise ValueError("Heart-tempo mode must be escalation, relief or off")
                    payload = {"mode": mode, "reason": args.get("reason") or "Gwendolyn heart-tempo intent"}
                    if mode != "off":
                        target = str(args.get("target") or "").lower()
                        if target not in {"left", "right"}:
                            raise ValueError("Heart-tempo target must be left or right")
                        payload["target"] = target
                    result = self.signal_lab_post("/v1/director/heart-tempo", payload)
                else:
                    raise ValueError(f"Unknown Signal Lab tool {name}")
                if not isinstance(result, dict) or not result.get("ok", False):
                    raise RuntimeError(str((result or {}).get("error") or "Signal Lab rejected the command"))
                self.last_action_ledger["executed"].append({"name": name, "args": dict(args), "result": result})
                self.signal_lab_context_until = time.monotonic() + 300.0
                self.emit("tool", {"name": name, "args": args, "result": result})
                log(f"SIGNAL LAB offline action applied {name} audio_enabled={result.get('audio_enabled', False)}")
                return result
            except Exception as e:
                result = {"ok": False, "error": str(e), "audio_enabled": False}
                self.last_action_ledger["failed"].append({"name": name, "args": dict(args), "error": str(e)})
                self.emit("tool", {"name": name, "args": args, "result": result})
                log(f"SIGNAL LAB action rejected {name}: {e}")
                return result
        # Do not spend a narrative beat or create a fake intervention by applying
        # a semantic value Vector already reports as current.
        if name != "vector_get_state":
            try:
                compact_fresh = self._refresh_authoritative_state(timeout=1.0)
                if self._action_is_noop(name, args, compact_fresh):
                    result = {"ok": True, "noop": True, "already": True}
                    self.last_action_ledger["executed"].append({"name": name, "args": dict(args), "result": result})
                    log(f"STATE ARC no-op suppressed {name} {args}")
                    self.emit("tool", {"name": name, "args": args, "result": result})
                    return result
            except Exception:
                pass
        mapping = {
            "vector_get_state": ("GET", "/v1/state", {}),
            "vector_set_preset": ("POST", "/v1/preset", {"preset": args.get("preset")}),
            "vector_set_rolling_variety": ("POST", "/v1/rolling-variety", {"enabled": bool(args.get("enabled"))}),
            "vector_neutral": ("POST", "/v1/neutral", {}),
            "vector_stop": ("POST", "/v1/stop", {}),
            "vector_resume": ("POST", "/v1/resume", {}),
            "vector_generated_motion_plan": ("POST", "/v1/generated-motion/plan", dict(args)),
            "vector_generated_motion_hold": ("POST", "/v1/generated-motion/hold", {}),
            "vector_generated_motion_resume": ("POST", "/v1/generated-motion/resume", {}),
            "vector_use_authored_tcode": ("POST", "/v1/generated-motion/authored", {}),
            "vector_set_texture": ("POST", "/v1/semantic/texture", {"texture": args.get("texture")}),
            "vector_set_primary_spatial": ("POST", "/v1/semantic/primary-spatial", {"primary_spatial": args.get("primary_spatial")}),
            "vector_set_variation": ("POST", "/v1/semantic/variation", {"variation": args.get("variation")}),
            "vector_set_top_focus": ("POST", "/v1/semantic/top-focus", {"top_focus": args.get("top_focus")}),
            "vector_set_bottom_focus": ("POST", "/v1/semantic/bottom-focus", {"bottom_focus": args.get("bottom_focus")}),
            "vector_top_spatial_gain": ("POST", "/v1/spatial-gain/top", {"action": args.get("action")}),
            "vector_bottom_spatial_gain": ("POST", "/v1/spatial-gain/bottom", {"action": args.get("action")}),
            "vector_adjust_stroke_range": ("POST", "/v1/modifier/stroke-range", {"action": args.get("action")}),
            "vector_set_targeting": ("POST", "/v1/modifier/target", {"preset": args.get("preset")}),
            "vector_tempo_window": ("POST", "/v1/modifier/tempo", {"scale": args.get("scale"), "duration_seconds": args.get("duration_seconds")}),
            "vector_restore_modifiers": ("POST", "/v1/modifier/restore", {}),
        }
        if name not in mapping:
            return {"ok": False, "error": f"Unknown tool {name}"}
        method, path, payload = mapping[name]
        try:
            result = self.vector_get(path) if method == "GET" else self.vector_post(path, payload)
            # Add authoritative post-state for physical changes. Semantic profile changes
            # use Vector-owned ~0.2 s transitions, so wait briefly for /v1/state to
            # report the requested dimension instead of comparing against stale state.
            if name != "vector_get_state":
                try:
                    key, desired = self._action_state_key_value(name, args)
                    if key and desired is not None:
                        compact_state = self._wait_for_semantic_state(key, desired, timeout=0.85)
                    else:
                        compact_state = self._refresh_authoritative_state(timeout=1.2)
                    engine_state = str(compact_state.get("engine_state") or "").strip().lower()
                    if name == "vector_stop" and engine_state != "stopped":
                        raise RuntimeError(f"Vector stop acknowledged but engine_state is {compact_state.get('engine_state')!r}")
                    if name == "vector_resume" and engine_state == "stopped":
                        raise RuntimeError("Vector resume acknowledged but engine_state is still 'Stopped'")
                    result = {"ok": True, "accepted": result, "state": compact_state}
                except Exception:
                    if name in ("vector_stop", "vector_resume"):
                        raise
                    result = {"ok": True, "accepted": result}
            self.last_action_ledger["executed"].append(
                {"name": name, "args": dict(args), "result": result}
            )
            if name != "vector_get_state":
                self.last_meaningful_vector_change_at = time.monotonic()
                self.proactive_snooze_until = 0.0
                if name == "vector_generated_motion_plan":
                    self.vector_motion_plan = dict(args)
                    self.heart_tempo_last_stroke_ms = float(args.get("stroke_duration_ms") or 0) or None
                self._record_intervention_action(name, args)
            self.emit("tool", {"name": name, "args": args, "result": result})
            return result
        except Exception as e:
            result = {"ok": False, "error": str(e)}
            self.last_action_ledger["failed"].append(
                {"name": name, "args": dict(args), "error": str(e)}
            )
            self.emit("tool", {"name": name, "args": args, "result": result})
            return result

    def compact_state(self, s: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "engine_state": s.get("engine_state") or s.get("engine"),
            "preset": s.get("preset"),
            "rolling_variety": s.get("rolling_variety"),
            "signal": s.get("signal"),
            "semantic": s.get("semantic"),
            "focus_history": s.get("focus_history"),
            "spatial_gain": s.get("spatial_gain"),
            "modifier": s.get("modifier"),
            "future": s.get("future"),
            "timeline": s.get("timeline"),
            "routing": s.get("routing") or s.get("routing_mode"),
        }

    def _log_timeline(self, timeline: Any) -> None:
        if not isinstance(timeline, dict):
            return
        if not timeline.get("loaded"):
            return
        source = timeline.get("clock_source")
        synced = bool(timeline.get("synced"))
        auth = bool(timeline.get("clock_authoritative"))
        conf = timeline.get("sync_confidence")
        pos = timeline.get("position_seconds")
        now = timeline.get("now") or {}
        near = timeline.get("next_10_seconds") or {}
        ahead = timeline.get("next_30_seconds") or {}
        state = "authoritative" if auth and synced else ("locked" if synced else "searching")
        def band(x):
            return x.get("energy_band") if isinstance(x, dict) else None
        log(
            "TIMELINE "
            f"source={source!r} state={state} pos={pos} confidence={conf} "
            f"now={band(now)!r} next10={band(near)!r} "
            f"next30={band(ahead)!r} trend={ahead.get('energy_trend') if isinstance(ahead, dict) else None!r} "
            f"focus_now={now.get('focus_region') if isinstance(now, dict) else None!r} "
            f"media_age={timeline.get('media_sample_age_seconds')!r}"
        )

    def live_context(self) -> str:
        try:
            s = self.vector_get("/v1/state", timeout=1.2)
            compact = self.compact_state(s)
            # Drop null top-level entries to keep prompt small.
            compact = {k:v for k,v in compact.items() if v is not None}
            self.last_live_state = compact
            self._update_authoritative_state(compact)
            self._log_timeline(compact.get("timeline"))
            # Put Director grounding first so it cannot be lost if an unusually
            # detailed future/timeline payload reaches the prompt-size ceiling.
            prompt_state = {
                "director_controls": self.director_control_context(compact),
                "heart_tempo": self.vector_heart_tempo_snapshot(),
                "restim_sensor_response": self.vector_sensor_snapshot(),
                **compact,
            }
            return json.dumps(prompt_state, ensure_ascii=False, separators=(",", ":"))[:8000]
        except Exception as e:
            self.last_live_state = {}
            return json.dumps({"vector_state_unavailable": str(e)})

    def director_control_context(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Expose current settings and only meaningful semantic alternatives.

        This is model-facing and UI-facing.  It prevents Qwen from having to
        rediscover no-ops after generation and gives the operator the same view.
        """
        state = state if isinstance(state, dict) else self.last_live_state
        current = dict(self.current_vector_semantic)
        current.update(self._semantic_snapshot_from_state(state))
        choices = {
            "top_focus": ["Glans Focus", "Shaft Focus", "Lower Shaft Focus", "Root Focus", "Top Sweep", "Top Full"],
            "bottom_focus": ["Prostate Focus", "Anal Focus", "Perineum Focus", "Bottom Sweep", "Bottom Full"],
            "texture": ["Smoothest", "Smooth", "Normal", "Rough", "Roughest"],
            "variation": ["Still", "Subtle", "Normal", "Lively", "Wild"],
        }
        available: Dict[str, List[str]] = {}
        for key, values in choices.items():
            active = str(current.get(key) or "").strip().lower()
            available[key] = [value for value in values if value.lower() != active]
        engine = str(state.get("engine_state") or "unknown")
        return {
            "hardware": {
                "top": self.profile.get("top_electrode_name", "Top electrode"),
                "bottom": self.profile.get("bottom_electrode_name", "Bottom electrode"),
            },
            "engine_state": engine,
            "preset": state.get("preset"),
            "spatial_gain": state.get("spatial_gain"),
            "modifier": state.get("modifier"),
            "active": current,
            "meaningful_alternatives": available,
            "instruction": "Never propose an active value; choose only from meaningful_alternatives.",
        }

    def control_snapshot(self) -> Dict[str, Any]:
        try:
            compact = self._refresh_authoritative_state(timeout=1.0)
            return {**self.director_control_context(compact),
                    "heart_tempo": self.vector_heart_tempo_snapshot(),
                    "restim_sensor_response": self.vector_sensor_snapshot()}
        except Exception as e:
            return {"error": str(e), "active": {}, "meaningful_alternatives": {}}

    def _choose_intervention_target(self) -> int:
        mode = str(self.profile.get("intervention_sequence", "2–4 changes") or "2–4 changes")
        if "2–3" in mode or "2-3" in mode:
            return self.narrative_rng.randint(2, 3)
        if "3–4" in mode or "3-4" in mode:
            return self.narrative_rng.randint(3, 4)
        return self.narrative_rng.randint(2, 4)

    def _baseline_values(self) -> Dict[str, str]:
        return {
            "texture": str(self.profile.get("baseline_texture", "Normal") or "Normal"),
            "variation": str(self.profile.get("baseline_variation", "Normal") or "Normal"),
            "top_focus": str(self.profile.get("baseline_top_focus", "Top Full") or "Top Full"),
            "bottom_focus": str(self.profile.get("baseline_bottom_focus", "Bottom Full") or "Bottom Full"),
        }

    def _baseline_summary(self) -> str:
        b = self._baseline_values()
        return f"texture={b['texture']}, variation={b['variation']}, top={b['top_focus']}, bottom={b['bottom_focus']}"

    @staticmethod
    def _find_state_value(obj: Any, wanted: str) -> Any:
        """Best-effort recursive lookup retained for non-authoritative display/context use."""
        aliases = {
            "top_focus": {"top_focus", "topfocus", "top"},
            "bottom_focus": {"bottom_focus", "bottomfocus", "bottom"},
            "texture": {"texture"},
            "variation": {"variation"},
        }.get(wanted, {wanted})
        if isinstance(obj, dict):
            for k, v in obj.items():
                nk = re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")
                if nk in aliases and isinstance(v, (str, int, float, bool)):
                    return v
            for v in obj.values():
                found = GwendolynCore._find_state_value(v, wanted)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            for v in obj:
                found = GwendolynCore._find_state_value(v, wanted)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _semantic_dimension_value(obj: Any, aliases: set[str]) -> Any:
        """Extract one semantic value from Vector's semantic/focus blocks only.

        v0.31 searched the entire compact state recursively. That allowed an
        unrelated value (notably another Normal profile) to masquerade as the
        requested semantic dimension. v0.32 deliberately limits authoritative
        lookup to Vector's semantic and focus-history blocks.
        """
        if not isinstance(obj, dict):
            return None
        # Direct semantic key wins.
        for k, v in obj.items():
            nk = re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")
            if nk in aliases:
                if isinstance(v, (str, int, float, bool)):
                    return v
                if isinstance(v, dict):
                    for inner in ("selected", "current", "label", "name", "value", "focus"):
                        iv = v.get(inner)
                        if isinstance(iv, (str, int, float, bool)):
                            return iv
        return None

    def _semantic_snapshot_from_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return authoritative semantic dimensions from a live /v1/state snapshot."""
        if not isinstance(state, dict):
            return {}
        semantic = state.get("semantic") if isinstance(state.get("semantic"), dict) else {}
        out: Dict[str, Any] = {}
        dims = {
            "texture": {"texture"},
            "variation": {"variation"},
            "top_focus": {"top_focus", "topfocus"},
            "bottom_focus": {"bottom_focus", "bottomfocus"},
        }
        for key, aliases in dims.items():
            v = self._semantic_dimension_value(semantic, aliases)
            if v is not None:
                out[key] = v

        # Focus history is an explicit Vector authority for selected anatomical focus.
        hist = state.get("focus_history")
        if isinstance(hist, dict):
            for key, lane_name in (("top_focus", "top"), ("bottom_focus", "bottom")):
                lane = hist.get(lane_name)
                if isinstance(lane, dict):
                    for field in ("selected", "selected_focus", "current", "current_focus", "focus", "label"):
                        v = lane.get(field)
                        if isinstance(v, (str, int, float, bool)):
                            out[key] = v
                            break
        return out

    def _update_authoritative_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        snap = self._semantic_snapshot_from_state(state)
        if snap:
            self.current_vector_semantic.update(snap)
        return snap

    def _refresh_authoritative_state(self, timeout: float = 1.0) -> Dict[str, Any]:
        raw = self.vector_get("/v1/state", timeout=timeout)
        compact = self.compact_state(raw)
        self.last_live_state = compact
        self._update_authoritative_state(compact)
        return compact

    def _wait_for_semantic_state(self, key: str, desired: Any, timeout: float = 0.85) -> Dict[str, Any]:
        """Poll Vector through its short smooth transition and return the freshest state."""
        deadline = time.monotonic() + max(0.05, timeout)
        latest: Dict[str, Any] = {}
        while True:
            try:
                latest = self._refresh_authoritative_state(timeout=min(0.35, max(0.1, deadline - time.monotonic())))
                current = self.current_vector_semantic.get(key)
                if current is not None and str(current).strip().lower() == str(desired).strip().lower():
                    return latest
            except Exception:
                pass
            if time.monotonic() >= deadline:
                return latest
            time.sleep(0.07)

    def _action_state_key_value(self, name: str, args: Dict[str, Any]) -> Tuple[Optional[str], Any]:
        mapping = {
            "vector_set_texture": ("texture", args.get("texture")),
            "vector_set_variation": ("variation", args.get("variation")),
            "vector_set_top_focus": ("top_focus", args.get("top_focus")),
            "vector_set_bottom_focus": ("bottom_focus", args.get("bottom_focus")),
        }
        return mapping.get(name, (None, None))

    def _action_is_noop(self, name: str, args: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> bool:
        key, desired = self._action_state_key_value(name, args)
        if not key or desired is None:
            return False
        if isinstance(state, dict):
            snap = self._semantic_snapshot_from_state(state)
            current = snap.get(key)
        else:
            current = self.current_vector_semantic.get(key)
            if current is None:
                current = self._semantic_snapshot_from_state(self.last_live_state).get(key)
        if current is None:
            return False
        return str(current).strip().lower() == str(desired).strip().lower()

    def _action_is_baseline_move(self, name: str, args: Dict[str, Any]) -> bool:
        key, desired = self._action_state_key_value(name, args)
        if key:
            return str(desired).strip().lower() == self._baseline_values()[key].strip().lower()
        if name in ("vector_restore_modifiers",):
            return True
        if name in ("vector_top_spatial_gain", "vector_bottom_spatial_gain"):
            return str(args.get("action") or "").lower() == "restore"
        return False

    def _record_intervention_action(self, name: str, args: Dict[str, Any]) -> None:
        if name in ("vector_get_state", "vector_stop", "vector_resume", "vector_neutral"):
            return
        label = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
        self.intervention_recent_actions.append(label)
        self.intervention_recent_actions = self.intervention_recent_actions[-8:]
        if self._action_is_baseline_move(name, args) and self.intervention_count >= self.intervention_target:
            log(f"STATE ARC recovery sequence={self.intervention_sequence_number} after={self.intervention_count} changes via {name} {args}")
            self.intervention_count = 0
            self.intervention_sequence_number += 1
            self.intervention_target = self._choose_intervention_target()
            log(f"STATE ARC new sequence={self.intervention_sequence_number} target={self.intervention_target}")
        elif not self._action_is_baseline_move(name, args):
            self.intervention_count += 1
            self.last_excursion_change_at = time.monotonic()
            log(f"STATE ARC excursion sequence={self.intervention_sequence_number} count={self.intervention_count}/{self.intervention_target} action={name} {args}")

    def _deterministic_recovery_action(self) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Choose one real baseline recovery move from authoritative live state.

        Recovery is intentionally deterministic/mechanical once an excursion target is
        reached: Qwen may narrate the contrast, but it does not invent whether a
        recovery happened.  Prefer the most recently-touched semantic lane that is
        still away from baseline, then fall back to other baseline dimensions.
        """
        if self.intervention_count < self.intervention_target:
            return None
        now = time.monotonic()
        minimum_dwell = max(0.0, float(self.cfg.get("state_arc_minimum_dwell_seconds", 90.0)))
        recovery_allowed_at = max(self.last_excursion_change_at + minimum_dwell, self.recovery_snooze_until)
        if now < recovery_allowed_at:
            log(f"STATE ARC recovery deferred remaining={recovery_allowed_at - now:.1f}s")
            return None
        try:
            self._refresh_authoritative_state(timeout=1.0)
        except Exception:
            pass
        baseline = self._baseline_values()
        name_for_key = {
            "texture": "vector_set_texture",
            "variation": "vector_set_variation",
            "top_focus": "vector_set_top_focus",
            "bottom_focus": "vector_set_bottom_focus",
        }
        arg_for_key = {
            "texture": "texture",
            "variation": "variation",
            "top_focus": "top_focus",
            "bottom_focus": "bottom_focus",
        }
        # Recover a recently-used lane first so excursions feel like phrases rather
        # than a global reset.  This also lets Top and Bottom recover independently.
        recent_keys = []
        for label in reversed(self.intervention_recent_actions):
            name = label.split(":", 1)[0]
            for key, tool_name in name_for_key.items():
                if name == tool_name and key not in recent_keys:
                    recent_keys.append(key)
        order = recent_keys + [k for k in ("top_focus", "bottom_focus", "texture", "variation") if k not in recent_keys]
        for key in order:
            current = self.current_vector_semantic.get(key)
            desired = baseline.get(key)
            if current is None or desired is None:
                continue
            if str(current).strip().lower() != str(desired).strip().lower():
                return name_for_key[key], {arg_for_key[key]: desired}
        # If semantic lanes are already at baseline but the excursion was created by
        # modifiers/gains, use a bounded restore from the recent action history.
        recent_names = [x.split(":", 1)[0] for x in reversed(self.intervention_recent_actions)]
        if any(n in ("vector_adjust_stroke_range", "vector_set_targeting", "vector_tempo_window") for n in recent_names):
            return "vector_restore_modifiers", {}
        if "vector_top_spatial_gain" in recent_names:
            return "vector_top_spatial_gain", {"action": "restore"}
        if "vector_bottom_spatial_gain" in recent_names:
            return "vector_bottom_spatial_gain", {"action": "restore"}
        return None

    def _execute_due_recovery(self) -> Optional[str]:
        """Execute one due recovery and return grounded narration, if any."""
        action = self._deterministic_recovery_action()
        if not action:
            return None
        name, args = action
        log(f"STATE ARC deterministic recovery due sequence={self.intervention_sequence_number} action={name} {args}")
        result = self.execute_tool(name, args)
        if not isinstance(result, dict) or not result.get("ok", False):
            log(f"STATE ARC deterministic recovery failed sequence={self.intervention_sequence_number} action={name} {args}")
            return "I tried to bring one part of Vector back toward baseline, but that recovery did not take. I’ve left the rest alone."
        if result.get("noop"):
            # This should be rare because selection uses authoritative state. Do not
            # falsely advance the sequence; another turn can select a different lane.
            log(f"STATE ARC deterministic recovery resolved as no-op action={name} {args}")
            return None
        return self._grounded_confirmation()

    def _state_arc_context(self) -> str:
        baseline = self._baseline_values()
        due = self.intervention_count >= self.intervention_target
        recent = "; ".join(self.intervention_recent_actions[-4:]) or "none"
        current = {
            k: self.current_vector_semantic.get(k)
            for k in ("texture", "variation", "top_focus", "bottom_focus")
        }
        if due:
            directive = (
                "A recovery/contrast move is now due. If you propose a Vector change, prefer returning ONE out-of-baseline component toward baseline rather than stacking another excursion. "
                "Top and bottom do not need to restore simultaneously. After a recovery move, begin a fresh variation sequence."
            )
        else:
            directive = (
                "The current excursion still has room for variation. Prefer a meaningful change that differs from current state and recent actions; do not repeat a setting already active."
            )
        return (
            "VECTOR STATE ARC (planning only; execution still requires normal grounding/approval):\n"
            f"sequence={self.intervention_sequence_number}; excursion_changes={self.intervention_count}/{self.intervention_target}; baseline_due={str(due).lower()}; "
            f"baseline={baseline}; current={current}; recent_actions={recent}.\n{directive}\n"
            "Never propose setting a semantic control to the value it already has. Baseline means Normal texture, Normal variation, the configured Top focus and configured Bottom focus; partial lane restores are encouraged for contrast."
        )

    def narrative_enabled(self) -> bool:
        value = str(self.profile.get("narrative_director", "Dynamic") or "Off").strip().lower()
        return value not in ("off", "false", "0", "disabled")

    def narrative_randomness(self) -> float:
        value = str(self.profile.get("narrative_randomness", "Medium") or "Medium").strip().lower()
        return {"low": 0.25, "medium": 0.55, "high": 0.85}.get(value, 0.55)

    def _narrative_arc_stage(self) -> str:
        timeline = self.last_live_state.get("timeline") or {}
        if not isinstance(timeline, dict) or not timeline.get("loaded"):
            # Conversation still has continuity even without a script clock.
            turns = max(0, self.narrative_turn_counter)
            if turns < 4: return "opening"
            if turns < 10: return "development"
            if turns < 18: return "escalation"
            return "late-session"
        try:
            pos = float(timeline.get("position_seconds") or 0.0)
            dur = float(timeline.get("duration_seconds") or timeline.get("media_duration_seconds") or 0.0)
            if dur <= 0:
                raise ValueError
            f = min(1.0, max(0.0, pos / dur))
            if f < 0.16: return "opening"
            if f < 0.45: return "development"
            if f < 0.75: return "escalation"
            if f < 0.92: return "culmination"
            return "denouement"
        except Exception:
            return "development"

    def _choose_narrative_arc_template(self) -> str:
        requested = str(self.effective_director_profile().get("narrative_arc", "Automatic") or "Automatic").strip()
        options = ["Slow burn", "Deceptive calm", "Challenge / relief", "Escalating control", "Contrast / restraint"]
        if requested and requested.lower() != "automatic" and requested in options:
            return requested
        # Automatic is chosen once per launch/session, so replaying the same funscript can
        # naturally take a different dramatic route without altering Vector math.
        return self.narrative_rng.choice(options)

    def _session_motif_candidates(self) -> List[str]:
        """Extract concrete operator-authored themes, not random adjectives or instructions."""
        brief = str(self.profile.get("session_brief") or "").strip()
        if not brief:
            return []
        low = brief.lower()
        # Strong generic cue phrases often used to introduce examples/themes.  Capture the
        # noun-like items that follow them, but keep the public build content-agnostic.
        anchors = []
        for m in re.finditer(r"(?:references? to|likes?|themes?|motifs?|include|including|such as|for example)\s+([^.;]+)", brief, re.I):
            anchors.append(m.group(1))
        chunks = anchors or re.split(r"[.;]", brief)
        stop = {
            "about","after","again","areas","could","feel","feels","feeling","feelings","should","would","there","their","these","those","thing","things","session","brief","user","gwendolyn","reference","references","opportunity","arises","inventive","dramatic","expressive","suggesting","increase","intensity","challenge","challenging","likes","liked","other","randomness","journey","progresses","particularly","assertive","express","describing","description","mood","produce","produced","light","please","feel free","if the opportunity","etc","refrence","reference"
        }
        out=[]
        def add(x: str):
            x=re.sub(r"\s+"," ",x).strip(" ,:-")
            x=re.sub(r"^(?:and|or|a|an|the)\s+","",x,flags=re.I)
            if not x or len(x) > 48:
                return
            xl=x.lower()
            if xl in stop or any(bad in xl for bad in ("feel free","if the opportunity","which explains","areas and the","should be","you are in the mood")):
                return
            if len(x.split()) > 5:
                return
            if xl not in {y.lower() for y in out}:
                out.append(x)
        # Prefer list items / short noun phrases over isolated words.
        for chunk in chunks:
            for item in re.split(r",|\band\b|\bor\b", chunk, flags=re.I):
                item=re.sub(r"^\s*if\s+the\s+opportunity\s+arises\s+(?:reference|refrence)?\s*", "", item, flags=re.I)
                item=re.sub(r"^\s*(?:reference|refrence|mention|mentions)\s+", "", item, flags=re.I)
                item=re.sub(r"\betc\.?\b", "", item, flags=re.I)
                item=re.sub(r"^\s*be\s+inventive\s*$", "", item, flags=re.I)
                item=re.sub(r"\b(?:to|be|very|more|less|some|any|the|a|an)\b", " ", item, flags=re.I)
                item=re.sub(r"\s+", " ", item).strip()
                content=[w for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", item) if w.lower() not in stop]
                if 1 <= len(content) <= 5:
                    add(item)
        return out[:12]

    def _select_narrative_beat(self, proactive: bool = False) -> str:
        if not self.narrative_enabled():
            return "natural"
        stage = self._narrative_arc_stage()
        timeline = self.last_live_state.get("timeline") or {}
        now = (timeline.get("now") or {}) if isinstance(timeline, dict) else {}
        energy = str(now.get("energy_band") or "").lower()
        # Weighted, state-aware variation.  No beat itself authorizes a Vector change.
        effective_profile = self.effective_director_profile()
        proposal_tune = tuning_value(effective_profile, "proposal_frequency", 3)
        pressure_tune = tuning_value(effective_profile, "director_pressure", 4)
        weights = {
            "observe": 1.4,
            "tease": 1.2,
            "callback": 0.8,
            "foreshadow": 0.9,
            "challenge": 0.8 + (pressure_tune - 3) * 0.18,
            "relief": 0.5,
            "propose": (0.7 if proactive else 0.35) * {1:0.25,2:0.55,3:1.0,4:1.55,5:2.1}[proposal_tune],
            "quiet": (0.55 if proposal_tune <= 2 else 0.25) if proactive else 0.0,
        }
        if bool(self.cfg.get("director_personality_profile", True)):
            for key, adjustment in (self.director_voice_profile().get("beat_bias") or {}).items():
                if key in weights:
                    weights[key] += float(adjustment)
        arc = getattr(self, "narrative_arc_template", "Slow burn")
        if arc == "Slow burn":
            weights["observe"] += 0.35; weights["foreshadow"] += 0.35; weights["challenge"] -= 0.15
        elif arc == "Deceptive calm":
            weights["relief"] += 0.25; weights["tease"] += 0.35; weights["challenge"] += 0.15
        elif arc == "Challenge / relief":
            weights["challenge"] += 0.45; weights["relief"] += 0.45
        elif arc == "Escalating control":
            weights["challenge"] += 0.55; weights["propose"] += 0.25
        elif arc == "Contrast / restraint":
            weights["callback"] += 0.30; weights["relief"] += 0.35; weights["foreshadow"] += 0.25
        if stage in ("opening", "development"):
            weights["observe"] += 0.5; weights["tease"] += 0.3
        if stage in ("escalation", "culmination"):
            weights["challenge"] += 0.8; weights["foreshadow"] += 0.4
        if stage == "denouement":
            weights["relief"] += 1.0; weights["callback"] += 0.4
        if energy in ("challenging", "testing"):
            weights["challenge"] += 0.45; weights["relief"] += 0.35
        elif energy == "relaxing":
            # A creative beat must not turn an authoritative lull into an
            # invented rise or challenge.
            weights["challenge"] = 0.0
            weights["relief"] += 1.0; weights["observe"] += 0.6; weights["tease"] += 0.25
        elif energy == "moderate":
            weights["tease"] += 0.35; weights["propose"] += 0.25
        # Strongly discourage repeating the same beat.
        if self.narrative_recent_beats:
            weights[self.narrative_recent_beats[-1]] = weights.get(self.narrative_recent_beats[-1], 0.0) * 0.12
        if len(self.narrative_recent_beats) >= 2 and self.narrative_recent_beats[-1] == self.narrative_recent_beats[-2]:
            weights[self.narrative_recent_beats[-1]] = 0.0
        # Randomness blends the weighted draw with a stable default.
        randomness = self.narrative_randomness()
        if self.narrative_rng.random() > randomness:
            # Stable mode still rotates through the least-recently used viable
            # beat; it must not collapse to observe/propose every interval.
            viable = [key for key, value in weights.items() if value > 0]
            recent_rank = {name: index for index, name in enumerate(self.narrative_recent_beats)}
            beat = min(viable, key=lambda name: (recent_rank.get(name, -1), -weights[name]))
        else:
            items=[(k,max(0.0,v)) for k,v in weights.items() if v > 0]
            total=sum(v for _,v in items) or 1.0
            r=self.narrative_rng.random()*total
            beat=items[-1][0]
            for k,v in items:
                r-=v
                if r <= 0:
                    beat=k; break
        return beat

    def _build_session_brief_milestones(self) -> List[Dict[str, Any]]:
        """Extract explicit, trackable sections from a prose session brief."""
        brief = str(self.profile.get("session_brief") or "").strip()
        configured = self.profile.get("session_brief_schedule")
        if isinstance(configured, list):
            result = []
            for item in configured:
                if not isinstance(item, dict) or not valid_time(item.get("due_seconds")):
                    log("SESSION BRIEF ignored malformed schedule entry")
                    continue
                action = item.get("action")
                if action is not None and (not isinstance(action, dict) or
                        not isinstance(action.get("name"), str) or not isinstance(action.get("args"), dict)):
                    log("SESSION BRIEF ignored malformed action")
                    continue
                control = bool(action is not None or item.get("control") or re.search(
                    r"\b(?:signal|carrier|volume|intensity|tempo|texture|neutral|restore|lower|raise)\b",
                    str(item.get("detail") or ""), re.I))
                result.append(dict(item, label=str(item.get("label") or "Scheduled beat"),
                                   detail=str(item.get("detail") or ""), control=control))
            return sorted(result, key=lambda m: m["due_seconds"])
        if not brief:
            return []
        legacy_heading = re.compile(
            r"(?:recall beat|name marker|dress code|pleasure-attire association|"
            r"user-defined sensation protocol|tension-restraint analogy|the ending)", re.I)
        flexible_heading = re.compile(
            r"^(?:\d+[.)]\s*)?(?:introduction|personalisation|personalization|"
            r"session body(?:\s*[123])?|session closure)\b", re.I)
        sections: List[List[str]] = []
        current: List[str] = []
        for raw_line in brief.splitlines():
            line = re.sub(r"^#{1,6}\s*", "", raw_line.strip())
            is_heading = bool(line and (legacy_heading.search(line) or flexible_heading.search(line)
                                       or re.match(r"^\d+[.)]\s+", line)
                                       or re.match(r"^\[\d+:\d{2}\]", line)))
            if is_heading:
                if current:
                    sections.append(current)
                current = [line]
            elif current and line:
                current.append(line)
        if current:
            sections.append(current)
        milestones: List[Dict[str, str]] = []
        for lines in sections:
            if not lines or (not legacy_heading.search(lines[0]) and not flexible_heading.search(lines[0])
                             and due_seconds(lines[0], None) is None
                             and not re.match(r"^\d+[.)]\s+", lines[0])):
                continue
            label = re.sub(r"[.\s]+$", "", lines[0])
            detail = " ".join(lines[1:]).strip() or lines[0]
            # Do not turn conditional prose, quoted scene instructions or volume
            # descriptions into physical commands. Complex controls need an explicit schedule action.
            simple_command = (len(detail) <= 150 and re.match(r"^(?:set|change|switch|restore)\b", detail, re.I)
                              and not re.search(r"\b(?:if|when|until|after|not|volume)\b", detail, re.I))
            action = direct_vector_action(detail, vector_stopped=False) if SESSION_MODE == "vector" and simple_command else None
            # Prose mentioning a control dimension is narrative guidance, not an
            # executable command. Only a parsed action (or an explicit structured
            # schedule entry above) becomes verification-gated control work.
            control = bool(action)
            milestones.append({"label": label, "detail": detail[:900],
                               "due_seconds": due_seconds(" ".join(lines), len(milestones) * 180),
                               "control": control,
                               "flexible_phase": bool(flexible_heading.search(lines[0])),
                               "action": {"name": action[0], "args": action[1]} if action else None})
        return milestones

    @staticmethod
    def _brief_milestone_complete(label: str, transcript: str) -> bool:
        """Use conservative evidence rules; elapsed time alone never completes a beat."""
        label_l = label.lower()
        text = transcript.lower()
        if "recall beat" in label_l:
            return bool(re.search(r"(?:felt|feel|pre-?charg|remember).{0,90}(?:stocking|nylon|lace)|(?:stocking|nylon|lace).{0,90}(?:felt|feel|pre-?charg|remember)", text))
        if "name marker" in label_l:
            return "stephanie emerges" in text
        if "dress code" in label_l:
            cues = ("stocking", "suspender", "heel", "mistress")
            return sum(cue in text for cue in cues) >= 3
        if "pleasure-attire" in label_l:
            return bool(re.search(r"(?:signal|carrier).{0,100}(?:silence|still|sleep|lower|absence)|(?:silence|absence).{0,100}(?:stocking|nylon|garter|lace)", text))
        if "user-defined sensation" in label_l:
            return bool(re.search(r"spread (?:your|her) legs", text) and "strap-on" in text)
        if "tension-restraint" in label_l:
            return bool(re.search(r"held (?:up|back).{0,80}held (?:up|back)|heels?.{0,100}knees?", text))
        if "ending" in label_l:
            return bool(re.search(r"(?:carry|familiar) (?:the |this )?ache|continuation.{0,80}next session", text))
        return False

    def _session_brief_conductor_context(self) -> str:
        due = self.conductor.due(time.monotonic())
        if not self.conductor.milestones:
            return ""
        rows = [f"{m['label']}: {m['state']} (due {m['due_seconds']:g}s)"
                for m in self.conductor.milestones]
        required_control = next((m for m in due if m.get("action")), None)
        flexible_due = [m for m in due if m.get("flexible_phase")]
        current = required_control or (flexible_due[-1] if flexible_due else (due[0] if due else None))
        current_label = ("CURRENT REQUIRED CONTROL" if current and current.get("action")
                         else "CURRENT FLEXIBLE PHASE")
        elapsed = max(0.0, time.monotonic() - self.conductor.started_at)
        duration_note = ""
        if SESSION_MODE == "signal_lab" and self.conducted_session_ends_at:
            total = max(elapsed, self.conducted_session_ends_at - self.conductor.started_at)
            configured = self.conducted_session_duration_minutes or total / 60.0
            if self.conducted_session_completed_at:
                neutral = "confirmed neutral" if self.conducted_session_neutral_confirmed else "neutral was not confirmed"
                duration_note = (
                    f"\nAUTHORITATIVE SESSION TIME: the preset's {configured:g}-minute conducted window is complete; "
                    f"Signal Lab {neutral}. The journal may be longer because it includes setup and closing conversation. "
                    "Never claim that the conducted session is still active or invent a different planned duration."
                )
            else:
                duration_note = (f"\nAUTHORITATIVE SESSION TIME: elapsed {elapsed/60:.1f} minutes; "
                                 f"configured end {total/60:.1f} minutes. Do not claim the configured "
                                 "duration has elapsed before that end time.")
        return (
            "\nSESSION BRIEF CONDUCTOR (higher priority than optional narrative beats):\n"
            + "\n".join(rows)
            + (f"\n{current_label}: {current['label']}: {current['detail']}" if current else
               "\nNo pending milestone is due. Respond naturally; do not replay completed beats.")
            + duration_note
            + "\nTreat a flexible phase as direction rather than a checklist: phases may overlap, adapt to feedback, "
              "or be revisited, and the latest phase whose window has begun takes precedence. Do not force a stock phrase "
              "to mark it complete. Explicit control actions remain mandatory once due and require matching controller "
              "verification; conversation cannot confirm them. Physical actions still require normal authority and safety checks."
        )

    def _brief_control_authorized(self, name: str) -> bool:
        if SESSION_MODE == "signal_lab":
            return (self.conducted_session_active and not self.conducted_session_held
                    and name in {"signal_lab_apply", "signal_lab_heart_tempo", "signal_lab_neutral"})
        return (self.vector_autonomous_active and not self.vector_autonomous_held
                and name.startswith("vector_") and name not in
                {"vector_stop", "vector_resume", "vector_neutral", "vector_get_state"})

    def _run_due_brief_control(self) -> bool:
        for milestone in self.conductor.due(time.monotonic()):
            action = milestone.get("action")
            if not action or not self._brief_control_authorized(action["name"]):
                continue
            if self.recording or self.current_state != "READY":
                return False
            if SESSION_MODE == "signal_lab":
                snapshot = self.signal_lab_get("/v1/state", timeout=1.5)
                if not ((snapshot.get("live") or {}).get("headphone_monitor") or {}).get("audible"):
                    return False
                if self.conducted_session_ends_at and time.monotonic() >= self.conducted_session_ends_at:
                    return False
            else:
                state = self._refresh_authoritative_state(timeout=1.2)
                if str(state.get("engine_state", "")).lower() in {"stopped", "neutral"}:
                    return False
            self.reset_action_ledger()
            self.execute_tool(action["name"], action["args"])
            reply = self._publish_reply("")
            self.enqueue_voice(reply, Timings(), voice_id=self.next_voice_tx("BRIEF"))
            return True
        return False

    def _verified_control(self, name: str, args: Dict[str, Any], result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict) or not result.get("ok") or result.get("noop") or result.get("error"):
            return False
        accepted = result.get("accepted")
        if isinstance(accepted, dict) and (accepted.get("ok") is False or accepted.get("error")):
            return False
        if name == "signal_lab_apply":
            command = result.get("command") or {}
            state = result.get("verification_state") or {}
            live = state.get("live") or {}
            return bool(result.get("status") == "applied_offline" and command.get("request_id")
                        and matches(args.get("left"), command.get("left_semantic"))
                        and matches(args.get("right"), command.get("right_semantic"))
                        and matches(args.get("transition_seconds"), command.get("transition_seconds"))
                        and (state.get("last_command") or {}).get("request_id") == command["request_id"]
                        and command.get("left_signal") and command.get("right_signal")
                        and matches(command["left_signal"], live.get("left_signal"))
                        and matches(command["right_signal"], live.get("right_signal")))
        if name == "signal_lab_neutral":
            state = result.get("verification_state") or {}
            live = state.get("live") or {}
            return bool(result.get("status") == "neutral_offline" and state.get("ok")
                        and state.get("last_command") is None and live.get("left_signal") == {}
                        and live.get("right_signal") == {} and live.get("playing") is False
                        and (live.get("headphone_monitor") or {}).get("audible") is False)
        if name == "signal_lab_heart_tempo":
            state = result.get("verification_state") or {}
            control = state.get("heart_tempo_control") or {}
            mode = str(args.get("mode") or "").lower()
            if mode == "off":
                return bool(result.get("status") == "heart_tempo_control_requested"
                            and not control.get("active") and control.get("mode") == "off")
            heart = state.get("heart_tempo") or {}
            live = state.get("live") or {}
            target = str(args.get("target") or "").lower()
            signal = live.get(f"{target}_signal") or {}
            expected_ratio = {"escalation": 1.0, "double": 2.0, "relief": 0.9}.get(mode)
            if expected_ratio is None:
                return False
            expected_hz = float(heart.get("bpm") or 0.0) * expected_ratio / 60.0
            actual_hz = signal.get("amFreq")
            return bool(result.get("status") == "heart_tempo_control_requested"
                        and control.get("active") and control.get("target") == target
                        and control.get("mode") == mode and matches(control.get("ratio"), expected_ratio)
                        and heart.get("fresh") and isinstance(actual_hz, (int, float))
                        and abs(float(actual_hz) - expected_hz) <= 0.02)
        if name == "vector_generated_motion_plan":
            motion = (accepted if isinstance(accepted, dict) else result).get("motion_source") or {}
            expected = {k: v for k, v in args.items() if k != "reason"}
            return bool(expected and motion.get("active") and not motion.get("held")
                        and matches(expected, motion.get("plan")))
        key, desired = self._action_state_key_value(name, args)
        if key and desired is not None:
            actual = self._semantic_snapshot_from_state(result.get("state") or {}).get(key)
            return actual is not None and matches(desired, actual)
        # Unknown verification schemas remain unconfirmed instead of trusting HTTP success.
        return False

    def _control_state_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        """Capture compact controller-owned evidence immediately around a control action."""
        try:
            if name.startswith("signal_lab_"):
                state = self.signal_lab_get("/v1/state", timeout=1.0)
                live = (state or {}).get("live") or {}
                return {
                    "mode": (state or {}).get("mode"),
                    "last_command": (state or {}).get("last_command"),
                    "live": {
                        "left_signal": live.get("left_signal"),
                        "right_signal": live.get("right_signal"),
                        "playing": live.get("playing"),
                        "headphone_monitor": live.get("headphone_monitor"),
                        "physical_output_enabled": live.get("physical_output_enabled"),
                    },
                    "heart_tempo": (state or {}).get("heart_tempo"),
                    "heart_tempo_control": (state or {}).get("heart_tempo_control"),
                }
            return self.compact_state(self._refresh_authoritative_state(timeout=1.0))
        except Exception as exc:
            return {"unavailable": f"{type(exc).__name__}: {exc}"}

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        record_state = name not in {"vector_get_state", "signal_lab_get_state"}
        before_state = self._control_state_snapshot(name) if record_state else None
        matching = [m for m in self.conductor.due(time.monotonic())
                    if m.get("action") == {"name": name, "args": args}]
        claimed = [m for m in matching if self.conductor.transition(m["id"], "attempted")]
        try:
            result = self._execute_tool(name, args)
        except Exception as exc:
            for m in claimed:
                self.conductor.transition(m["id"], "failed", {"error": str(exc)})
            raise
        autonomous = getattr(self.director_thread, "generation", None) is not None
        if (claimed or autonomous) and name in {"signal_lab_apply", "signal_lab_heart_tempo", "signal_lab_neutral"} and result.get("ok"):
            # The bridge's POST queues a GUI event. Confirm only after its live
            # readback matches that exact command, not merely the HTTP acknowledgement.
            deadline = time.monotonic() + 1.0
            while True:
                try:
                    result = dict(result, verification_state=self.signal_lab_get("/v1/state", timeout=0.3))
                except Exception:
                    break
                if self._verified_control(name, args, result) or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        verified = self._verified_control(name, args, result)
        if record_state:
            after_state = (result.get("verification_state") or result.get("state")
                           or self._control_state_snapshot(name))
            simulation = bool(name.startswith("signal_lab_") and (
                result.get("offline_simulation") is True
                or str(result.get("status") or "").endswith("_offline")
                or ((after_state or {}).get("live") or {}).get("physical_output_enabled") is False
            ))
            verification_status = "simulation" if simulation else ("verified" if verified else "unverified")
            journal_event = getattr(self, "_journal_event", None)
            if journal_event is not None:
                journal_event("control_action_executed" if result.get("ok") else "control_action_failed", {
                    "name": name, "args": dict(args), "ok": bool(result.get("ok")),
                    "verified": bool(verified), "verification_status": verification_status,
                    "status": result.get("status"),
                    "noop": bool(result.get("noop")), "error": result.get("error"),
                    "before_state": before_state, "after_state": after_state,
                })
        for m in claimed:
            self.conductor.transition(m["id"], "confirmed" if verified else "failed", result)
        if autonomous and name not in {"vector_get_state", "signal_lab_get_state"} and not result.get("noop"):
            if verified:
                ledger = {"executed": [{"name": name, "args": args, "result": result}], "failed": []}
                factual = re.sub(r"^Done\.\s*", "", self._grounded_confirmation(ledger), flags=re.I)
                if name == "vector_generated_motion_plan":
                    factual = (f"I’ve selected {args.get('pattern')} motion with a stroke duration of "
                               f"{args.get('stroke_duration_ms')} milliseconds.")
                self.conductor.announce(factual)
            elif claimed or result.get("ok"):
                self.conductor.announce("I tried the scheduled change, but couldn’t verify that it took effect."
                                        if claimed else "I requested a change, but couldn’t verify that it took effect.")
        return result

    def _autonomous_interrupted(self) -> bool:
        generation = getattr(self.director_thread, "generation", None)
        return generation is not None and (self.recording or self.current_state in {"LISTENING", "TRANSCRIBING"}
                                           or generation != self.current_voice_generation())

    def _publish_reply(self, text: str) -> str:
        original = text
        generation = getattr(self.director_thread, "generation", None)
        deferred = self._autonomous_interrupted()
        pending = self.conductor.pending_text()
        if deferred:
            text = ""
        elif pending:
            # Facts are inserted after model grounding/repetition gates. They retain their
            # own controller evidence even after a reactive turn resets its action ledger.
            text = pending if generation is not None or text == self._profile_fallback() else pending + (" " + text if text else "")
        if self.history and self.history[-1] == {"role": "assistant", "content": original}:
            if text:
                self.history[-1]["content"] = text
            else:
                self.history.pop()
        elif text:
            self.history.append({"role": "assistant", "content": text})
            self.trim_history()
        if text:
            self.emit("assistant", text)
        return text

    def narrative_context(self, proactive: bool = False) -> Tuple[str, str]:
        """Return deterministic creative direction for Qwen, never physical authority."""
        beat = self._select_narrative_beat(proactive=proactive)
        stage = self._narrative_arc_stage()
        motifs = self._session_motif_candidates()
        unused = [m for m in motifs if m.lower() not in {x.lower() for x in self.narrative_recent_motifs[-5:]}]
        motif_hint = "none required"
        if unused and beat in ("tease", "callback", "foreshadow", "challenge") and self.narrative_rng.random() < self.narrative_randomness():
            motif_hint = self.narrative_rng.choice(unused[:8])
            self.narrative_recent_motifs.append(motif_hint)
            self.narrative_recent_motifs = self.narrative_recent_motifs[-8:]
        recent = ", ".join(self.narrative_recent_beats[-4:]) or "none"
        callbacks = ", ".join(self.narrative_callback_queue[-3:]) or "none"
        modes = ", ".join(self.narrative_recent_modes[-3:]) or "none"
        brief_conductor = self._session_brief_conductor_context()
        text = (
            self._state_arc_context() + "\n" +
            "NARRATIVE DIRECTOR STATE (creative direction only; never overrides grounding or Vector limits):\n"
            f"arc_template={self.narrative_arc_template}; arc_stage={stage}; selected_beat={beat}; recent_beats={recent}; optional_motif={motif_hint}; callback_candidates={callbacks}; recent_modes={modes}.\n"
            "Use the selected beat as a light dramatic intention, not a mandatory phrase. Build continuity from earlier conversation. "
            "If an optional motif is supplied, use it only if it fits naturally; do not repeat it mechanically. A callback candidate is something worth revisiting later, not immediately. "
            "Vary emotional mode and sentence shape. Avoid recently repeated metaphors, catchphrases, repeated readiness questions, and back-to-back proposals. "
            "A beat may be purely conversational and does not authorize any Vector action." + brief_conductor
        )
        log(f"NARRATIVE arc={self.narrative_arc_template!r} stage={stage} beat={beat!r} motif={motif_hint!r} recent={self.narrative_recent_beats[-4:]}")
        return beat, text

    def _record_narrative_beat(self, beat: str) -> None:
        if not beat or beat == "natural":
            return
        self.narrative_recent_beats.append(beat)
        self.narrative_recent_beats = self.narrative_recent_beats[-8:]
        mode = {"challenge":"assertive", "relief":"gentle", "tease":"playful", "foreshadow":"anticipatory", "callback":"reflective", "observe":"attentive", "propose":"decisive", "quiet":"restrained"}.get(beat, beat)
        self.narrative_recent_modes.append(mode)
        self.narrative_recent_modes = self.narrative_recent_modes[-8:]
        # A motif used once becomes a possible later callback, but recent-motif suppression
        # prevents immediate repetition.
        if self.narrative_recent_motifs:
            m = self.narrative_recent_motifs[-1]
            if m not in self.narrative_callback_queue:
                self.narrative_callback_queue.append(m)
                self.narrative_callback_queue = self.narrative_callback_queue[-6:]
        self.narrative_turn_counter += 1

    def _capture_conversational_proposal(self, reply: str) -> bool:
        """Capture executable proposals made during ordinary conversation.

        v0.28 only captured scheduled proactive proposals, so a user asking
        'what do you suggest?' could hear a valid proposal that 'Please do' could
        not execute. Narrative Director needs proposal continuity everywhere.
        """
        if not reply:
            return False
        if re.search(r"\b(?:can(?:not|'t) change|unable to change|do not have|don't have|none of (?:the )?available)\b", reply, re.I):
            log("PROPOSAL capture suppressed for explicit inability statement")
            return False
        if not has_proposal_cue(reply):
            return False
        action = direct_vector_action(reply, vector_stopped=False)
        if not action:
            return False
        if self._action_is_noop(action[0], action[1]):
            log(f"PROPOSAL rejected no-op {action[0]} {action[1]}")
            self.last_proposal_rejected_noop = True
            self.last_rejected_noop_action = (action[0], dict(action[1]))
            return False
        self.pending_proposal_text = reply
        self.pending_proposal_action = action
        self.pending_proposal_at = time.monotonic()
        log(f"PROPOSAL captured conversational action {action[0]} {action[1]}")
        return True

    def _capture_signal_lab_proposal(self, reply: str) -> bool:
        """Remember a proposed Signal Lab edit so a short approval can execute it."""
        if SESSION_MODE != "signal_lab" or not reply:
            return False
        proposes = re.search(
            r"\b(?:i can|i could|i would|i(?:'|’)ll|let me|shall i|would you like|i propose|i suggest)\b",
            reply, re.I,
        )
        has_signal_dimension = re.search(
            r"\b(?:signal|lane|stairway|moaner|intensity|volume|texture|smooth|rough|carrier|"
            r"vibration|pulse|tap|modulation|drift|secondary)\w*\b",
            reply, re.I,
        )
        has_change = re.search(
            r"\b(?:change|adjust|set|make|increase|decrease|raise|lower|sharpen|soften|"
            r"smooth|roughen|slow|speed|tighten|strengthen|weaken|hold|start)\w*\b",
            reply, re.I,
        )
        if not (proposes and has_signal_dimension and has_change):
            return False
        self.pending_signal_lab_proposal = reply
        log("SIGNAL LAB proposal captured for short-form approval")
        return True

    @staticmethod
    def _proposal_mentions_multiple_actions(reply: str) -> bool:
        """Detect stacked control proposals that cannot share one approval."""
        t = (reply or "").lower()
        change = r"(?:shift|move|set|switch|change|ramp|bump|raise|lower|increase|decrease|make|push)"
        categories = set()
        for clause in re.split(r"\b(?:and|while|but)\b|[.;!?]", t):
            if not re.search(rf"\b{change}\w*\b", clause):
                continue
            if re.search(r"\b(?:top|glans|shaft|root)\b", clause):
                categories.add("top")
            if re.search(r"\b(?:bottom|prostate|anal|perineum)\b", clause):
                categories.add("bottom")
            if re.search(r"\b(?:rough|smooth)\b", clause):
                categories.add("texture")
            if re.search(r"\b(?:still|subtle|normal|lively)\b", clause):
                categories.add("variation")
            if re.search(r"\b(?:spatial gain|gain|intensity)\b", clause):
                categories.add("gain")
        return len(categories) > 1

    def _reroll_after_noop(self, base_messages: List[Dict[str, Any]], rejected: Optional[Tuple[str, Dict[str, Any]]], draft: str = "") -> Tuple[str, bool]:
        """Resolve a model-generated no-op locally, without another LLM request."""
        rejected_desc = f"{rejected[0]} {rejected[1]}" if rejected else "unknown setting"
        choices = {
            "gwendolyn": ["I’ll keep the present balance and watch what the script gives us next.", "The current arrangement already suits this moment, so I’ll hold it steady."],
            "anna": ["We’ll keep the current configuration steady and observe the next genuine change.", "Nothing needs forcing here. I’ll maintain the present balance and continue watching."],
            "natasha": ["No need to meddle with a setting that is already doing its job. We hold here.", "The arrangement is already where I want it. Let the script make the next move."],
            "sophie": ["We keep this balance for now; there is more interest in patience than needless adjustment.", "This setting is already serving the moment, so I shall let it breathe."],
            "vesper": ["Hold. The present contrast is enough; the next change should earn its place.", "Leave it. Restraint is the more useful choice at this point."],
            "aurelia": ["We can leave the balance exactly here and allow the next real opportunity to reveal itself.", "There is no advantage in disturbing what is already settled. I’ll wait."],
        }
        pool = choices.get(self.director_voice_key(), choices["gwendolyn"])
        recent = {self._normalized_reply(item) for item in self.recent_spoken_replies[-6:]}
        pieces = re.split(r"(?<=[.!?])\s+|\n+", self._sanitize_model_text(draft))
        preserved = " ".join(
            piece.strip() for piece in pieces
            if piece.strip()
            and not direct_vector_action(piece, vector_stopped=False)
            and not self._contains_unexecuted_action_claim(piece)
            and not has_proposal_cue(piece)
        ).strip()
        if len(preserved) >= 30:
            reply = preserved
            log("PROPOSAL no-op removed while preserving conversational draft")
        else:
            reply = next((item for item in pool if self._normalized_reply(item) not in recent), pool[0])
        log(f"PROPOSAL no-op resolved locally rejected={rejected_desc} voice={self.director_voice_key()!r}")
        self.last_proposal_rejected_noop = False
        self.last_rejected_noop_action = None
        return reply, False

    def proactive_interval_seconds(self) -> Optional[float]:
        value = str(self.profile.get("proactive_director", "5 min") or "Off").strip().lower()
        base = None
        if value.startswith("5"):
            base = 300.0
        elif value.startswith("10"):
            base = 600.0
        if base is None:
            return None
        if bool(self.cfg.get("director_personality_profile", True)):
            base *= float(self.director_voice_profile().get("initiative_multiplier", 1.0))
        return base

    def conducted_interval_seconds(self) -> float:
        return max(60.0, float(self.cfg.get("conducted_session_interval_seconds", 180.0)))

    def vector_autonomous_interval_seconds(self) -> float:
        return max(60.0, float(self.cfg.get("vector_autonomous_interval_seconds", 180.0)))

    def vector_autonomous_generates_motion(self) -> bool:
        return bool(self.cfg.get("vector_autonomous_generate_tcode", False))

    def _design_and_apply_generated_motion(self, state: Dict[str, Any], narrative: str = "") -> Dict[str, Any]:
        prompt = (
            "AUTONOMOUS VECTOR INCOMING MOTION. Return ONLY one JSON object with pattern (sine, triangle, or breathing), "
            "minimum (0.0-0.90), maximum (0.10-1.0 and at least 0.10 above minimum), stroke_duration_ms (125-3000), "
            "transition_seconds (1-15), duration_seconds (30-600 review horizon), and a short reason. Motion continues beyond the "
            "review horizon until you replace it or the user explicitly holds, stops, or returns to authored T-code. Choose a coherent bounded plan from the "
            "authoritative Vector state, selected narrative arc, and recent user feedback. Positive feedback can develop the current "
            "quality; discomfort or reduction language must reduce travel or increase stroke_duration_ms. Treat stroke_duration_ms as "
            "the one-way funscript point-to-point time, so speed depends on both travel distance and duration. Do not emit raw T-code samples."
        )
        messages = [
            {"role": "system", "content": self.system_prompt + "\n\n" + narrative},
            {"role": "system", "content": "AUTHORITATIVE VECTOR STATE:\n" + json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
            *self.history[-8:],
            {"role": "user", "content": prompt},
        ]
        designed = self.ollama_chat(messages, None, phase="vector-motion-json")
        args = extract_generated_motion_args(str(designed.get("content") or ""))
        if not args:
            return {"ok": False, "error": "malformed generated-motion plan"}
        result = self.execute_tool("vector_generated_motion_plan", args)
        if isinstance(result, dict):
            result["autonomous_plan"] = dict(args)
        return result

    def _autonomous_change_narration(self, action_name: str, action_args: Dict[str, Any],
                                     reason: str, narrative: str) -> str:
        """Narrate a verified autonomous decision; reactive confirmations stay terse."""
        grounded = self._grounded_confirmation()
        if action_name == "vector_generated_motion_plan":
            grounded = (
                f"I’ve selected {action_args.get('pattern', 'a')} incoming motion across L0 "
                f"{float(action_args.get('minimum', 0.0)):.2f} to {float(action_args.get('maximum', 1.0)):.2f}, "
                f"with {int(float(action_args.get('stroke_duration_ms', 0)))} milliseconds per one-way stroke."
            )
        prompt = (
            "AUTONOMOUS VECTOR CHANGE NARRATION. The change has already been validated and executed. Speak as the active Director "
            "in two to four expressive sentences. Announce the specific change naturally, then explain why you chose it now and how "
            "it advances the selected session description, goal, narrative arc, or the user’s recent feedback. This was your autonomous "
            "decision, so do not say 'Done', sound like an assistant acknowledging an order, ask permission, mention tools/APIs/JSON, "
            "or invent a physical sensation the user has not reported. Keep the factual change consistent with the grounded confirmation.\n\n"
            f"GROUNDED CHANGE: {grounded}\n"
            f"ACTION: {action_name} {json.dumps(action_args, ensure_ascii=False, sort_keys=True)}\n"
            f"DECISION REASON: {reason}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt + "\n\n" + narrative},
            *self.history[-6:],
            {"role": "user", "content": prompt},
        ]
        narrated = self.ollama_chat(messages, None, phase="vector-autonomous-narration")
        spoken = self._finalize_reply(
            self._sanitize_model_text(narrated.get("content") or ""), fallback="", max_chars=650)
        if spoken and not re.match(r"^\s*done\b", spoken, re.I):
            return spoken
        factual = re.sub(r"^Done\.\s*", "", grounded, flags=re.I)
        arc = str(self.narrative_arc_template or "current session arc").strip()
        why = str(reason or "to develop the current session deliberately").strip().rstrip(".")
        purpose = why[0].lower() + why[1:] if why else f"to serve the {arc}"
        return f"{factual} I chose it now because {purpose}."

    def _handle_vector_autonomous_command(self, text: str, timings: Timings) -> bool:
        """Explicitly arm or disarm bounded Vector autonomy for this Vector-only process."""
        if SESSION_MODE != "vector":
            return False
        raw = text or ""
        start = is_vector_autonomous_start_command(raw)
        hold = bool(re.search(r"\b(?:hold|pause|freeze)\b.{0,35}\b(?:autonomous session|vector autonomy|your changes)\b", raw, re.I))
        resume = bool(re.search(r"\b(?:resume|continue)\b.{0,35}\b(?:autonomous session|vector autonomy|conducting)\b", raw, re.I))
        stop = bool(re.search(r"\b(?:stop|end|disarm)\b.{0,35}\b(?:autonomous session|vector autonomy|conducting)\b", raw, re.I))
        if not any((start, hold, resume, stop)):
            return False
        if start:
            try:
                state = self._refresh_authoritative_state(timeout=1.5)
                if str(state.get("engine_state") or "").strip().lower() == "stopped":
                    reply = "Vector is stopped, so I haven’t armed autonomous control. Resume Vector first, then ask me again."
                else:
                    if not self.vector_autonomous_active:
                        self.conductor.restart(time.monotonic())
                    self.vector_autonomous_active = True
                    self.vector_autonomous_held = False
                    self.vector_autonomous_decision_count = 0
                    self.last_meaningful_vector_change_at = time.monotonic() - self.vector_autonomous_interval_seconds()
                    self.proactive_snooze_until = 0.0
                    self.pending_proposal_text = ""
                    self.pending_proposal_action = None
                    self.pending_proposal_at = 0.0
                    generated_result = None
                    if self.vector_autonomous_generates_motion():
                        generated_result = self._design_and_apply_generated_motion(state)
                        if not generated_result.get("ok"):
                            raise RuntimeError(generated_result.get("error") or "Vector rejected generated motion")
                        self.vector_autonomous_decision_count = 1
                    self._journal_event("vector_autonomous_session_started", {
                        "decision_interval_seconds": self.vector_autonomous_interval_seconds(),
                        "narrative_arc": self.narrative_arc_template,
                        "initial_vector_state": state,
                        "authority": "explicit_user_grant",
                        "generated_incoming_motion": self.vector_autonomous_generates_motion(),
                    })
                    source_note = (" I have also replaced authored incoming T-code with a Vector-validated motion plan."
                                   if self.vector_autonomous_generates_motion() else "")
                    reply = ("Autonomous Vector control is armed by your explicit authority. I may now make one considered change at a time, "
                             "inside Vector’s existing operator-owned limits, and I will respond immediately to hold, stop, neutral, or shutdown requests.")
                    reply += source_note
                    log(f"VECTOR AUTONOMOUS armed interval={self.vector_autonomous_interval_seconds():.0f}s")
            except Exception as exc:
                log(f"VECTOR AUTONOMOUS start failed: {exc}")
                reply = "I couldn’t verify Vector’s live state, so autonomous control has not been armed."
        elif hold:
            self.vector_autonomous_held = True
            if self.vector_autonomous_generates_motion():
                self.execute_tool("vector_generated_motion_hold", {})
            self._journal_event("vector_autonomous_session_held", {"decision_count": self.vector_autonomous_decision_count})
            reply = "I’m holding the current Vector state. Autonomous authority remains armed, but I’ll make no changes until you tell me to resume."
        elif resume:
            if not self.vector_autonomous_active:
                reply = "There is no autonomous Vector session armed to resume."
            else:
                self.vector_autonomous_held = False
                if self.vector_autonomous_generates_motion():
                    self.execute_tool("vector_generated_motion_resume", {})
                self.last_meaningful_vector_change_at = time.monotonic() - self.vector_autonomous_interval_seconds()
                self.proactive_snooze_until = 0.0
                self._journal_event("vector_autonomous_session_resumed", {"decision_count": self.vector_autonomous_decision_count})
                reply = "Autonomous Vector control is resumed. I’ll reassess the live state, narrative arc, timeline, and your recent feedback before the next change."
        else:
            self.vector_autonomous_active = False
            self.vector_autonomous_held = False
            if self.vector_autonomous_generates_motion():
                self.execute_tool("vector_use_authored_tcode", {})
            self._journal_event("vector_autonomous_session_disarmed", {"decision_count": self.vector_autonomous_decision_count, "current_vector_state_preserved": True})
            reply = "Autonomous Vector control is disarmed. I’ve stopped making autonomous changes and left the current Vector state in place."
        self._emit_local_turn(text, reply, timings, "VECTOR-AUTO")
        return True

    def _handle_conducted_session_command(self, text: str, timings: Timings) -> bool:
        if SESSION_MODE != "signal_lab":
            return False
        raw = text or ""
        consent_confirmation = bool(self.conducted_consent_pending and self._looks_like_approval(raw))
        consent_rejection = bool(self.conducted_consent_pending and self._looks_like_rejection(raw))
        start = is_conducted_session_start_command(raw)
        if (self.conducted_start_pending and re.search(
                r"\bmonitor\b.{0,45}\b(?:armed|playing|audible|started)\b|"
                r"\b(?:armed|started)\b.{0,35}\bmonitor\b", raw, re.I)):
            start = True
            log("CONDUCTED SESSION pending start acknowledged; rechecking live monitor")
        if consent_confirmation:
            start = True
        if consent_rejection:
            self.conducted_consent_pending = False
            self.conducted_start_pending = False
            self._journal_event("conducted_go_ahead_declined", {"user_text": raw})
            self._emit_local_turn(raw, "Understood. Conducted Session remains disarmed and the current signal is unchanged.", timings, "CONDUCT")
            return True
        hold = bool(re.search(r"\b(?:hold|freeze)\b.{0,35}\b(?:conducted session|current signal|autonomy)\b", raw, re.I))
        resume = bool(re.search(r"\b(?:resume|continue)\b.{0,35}\b(?:conducted session|conducting|autonomy)\b", raw, re.I))
        stop = bool(re.search(r"\b(?:stop|end|disarm)\b.{0,35}\b(?:conducted session|conducting|autonomy)\b", raw, re.I))
        if not any((start, hold, resume, stop)):
            return False
        if start:
            try:
                snapshot = self.signal_lab_get("/v1/state", timeout=1.5)
                live = (snapshot or {}).get("live") or {}
                monitor = live.get("headphone_monitor") or {}
                if not monitor.get("audible"):
                    self.conducted_start_pending = True
                    self._journal_event("conducted_start_waiting_for_monitor", {"monitor": monitor})
                    reply = "Arm the headphone monitor and start playback first; I won’t begin an autonomous session without an audible operator-controlled output."
                elif not consent_confirmation:
                    self.conducted_consent_pending = True
                    self.conducted_start_pending = False
                    self._journal_event("conducted_go_ahead_announced", {
                        "announcement": "bounded autonomous lane changes inside the active preset",
                        "monitor": monitor,
                    })
                    reply = (
                        "The monitor is audible and the preset is ready. If you confirm, I may make bounded autonomous "
                        "changes to both Signal Lab lanes inside the active preset, one considered change at a time. "
                        "Would you like me to begin?"
                    )
                else:
                    preset = (snapshot or {}).get("active_preset") or {}
                    minutes = max(1.0, float(preset.get("session_duration_minutes") or 45.0))
                    now = time.monotonic()
                    session_elapsed = max(0.0, now - self.conductor.started_at)
                    if session_elapsed >= minutes * 60.0:
                        self.conducted_consent_pending = False
                        self._journal_event("conducted_session_start_rejected", {
                            "reason": "closure_window_open",
                            "elapsed_seconds": round(session_elapsed, 3),
                            "configured_duration_seconds": round(minutes * 60.0, 3),
                        })
                        reply = "The session’s closure window is already open, so I won’t restart its opening logic or arm new autonomous work now."
                        self._emit_local_turn(text, reply, timings, "CONDUCT")
                        return True
                    # A Conducted Session begins inside its operator-owned
                    # envelope. The loaded reference JSON is inspiration, not
                    # an exemption from the selected preset. Establish and
                    # verify a deterministic baseline before granting autonomy.
                    baseline_args = signal_lab_preset_start_args(snapshot)
                    self.reset_action_ledger()
                    baseline_result = self.execute_tool("signal_lab_apply", baseline_args)
                    if not self._verified_control("signal_lab_apply", baseline_args, baseline_result):
                        self.conducted_consent_pending = False
                        self.conducted_start_pending = False
                        self._journal_event("conducted_preset_baseline_rejected", {
                            "active_preset": preset,
                            "requested_baseline": baseline_args,
                            "result": baseline_result,
                        })
                        raise RuntimeError("Signal Lab did not verify the active preset baseline")
                    snapshot = baseline_result.get("verification_state") or self.signal_lab_get("/v1/state", timeout=1.5)
                    live = (snapshot or {}).get("live") or live
                    self._journal_event("conducted_preset_baseline_confirmed", {
                        "active_preset": preset,
                        "applied_baseline": baseline_args,
                        "verified_live_state": live,
                    })
                    self.conducted_session_active = True
                    self.conducted_session_held = False
                    self.conducted_session_started_at = now
                    self.conducted_session_ends_at = self.conductor.started_at + minutes * 60.0
                    self.conducted_session_duration_minutes = minutes
                    self.conducted_session_completed_at = 0.0
                    self.conducted_session_neutral_confirmed = False
                    self.last_conducted_decision_at = now - self.conducted_interval_seconds()
                    self.conducted_decision_count = 0
                    self.conducted_start_pending = False
                    self.conducted_consent_pending = False
                    self.conducted_final_wave_announced = False
                    self.conducted_narrative_anchor = self._narrative_arc_stage()
                    self.proactive_snooze_until = 0.0
                    self.pending_signal_lab_proposal = ""
                    self._journal_event("conducted_session_started", {
                        "duration_minutes": minutes,
                        "decision_interval_seconds": self.conducted_interval_seconds(),
                        "narrative_arc": self.narrative_arc_template,
                        "active_preset": preset,
                        "initial_live_state": live,
                        "authority": "assistant_announcement_then_explicit_user_consent",
                        "consent_text": raw,
                        "session_elapsed_at_activation_seconds": round(session_elapsed, 3),
                        "clock_source": "session_conductor_started_at",
                        "configured_deadline_seconds_from_session_start": round(minutes * 60.0, 3),
                        "narrative_anchor": self.conducted_narrative_anchor,
                    })
                    reply = (
                        f"Conducted Session is armed for the preset’s {minutes:g}-minute duration. "
                        f"I’ve established and verified the {preset.get('name') or 'active'} preset baseline. "
                        f"Your go-ahead is recorded. I’ll continue from the current {self.conducted_narrative_anchor} stage rather than restarting the opening."
                    )
                    log(f"CONDUCTED SESSION armed duration={minutes:g}m interval={self.conducted_interval_seconds():.0f}s "
                        f"clock=session_conductor_started_at elapsed_at_activation={session_elapsed:.3f}s")
            except Exception as exc:
                log(f"CONDUCTED SESSION start failed: {exc}")
                reply = "I couldn’t verify Signal Lab’s live preset and monitor, so Conducted Session has not started."
        elif hold:
            self.conducted_session_held = True
            self._journal_event("conducted_session_held", {"decision_count": self.conducted_decision_count})
            reply = "I’m holding the current signal. Conducted Session remains armed, but I’ll make no autonomous changes until you tell me to resume."
            log("CONDUCTED SESSION held")
        elif resume:
            if not self.conducted_session_active:
                reply = "There is no Conducted Session armed to resume."
            else:
                self.conducted_session_held = False
                self.last_conducted_decision_at = time.monotonic() - self.conducted_interval_seconds()
                self.proactive_snooze_until = 0.0
                self._journal_event("conducted_session_resumed", {"decision_count": self.conducted_decision_count})
                reply = "Conducting resumed. I’ll reassess the live state and our recent conversation before making the next change."
                log("CONDUCTED SESSION resumed")
        else:
            self.conducted_session_active = False
            self.conducted_session_held = False
            self.conducted_start_pending = False
            self._journal_event("conducted_session_disarmed", {
                "decision_count": self.conducted_decision_count,
                "current_signal_preserved": True,
            })
            reply = "Conducted Session is disarmed. I’ve stopped making autonomous changes and left the current signal in place."
            log("CONDUCTED SESSION disarmed; current signal held")
        self._emit_local_turn(text, reply, timings, "CONDUCT")
        return True

    def proactive_tick(self) -> None:
        interval = (
            self.conducted_interval_seconds()
            if SESSION_MODE == "signal_lab" and self.conducted_session_active else
            self.vector_autonomous_interval_seconds()
            if SESSION_MODE == "vector" and self.vector_autonomous_active else
            self.proactive_interval_seconds()
        )
        if not interval or self.current_state != "READY" or self.recording:
            return
        now = time.monotonic()
        if SESSION_MODE == "vector" and self.vector_autonomous_active and self.vector_autonomous_held:
            return
        duration_due = False
        if SESSION_MODE == "signal_lab" and self.conducted_session_active:
            if self.conducted_session_held:
                return
            activity_at = self.last_conducted_decision_at
            duration_due = bool(self.conducted_session_ends_at and now >= self.conducted_session_ends_at)
        else:
            activity_at = (
                self.last_conversation_activity_at
                if SESSION_MODE == "signal_lab"
                else self.last_meaningful_vector_change_at
            )
        brief_due = any(m.get("action") and self._brief_control_authorized(m["action"]["name"])
                        for m in self.conductor.due(now))
        if now < self.proactive_snooze_until and not brief_due and not duration_due:
            return
        if not brief_due and not duration_due and now - activity_at < interval:
            return
        if (not brief_due and not self.conducted_session_active and self.last_proactive_proposal_at
                and now - self.last_proactive_proposal_at < interval):
            return
        with self.proactive_lock:
            if self.proactive_inflight:
                return
            self.proactive_inflight = True
        target = self._run_signal_lab_presence if SESSION_MODE == "signal_lab" else self._run_proactive_proposal
        thread_name = "SignalLabPresence" if SESSION_MODE == "signal_lab" else "ProactiveDirector"
        threading.Thread(target=target, daemon=True, name=thread_name).start()

    def _run_signal_lab_presence(self) -> None:
        """Restore conversational presence without turning every idle beat into a control proposal."""
        self.director_thread.generation = self.current_voice_generation()
        try:
            if self.current_state != "READY":
                return
            if self._run_due_brief_control():
                return
            if self.conducted_session_active:
                self._run_conducted_signal_decision()
                return
            try:
                snapshot = self.signal_lab_get("/v1/state")
                live_state = (snapshot or {}).get("live") or {}
                monitor = live_state.get("headphone_monitor") or {}
                if not live_state.get("playing") or not monitor.get("audible"):
                    log("SIGNAL LAB PRESENCE suppressed because no audible live signal is playing")
                    return
                live = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
            except Exception as exc:
                log(f"SIGNAL LAB PRESENCE state unavailable: {exc}")
                return
            self.set_state("THINKING")
            beat, narrative = self.narrative_context(proactive=True)
            if beat == "quiet":
                # The timer has already followed a substantial silence. A second
                # silent interval feels like absence, so make it an observation.
                beat = "observe"
            system = (
                self.system_prompt
                + "\n\nLIVE SIGNAL LAB STATE NOW:\n" + live
                + "\n\n" + narrative
                + self._recent_language_instruction()
            )
            autonomous = bool(getattr(self, "vector_autonomous_active", False) and not getattr(self, "vector_autonomous_held", False))
            prompt = (
                "SIGNAL LAB PRESENCE MOMENT. the user has been quiet for approximately the configured idle interval. "
                "Restore the feeling of an attentive companion by speaking naturally from the ongoing conversation and the supplied live two-lane state. "
                "Prefer an observation, callback, imaginative continuation, gentle tease, or an open conversational question. "
                "Do not merely announce that you are still here and do not ask generic wellbeing questions. "
                "Normally use two to four expressive sentences; this is conversation, not a terse status report. "
                "Most presence moments must leave the signal unchanged. Only when the selected narrative beat is explicitly 'propose' may you propose ONE named Signal Lab dimension and direction. "
                "A proposal must use the form 'I propose ... Shall I do that?' and must remain unexecuted until the user approves it. "
                "Never claim that you changed, adjusted, started, or stopped the signal. Never mention Vector, internal JSON, tools, timers, or diagnostic state."
            )
            messages = [{"role":"system", "content":system}] + self.history[-8:] + [{"role":"user", "content":prompt}]
            msg = self.ollama_chat(messages, None, phase="signal-presence")
            reply = self._sanitize_model_text(msg.get("content") or "")
            if not reply:
                return
            proposal_captured = self._capture_signal_lab_proposal(reply) if beat == "propose" else False
            if not proposal_captured and self._contains_unexecuted_action_claim(reply):
                pieces = re.split(r"(?<=[.!?])\s+|\n+", reply)
                reply = " ".join(
                    piece.strip() for piece in pieces
                    if piece.strip() and not self._contains_unexecuted_action_claim(piece)
                ).strip()
                log("SIGNAL LAB PRESENCE removed unsupported execution language")
            reply = self._finalize_reply(reply, max_chars=650)
            if not reply:
                return
            now = time.monotonic()
            self.last_proactive_proposal_at = now
            self.last_conversation_activity_at = now
            self.proactive_snooze_until = now + (self.proactive_interval_seconds() or 300.0)
            self.history.append({"role":"assistant", "content":reply})
            self.trim_history()
            self._record_narrative_beat(beat)
            log(f"SIGNAL LAB PRESENCE issued beat={beat!r} proposal={proposal_captured}")
            reply = self._publish_reply(reply)
            self.enqueue_voice(reply, Timings(), voice_id=self.next_voice_tx("PRESENCE"))
        except Exception as e:
            log(f"SIGNAL LAB PRESENCE failed: {e}")
            log(traceback.format_exc())
        finally:
            if self.current_state == "THINKING" and not self._autonomous_interrupted():
                self.set_state("READY")
            self.director_thread.generation = None
            with self.proactive_lock:
                self.proactive_inflight = False

    def _run_conducted_signal_decision(self) -> None:
        """Choose, validate and narrate one autonomous Signal Lab state."""
        now = time.monotonic()
        if self.conducted_session_held or not self.conducted_session_active:
            return
        if now < self.conducted_user_priority_until:
            log("CONDUCTED SESSION decision deferred during user-priority window")
            return
        user_revision = self.signal_user_revision
        if self.conducted_session_ends_at and now >= self.conducted_session_ends_at:
            self.conducted_session_active = False
            self.conducted_session_held = False
            self.conducted_session_completed_at = now
            self.reset_action_ledger()
            result = self.execute_tool("signal_lab_neutral", {"reason": "Conducted Session preset duration complete"})
            self.conducted_session_neutral_confirmed = bool(result.get("ok"))
            elapsed_seconds = max(0.0, now - self.conductor.started_at)
            overrun_seconds = max(0.0, now - self.conducted_session_ends_at)
            self._journal_event("conducted_session_completed", {
                "decision_count": self.conducted_decision_count,
                "neutral_confirmed": bool(result.get("ok")),
                "clock_source": "session_conductor_started_at",
                "configured_duration_seconds": round(self.conducted_session_duration_minutes * 60.0, 3),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "deadline_overrun_seconds": round(overrun_seconds, 3),
            })
            reply = (
                "The preset duration is complete. I’ve brought Signal Lab to neutral and ended the Conducted Session."
                if result.get("ok") else
                "The preset duration is complete, but Signal Lab did not confirm neutral; please stop it manually."
            )
            reply = self._publish_reply(reply)
            self.enqueue_voice(reply, Timings(), voice_id=self.next_voice_tx("CONDUCT-END"))
            log(f"CONDUCTED SESSION duration complete clock=session_conductor_started_at "
                f"elapsed={elapsed_seconds:.3f}s overrun={overrun_seconds:.3f}s neutral_confirmed={bool(result.get('ok'))}")
            return
        snapshot = self.signal_lab_get("/v1/state", timeout=1.5)
        live = (snapshot or {}).get("live") or {}
        monitor = live.get("headphone_monitor") or {}
        if not live.get("playing") or not monitor.get("audible"):
            self.conducted_session_held = True
            self._journal_event("conducted_session_auto_held", {
                "reason": "live_signal_not_playing_or_monitor_not_audible",
                "playing": bool(live.get("playing")),
                "monitor": monitor,
            })
            reply = "I’ve automatically held Conducted Session because there is no confirmed audible live signal."
            reply = self._publish_reply(reply)
            self.enqueue_voice(reply, Timings(), voice_id=self.next_voice_tx("CONDUCT-HOLD"))
            log("CONDUCTED SESSION auto-held because monitor is not audible")
            return
        self.set_state("THINKING")
        beat, narrative = self.narrative_context(proactive=True)
        elapsed = max(0.0, now - self.conductor.started_at)
        total = max(1.0, self.conducted_session_ends_at - self.conductor.started_at)
        progress = min(1.0, elapsed / total)
        if progress >= 0.80 and not self.conducted_final_wave_announced:
            self.conducted_final_wave_announced = True
            self.last_conducted_decision_at = now
            announcement = "This is the final wave. I’ll shape this last cycle deliberately, then move into the closing sequence."
            self._journal_event("conducted_final_wave_announced", {
                "session_progress": round(progress, 4),
                "announcement": announcement,
            })
            announcement = self._publish_reply(announcement)
            self.enqueue_voice(announcement, Timings(), voice_id=self.next_voice_tx("FINAL-WAVE"))
            log(f"CONDUCTED SESSION final wave announced progress={progress:.3f}")
            return
        design_prompt = (
            "CONDUCTED SIGNAL LAB DECISION. Return ONLY one JSON object with left and right objects containing "
            "intensity, texture, vibration_rate, vibration_depth, secondary_rate, secondary_depth, and modulation, "
            "all numeric 0-100, plus transition_seconds and a short reason. You may optionally include heart_tempo with "
            "target 'left' or 'right' and mode 'escalation', 'double', 'relief', or 'off'. If heart_tempo is present it is the ONE principal "
            "change: preserve the authored lane design and use only one lane so the push-pull asymmetry remains. Escalation is "
            "1:1 with live BPM; double is HR ×2 and deliberately more escalatory; relief is 90 percent. Never alter intensity as part of heart tempo. Include an enabled heart_tempo "
            "choice only when AUTHORITATIVE SIGNAL LAB STATE says heart_tempo.fresh=true; otherwise omit it. "
            "You have autonomous authority inside the active preset. "
            "Use the chosen narrative arc, current progress, recent conversation and exact live two-lane state. Respond to the user’s feedback: "
            "positive feedback normally supports holding or developing the successful quality; discomfort or reduction language must reduce the relevant dimension. "
            "Make a coherent perceptual decision, not random independent numbers. Preserve useful asymmetry. Prefer one principal perceptual change per decision; "
            "Electron will stage character changes before intensity. During the final ten percent, move toward a calmer ending. "
            f"The conducted session was anchored at narrative stage {self.conducted_narrative_anchor!r}; do not replay opening or baseline-establishment logic. "
            f"Session progress={progress:.3f}; decision_number={self.conducted_decision_count + 1}."
        )
        messages = [
            {"role": "system", "content": self.system_prompt + "\n\n" + narrative},
            {"role": "system", "content": "AUTHORITATIVE SIGNAL LAB STATE:\n" + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))},
        ] + self.history[-8:] + [{"role": "user", "content": design_prompt}]
        designed = self.ollama_chat(messages, None, phase="signal-json-primary")
        if user_revision != self.signal_user_revision or self._autonomous_interrupted():
            log("CONDUCTED SESSION discarded generated decision because the user spoke while it was being prepared")
            return
        designed_text = str(designed.get("content") or "").strip()
        args = extract_signal_lab_args(designed_text)
        if not signal_lab_apply_args_complete(args):
            log(f"CONDUCTED SESSION malformed primary decision; requesting one repair preview={designed_text[:240]!r}")
            repair_messages = messages + [
                {"role": "assistant", "content": designed_text},
                {"role": "user", "content":
                    "REPAIR ONCE. Your previous response failed the required Signal Lab schema. Return ONLY one complete JSON object "
                    "with both full lane objects, numeric transition_seconds, and a short reason. No markdown or prose."},
            ]
            repaired = self.ollama_chat(repair_messages, None, phase="signal-json-repair")
            if user_revision != self.signal_user_revision or self._autonomous_interrupted():
                log("CONDUCTED SESSION discarded repaired decision because the user spoke while it was being prepared")
                return
            repaired_text = str(repaired.get("content") or "").strip()
            args = extract_signal_lab_args(repaired_text)
            if not signal_lab_apply_args_complete(args):
                self.last_conducted_decision_at = now
                self._journal_event("conducted_decision_rejected", {
                    "reason": "malformed_generated_state_after_single_retry",
                    "primary_output_preview": designed_text[:240],
                    "repair_output_preview": repaired_text[:240],
                    "narrative_beat": beat,
                    "session_progress": round(progress, 4),
                })
                log(f"CONDUCTED SESSION malformed repair rejected preview={repaired_text[:240]!r}")
                return
            self._journal_event("conducted_decision_repaired", {
                "primary_output_preview": designed_text[:240],
                "session_progress": round(progress, 4),
            })
            log("CONDUCTED SESSION single schema repair accepted")
        self.reset_action_ledger()
        if user_revision != self.signal_user_revision or time.monotonic() < self.conducted_user_priority_until:
            log("CONDUCTED SESSION cancelled before apply because a newer user turn has priority")
            return
        heart_choice = args.get("heart_tempo") if isinstance(args.get("heart_tempo"), dict) else None
        heart_fresh = bool(((snapshot or {}).get("heart_tempo") or {}).get("fresh"))
        if (heart_choice and str(heart_choice.get("target") or "").lower() in {"left", "right"}
                and str(heart_choice.get("mode") or "").lower() in {"escalation", "double", "relief", "off"}
                and (str(heart_choice.get("mode") or "").lower() == "off" or heart_fresh)):
            heart_args = {
                "target": str(heart_choice["target"]).lower(),
                "mode": str(heart_choice["mode"]).lower(),
                "reason": str(args.get("reason") or "Autonomous bounded heart-tempo decision"),
            }
            result = self.execute_tool("signal_lab_heart_tempo", heart_args)
            applied_name = "signal_lab_heart_tempo"
            applied_command = heart_args
        else:
            if heart_choice and str(heart_choice.get("mode") or "").lower() != "off" and not heart_fresh:
                self.last_conducted_decision_at = now
                self._journal_event("conducted_decision_rejected", {
                    "reason": "heart_tempo_requested_without_fresh_telemetry",
                    "proposed_command": heart_choice,
                    "narrative_beat": beat,
                    "session_progress": round(progress, 4),
                })
                log("CONDUCTED SESSION rejected heart-tempo choice without fresh telemetry")
                return
            args = bound_signal_lab_step(args, snapshot)
            result = self.execute_tool("signal_lab_apply", args)
            applied_name = "signal_lab_apply"
            applied_command = args
        self.last_conducted_decision_at = now
        action_verified = self._verified_control(applied_name, applied_command, result)
        if not result.get("ok") or not action_verified:
            self._journal_event("conducted_decision_rejected", {
                "reason": "signal_lab_validation_failed" if not result.get("ok") else "signal_lab_readback_unverified",
                "error": result.get("error"),
                "proposed_command": applied_command,
                "narrative_beat": beat,
                "session_progress": round(progress, 4),
            })
            log(f"CONDUCTED SESSION decision rejected verified={action_verified} error={result.get('error')!r}")
            return
        self.conducted_decision_count += 1
        reason = str(args.get("reason") or "a deliberate change").strip()
        recent_feedback = [
            item.get("data", {}).get("content", "")
            for item in self._journal_copy()
            if item.get("event") == "user"
        ][-8:]
        self._journal_event("conducted_decision_applied", {
            "decision_number": self.conducted_decision_count,
            "session_progress": round(progress, 4),
            "narrative_arc": self.narrative_arc_template,
            "narrative_stage": self._narrative_arc_stage(),
            "narrative_beat": beat,
            "reason": reason,
            "feedback_context": recent_feedback,
            "before_state": snapshot,
            "action_name": applied_name,
            "semantic_command": applied_command,
            "validated_result": result,
        })
        narration_prompt = (
            "Speak as the session Director in two to four expressive sentences. You have just autonomously applied the validated "
            f"Signal Lab decision described as: {reason}. Relate it naturally to the selected narrative beat and recent feedback. "
            "Do not ask permission, mention JSON, numeric control values, tools, safety systems, or claim what the user physically feels."
        )
        try:
            audible_after = self.signal_lab_get("/v1/state", timeout=1.5)
        except Exception:
            audible_after = result.get("verification_state") or {}
        narration_messages = [
            {"role": "system", "content": self.system_prompt + "\n\n" + narrative},
            {"role": "system", "content":
                "AUDIBLE LIVE STATE AFTER THE VERIFIED CHANGE:\n" +
                json.dumps(audible_after, ensure_ascii=False, separators=(",", ":")) +
                "\nDescribe only this live state. Presets are bounds, not current output; the requested command is not evidence of what is audible."},
            *self.history[-6:],
            {"role": "user", "content": narration_prompt},
        ]
        narrated = self.ollama_chat(narration_messages, None, phase="signal-presence")
        reply = self._finalize_reply(self._sanitize_model_text(narrated.get("content") or ""), fallback="", max_chars=650)
        if not reply:
            reply = self._grounded_confirmation()
        self.last_conversation_activity_at = now
        self.history.append({"role": "assistant", "content": reply})
        self.trim_history()
        self._record_narrative_beat(beat)
        log(f"CONDUCTED SESSION applied decision={self.conducted_decision_count} beat={beat!r} reason={reason!r}")
        reply = self._publish_reply(reply)
        self.enqueue_voice(reply, Timings(), voice_id=self.next_voice_tx("CONDUCT"))

    def _run_proactive_proposal(self) -> None:
        self.director_thread.generation = self.current_voice_generation()
        try:
            if self.current_state != "READY":
                return
            if self._run_due_brief_control():
                return
            live = self.live_context()
            state = self.last_live_state
            if str(state.get("engine_state", "")).lower() in ("stopped", "neutral"):
                return
            timeline = state.get("timeline") or {}
            if isinstance(timeline, dict) and timeline.get("loaded") and not timeline.get("synced"):
                # Do not use an unaligned script as justification for a proposal.
                return
            self.set_state("THINKING")
            beat, narrative = self.narrative_context(proactive=True)
            autonomous = bool(getattr(self, "vector_autonomous_active", False)
                              and not getattr(self, "vector_autonomous_held", False))
            if autonomous and self.vector_autonomous_generates_motion() and self.vector_autonomous_decision_count % 2 == 0:
                before = dict(self.last_live_state)
                result = self._design_and_apply_generated_motion(before, narrative)
                self.last_proactive_proposal_at = time.monotonic()
                self.proactive_snooze_until = self.last_proactive_proposal_at + self.vector_autonomous_interval_seconds()
                if result.get("ok"):
                    self.vector_autonomous_decision_count += 1
                    self._journal_event("vector_autonomous_motion_applied", {
                        "decision_number": self.vector_autonomous_decision_count,
                        "narrative_arc": self.narrative_arc_template,
                        "before_state": before,
                        "validated_result": result,
                    })
                    plan_args = result.get("autonomous_plan") or {}
                    spoken = self._autonomous_change_narration(
                        "vector_generated_motion_plan", plan_args,
                        str(plan_args.get("reason") or "a deliberate development of the current motion"), narrative)
                    self.history.append({"role": "assistant", "content": spoken})
                    self.trim_history()
                    spoken = self._publish_reply(spoken)
                    self.enqueue_voice(spoken, Timings(), voice_id=self.next_voice_tx("VECTOR-MOTION"))
                else:
                    log(f"VECTOR AUTONOMOUS generated-motion plan rejected: {result.get('error')}")
                if self.current_state == "THINKING" and not self._autonomous_interrupted():
                    self.set_state("READY")
                return
            if beat == "quiet":
                self.last_proactive_proposal_at = time.monotonic()
                self.proactive_snooze_until = self.last_proactive_proposal_at + (self.proactive_interval_seconds() or 300.0)
                self._record_narrative_beat(beat)
                log("NARRATIVE proactive quiet beat; remaining silent")
                if self.current_state == "THINKING" and not self._autonomous_interrupted():
                    self.set_state("READY")
                return
            system = (self.system_prompt + "\n\nLIVE VECTOR STATE NOW:\n" + live +
                      "\n\n" + narrative + self._recent_language_instruction())
            prompt = (
                "PROACTIVE DIRECTOR MOMENT. About the configured interval has passed since the last meaningful Vector change. "
                "This is also a narrative beat: you may make a brief observation, tease, callback, foreshadow, deliberately remain quiet, or propose ONE specific bounded Vector change. Use the selected narrative beat and authoritative timeline when available. "
                "Do not force a Vector proposal simply because the timer fired. "
                "If a strong or interesting authored section is about to arrive, you may deliberately defer and briefly say why instead of proposing a change. "
                "If the next section is steady or repetitive, prefer a contrasting but sensible option such as texture, variation, Moving Focus/Depth Spread, top/bottom focus, a targeted-stroke preset, or a temporary tempo window. For 2x tempo, use only 10-15 seconds when current authored energy is challenging/testing; relaxing/moderate sections can support 30-60 seconds. Half speed may be used as longer relief or contrast. "
                "Never invent an upcoming rise, transition, focus shift, or dramatic turn that is not present in the supplied timeline. "
                "For a proactive spoken remark, prefer one or two expressive sentences so it arrives promptly. "
                "If proposing, use the exact form 'I propose [one exact change]. Shall I do that?' and name a deterministic dimension and direction/value (for example top intensity up, bottom intensity down, Smooth texture, or Lower Shaft Focus); never ask approval for vague 'more intensity'. "
                + ("EXPLICIT AUTONOMOUS AUTHORITY IS ACTIVE. Choose exactly ONE useful bounded Vector change now and state it as a short imperative that the deterministic parser can execute, for example 'Set texture to Smooth', 'Set variation to Lively', 'Set top focus to Lower Shaft Focus', or 'Set bottom focus to Prostate Focus'. Do not phrase it as a proposal or question. "
                   "Do not choose stop, resume, or neutral autonomously. All other recognised actions remain inside Vector’s own operator-configured bounds and transition rules. "
                   if autonomous else
                   "Do not execute any tool. Do not claim a change happened. Keep it conversational and ask for approval only when you actually propose a change.")
            )
            messages = [{"role":"system","content":system}] + self.history[-8:] + [{"role":"user","content":prompt}]
            msg = self.ollama_chat(messages, None, phase="proactive")
            reply = self._sanitize_model_text(msg.get("content") or "")
            if not reply:
                if self.current_state == "THINKING" and not self._autonomous_interrupted():
                    self.set_state("READY")
                return
            proposal_cue = has_proposal_cue(reply)
            proposal_action = direct_vector_action(reply, vector_stopped=False) if (proposal_cue or autonomous) else None
            if autonomous and proposal_action:
                forbidden = proposal_action[0] in {"vector_stop", "vector_resume", "vector_neutral"}
                if forbidden:
                    log(f"VECTOR AUTONOMOUS rejected prohibited action {proposal_action[0]} {proposal_action[1]}")
                    proposal_action = None
            if autonomous and proposal_action:
                self.reset_action_ledger()
                before = dict(self.last_live_state)
                result = self.execute_tool(proposal_action[0], proposal_action[1])
                self.last_proactive_proposal_at = time.monotonic()
                self.proactive_snooze_until = self.last_proactive_proposal_at + self.vector_autonomous_interval_seconds()
                if result.get("ok") and not result.get("noop"):
                    self.vector_autonomous_decision_count += 1
                    self._journal_event("vector_autonomous_decision_applied", {
                        "decision_number": self.vector_autonomous_decision_count,
                        "narrative_arc": self.narrative_arc_template,
                        "reason": reply,
                        "before_state": before,
                        "action": {"name": proposal_action[0], "args": proposal_action[1]},
                        "validated_result": result,
                    })
                    spoken = self._autonomous_change_narration(
                        proposal_action[0], proposal_action[1], reply, narrative)
                    self.history.append({"role":"assistant", "content":spoken})
                    self.trim_history()
                    spoken = self._publish_reply(spoken)
                    self.enqueue_voice(spoken, Timings(), voice_id=self.next_voice_tx("VECTOR-AUTO"))
                    log(f"VECTOR AUTONOMOUS decision applied {proposal_action[0]} {proposal_action[1]}")
                else:
                    self._journal_event("vector_autonomous_decision_rejected", {
                        "reason": reply,
                        "action": {"name": proposal_action[0], "args": proposal_action[1]},
                        "result": result,
                    })
                if self.current_state == "THINKING" and not self._autonomous_interrupted():
                    self.set_state("READY")
                return
            if autonomous:
                log("VECTOR AUTONOMOUS produced no safe deterministic action; current state held")
                self.last_proactive_proposal_at = time.monotonic()
                self.proactive_snooze_until = self.last_proactive_proposal_at + self.vector_autonomous_interval_seconds()
                if self.current_state == "THINKING" and not self._autonomous_interrupted():
                    self.set_state("READY")
                return
            if proposal_cue and not proposal_action:
                log("PROACTIVE suppressed: spoken proposal had no deterministic executable action")
                self.last_proactive_proposal_at = time.monotonic()
                self.proactive_snooze_until = self.last_proactive_proposal_at + (self.proactive_interval_seconds() or 300.0)
                if self.current_state == "THINKING" and not self._autonomous_interrupted():
                    self.set_state("READY")
                return
            if proposal_action:
                action_label = f"{proposal_action[0]}:{json.dumps(proposal_action[1], sort_keys=True)}"
                if action_label in self.recent_proactive_actions[-4:] or self._action_is_noop(*proposal_action):
                    log(f"PROACTIVE suppressed repeated/no-op action {action_label}")
                    self.last_proactive_proposal_at = time.monotonic()
                    self.proactive_snooze_until = self.last_proactive_proposal_at + (self.proactive_interval_seconds() or 300.0)
                    if self.current_state == "THINKING" and not self._autonomous_interrupted():
                        self.set_state("READY")
                    return
                self.recent_proactive_actions.append(action_label)
                self.recent_proactive_actions = self.recent_proactive_actions[-8:]
            reply = self._finalize_reply(reply, "", max_chars=450)
            if not reply:
                log("PROACTIVE suppressed by repetition guard")
                self.last_proactive_proposal_at = time.monotonic()
                if self.current_state == "THINKING" and not self._autonomous_interrupted():
                    self.set_state("READY")
                return
            self.last_proactive_proposal_at = time.monotonic()
            self.proactive_snooze_until = self.last_proactive_proposal_at + (self.proactive_interval_seconds() or 300.0)
            self.history.append({"role":"assistant","content":reply})
            self.trim_history()
            # Keep exactly one spoken proposal in conversational state so a later
            # approval can be tied to one action and cannot accidentally be replayed.
            if proposal_cue:
                self.pending_proposal_text = reply
                self.pending_proposal_action = proposal_action
                self.pending_proposal_at = time.monotonic()
                if self.pending_proposal_action:
                    log(f"PROACTIVE captured action {self.pending_proposal_action[0]} {self.pending_proposal_action[1]}")
            else:
                self.pending_proposal_text = ""
                self.pending_proposal_action = None
                self.pending_proposal_at = 0.0
            self._record_narrative_beat(beat)
            log("PROACTIVE proposal/defer issued")
            reply = self._publish_reply(reply)
            self.enqueue_voice(reply, Timings(), voice_id=self.next_voice_tx("PROACTIVE"))
        except Exception as e:
            log(f"PROACTIVE failed: {e}")
            log(traceback.format_exc())
            if self.current_state == "THINKING" and not self._autonomous_interrupted():
                self.set_state("READY")
        finally:
            self.director_thread.generation = None
            with self.proactive_lock:
                self.proactive_inflight = False

    def trim_history(self) -> None:
        max_messages = int(self.cfg.get("conversation_turns", 20)) * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    @staticmethod
    def _sanitize_model_text(text: str) -> str:
        """Remove hidden-reasoning leakage and malformed model artifacts."""
        value = (text or "").strip()
        if "</think>" in value.lower():
            value = re.split(r"</think>", value, flags=re.I)[-1]
        value = re.sub(r"<think>.*?</think>", " ", value, flags=re.I | re.S)
        value = re.sub(r"</?think>", " ", value, flags=re.I)
        value = re.sub(r"\n{3,}", "\n\n", value).strip()
        paragraphs, seen = [], set()
        for paragraph in re.split(r"\n\s*\n", value):
            normalized = re.sub(r"\W+", " ", paragraph.lower()).strip()
            if normalized and normalized not in seen:
                paragraphs.append(paragraph.strip())
                seen.add(normalized)
        return "\n\n".join(paragraphs).strip()

    def _filter_spoken_signal_telemetry(self, text: str) -> str:
        """Keep technical lane dumps out of Chatterbox unless explicitly requested."""
        value = str(text or "").strip()
        if SESSION_MODE != "signal_lab" or getattr(
            self.director_thread, "allow_raw_signal_telemetry", False
        ):
            return value
        parameter_pattern = re.compile(
            r"\b(?:intensity|texture|vibration[_ ]rate|vibration[_ ]depth|secondary[_ ]rate|"
            r"secondary[_ ]depth|modulation|carrier|am[_ ]?(?:rate|depth)|fm[_ ]?(?:rate|depth))\b",
            re.I,
        )
        if len(parameter_pattern.findall(value)) < 2 and value.count(" / ") < 2:
            return value
        pieces = re.split(r"(?<=[.!?])\s+|\n+", value)
        natural = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            technical_count = len(parameter_pattern.findall(piece))
            if technical_count >= 2 or piece.count(" / ") >= 2:
                continue
            natural.append(piece)
        filtered = " ".join(natural).strip()
        if not filtered:
            executed = (self.last_action_ledger or {}).get("executed") or []
            if executed and str(executed[-1].get("name") or "").startswith("signal_lab_"):
                filtered = self._grounded_confirmation()
            else:
                filtered = "The signal remains steady. I’ll describe its character rather than reciting the controls."
        log(f"SIGNAL LAB spoken telemetry filtered chars={len(value)}->{len(filtered)}")
        return filtered

    def _recent_language_instruction(self) -> str:
        recent = [self._sanitize_model_text(item)[:260]
                  for item in self.recent_spoken_replies[-5:] if item]
        if not recent:
            return ""
        return (
            "\n\nANTI-REPETITION MEMORY: Do not reuse the openings, sentence shapes, "
            "metaphors, proposal category, or closing cadence in these recent replies:\n- "
            + "\n- ".join(recent)
            + "\nAvoid stock scaffolding such as 'Ah, the user', 'But I must say', "
              "'I can feel it already', repeated tide/silk imagery, and 'For now' when recently used."
        )

    def _profile_fallback(self) -> str:
        fallbacks = {
            "gwendolyn": "I’m here, keeping a close eye on the pattern while we hold steady.",
            "anna": "We’ll hold the current state and continue when there is something meaningful to add.",
            "natasha": "Steady. I’m watching, and I won’t fill the silence merely to hear myself speak.",
            "sophie": "Let us give this moment room; I am still here and paying attention.",
            "vesper": "Hold steady. Silence can do useful work too.",
            "aurelia": "We can remain here for a moment. I’m watching what develops.",
        }
        return fallbacks.get(self.director_voice_key(), fallbacks["gwendolyn"])

    def _scenario_fallback(self, connection_error: bool = False) -> str:
        """Stay in the known session frame without inventing state or sensation."""
        if SESSION_MODE == "signal_lab":
            prefix = "The local model connection stumbled, but " if connection_error else ""
            return prefix + "I’m still with you. The current two-lane signal remains unchanged while I recover the thread."
        engine = str((self.last_live_state or {}).get("engine_state") or "").strip().lower()
        if engine == "stopped":
            return "I’m still here. Vector remains stopped while I recover the thread."
        prefix = "The local model connection stumbled, but " if connection_error else ""
        return prefix + "I’m still with you. I’m holding to the last confirmed Vector state while I recover the thread."

    def _positive_feedback_fallback(self) -> str:
        """A grounded alternative when otherwise-good praise narration is rejected as repetitive."""
        if SESSION_MODE == "signal_lab":
            if self.conducted_session_active:
                return "Good. I’ve recorded that the current signal is working well, and I’ll preserve its successful character as the conducted session develops."
            return "Good. I’ve recorded that the current Signal Lab state is working well, so it remains unchanged."
        return "Good. I’ve recorded that the current Vector state is working well, so it remains unchanged."

    def _near_duplicate_similarity(self, text: str) -> float:
        """Return the strongest recent spoken-reply match for substantial prose."""
        normalized = self._normalized_reply(self._sanitize_model_text(text))
        if len(normalized) <= 120:
            return 0.0
        # Whole-reply matching misses a model that has learned one conspicuous
        # opening cadence and then wraps different prose around it.  Once such a
        # cadence has already appeared, treat its reuse as a duplicate so the
        # normal one-shot recovery path can request genuinely fresh language.
        cadence_patterns = (
            r"\bthat took\b",
            r"\bi(?:ll| will) let that .{0,55}\b(?:sit|settle|hang)\b",
            r"\bso i(?:ll| will) say it (?:straight|plainly)\b",
            r"\bexactly the kind of honest\b",
        )
        recent_normalized = [
            self._normalized_reply(previous)
            for previous in self.recent_spoken_replies[-8:]
        ]
        for pattern in cadence_patterns:
            if re.search(pattern, normalized) and any(
                    re.search(pattern, previous) for previous in recent_normalized):
                log(f"REPETITION GUARD recurring cadence pattern={pattern!r}")
                return 1.0
        return max(
            (SequenceMatcher(None, normalized, self._normalized_reply(previous)).ratio()
             for previous in self.recent_spoken_replies[-6:]),
            default=0.0,
        )

    def _repeated_signal_template(self, text: str) -> bool:
        """Catch recurring multi-clause 9B templates despite a novel first sentence."""
        def signature(value: str) -> Tuple[bool, ...]:
            lower = (value or "").lower()
            return tuple(token in lower for token in (
                "left lane", "right lane", "abcba", "prostate to ass to balls",
                "tell me", "does it feel like", "beginning or a continuation",
            ))
        current = signature(text)
        if sum(current) < 4:
            return False
        repeats = 0
        for previous in self.recent_spoken_replies[-8:]:
            old = signature(previous)
            if sum(a and b for a, b in zip(current, old)) >= 4:
                repeats += 1
        return repeats >= 2

    def _next_repetition_fallback(self) -> str:
        """Rotate grounded recovery lines so the repetition guard cannot create a loop."""
        if SESSION_MODE == "signal_lab":
            choices = (
                "I’m still with you. The current two-lane signal remains unchanged while I find a more useful way forward.",
                "I’m holding the present Signal Lab state. I’ll leave room here rather than repeat what has already been said.",
                "The signal remains steady and unchanged. I’m following the session without forcing another observation into this moment.",
            )
        else:
            choices = (
                "I’m still with you. The last confirmed Vector state remains unchanged while I find a more useful way forward.",
                "I’m holding the current Vector state. I’ll leave room here rather than repeat what has already been said.",
                "Vector remains steady and unchanged. I’m following the session without forcing another observation into this moment.",
            )
        index = int(getattr(self, "repetition_fallback_index", 0))
        self.repetition_fallback_index = index + 1
        return choices[index % len(choices)]

    def _retry_near_duplicate(self, reply: str, base_messages: List[Dict[str, Any]]) -> str:
        """Request one materially different reply, retaining normal grounding gates."""
        similarity = self._near_duplicate_similarity(reply)
        if similarity < 0.78:
            return reply
        log(f"REPETITION GUARD retrying near-duplicate similarity={similarity:.3f}")
        retry_messages = list(base_messages)
        retry_messages.append({
            "role": "system",
            "content": (
                "REPETITION RECOVERY: The previous draft repeated language or ideas used recently. "
                "Reply once with a short, materially different response in character. Change the subject, imagery, sentence structure, and closing move. "
                "Do not restate both lane assignments, ABCBA, a cage/anchor/promise metaphor, or end with another either/or 'Tell me' question. Add only something "
                "useful to the present exchange; it is acceptable to acknowledge briefly and leave space. "
                "Do not use tools, hidden reasoning, XML tags, or claim that an action was executed.\n\n"
                f"REJECTED DRAFT:\n{reply}"
            ),
        })
        try:
            message = self.ollama_chat(retry_messages, None, phase="noop-reroll")
            candidate = self._sanitize_model_text(message.get("content") or "")
        except requests.exceptions.RequestException as exc:
            log(f"REPETITION GUARD retry failed: {type(exc).__name__}: {exc}")
            return ""
        if (not candidate or re.search(r"[\u3400-\u9fff]", candidate)
                or self._contains_unexecuted_action_claim(candidate)):
            log("REPETITION GUARD retry rejected malformed or ungrounded reply")
            return ""
        candidate = self._enforce_authoritative_energy(candidate)
        retry_similarity = self._near_duplicate_similarity(candidate)
        if retry_similarity >= 0.78:
            log(f"REPETITION GUARD retry remained near-duplicate similarity={retry_similarity:.3f}")
            return ""
        log("REPETITION GUARD retry produced distinct user-visible content")
        return candidate

    def _finalize_reply(self, text: str, fallback: Optional[str] = None,
                        max_chars: Optional[int] = None) -> str:
        fallback = self._profile_fallback() if fallback is None else fallback
        cleaned = self._sanitize_model_text(text)
        cleaned = self._filter_spoken_signal_telemetry(cleaned)
        if not cleaned or re.search(r"[\u3400-\u9fff]", cleaned):
            log("REPETITION GUARD rejected malformed or mixed-script reply")
            return fallback
        limit = int(max_chars or self.cfg.get("max_spoken_reply_chars", 850))
        if len(cleaned) > limit:
            boundary = max(cleaned.rfind(mark, 0, limit + 1) for mark in (". ", "! ", "? "))
            cleaned = cleaned[:boundary + 1].strip() if boundary >= max(80, limit // 2) else cleaned[:limit].rsplit(" ", 1)[0].strip() + "…"
            log(f"REPLY LENGTH limited to chars={len(cleaned)}")
        # Some reasoning models can spend their generation allowance before the
        # visible answer is complete.  Never hand TTS a dangling final clause
        # when the model has already supplied at least one complete sentence.
        if len(cleaned) >= 160 and not re.search(r'[.!?…][\"\'’”)]*$', cleaned):
            boundaries = [cleaned.rfind(mark) for mark in (". ", "! ", "? ")]
            boundary = max(boundaries)
            if boundary >= 80 and len(cleaned) - boundary > 12:
                cleaned = cleaned[:boundary + 1].strip()
                log(f"REPLY COMPLETION removed incomplete trailing fragment chars={len(cleaned)}")
        ratio = self._near_duplicate_similarity(cleaned)
        if ratio >= 0.78:
            log(f"REPETITION GUARD rejected near-duplicate similarity={ratio:.3f}")
            return fallback
        self.recent_spoken_replies.append(cleaned)
        self.recent_spoken_replies = self.recent_spoken_replies[-12:]
        return cleaned

    def _enforce_grounded_language(self, text: str) -> str:
        """Remove unsupported sensation claims and invented imminent transitions."""
        value = text or ""
        replacements = (
            (r"\bI want you to feel that shift(?: fully)?\b", "I’ll let that change settle before deciding what comes next"),
            (r"\bCan you feel (?:that|the) shift\??", "Tell me what you notice, if anything."),
            (r"\bI can sense (?:it|that)(?: already)?\b", "I’m watching the live state"),
        )
        for pattern, replacement in replacements:
            value = re.sub(pattern, replacement, value, flags=re.I)
        timeline = self.last_live_state.get("timeline") or {}
        has_forecast = isinstance(timeline, dict) and any(
            timeline.get(key) not in (None, "", {}, [])
            for key in ("next", "next_event", "upcoming", "seconds_to_next", "time_to_next")
        )
        if not has_forecast and re.search(r"\b(?:the script|it) (?:is|'s) about to\b|\bin (?:just )?a few seconds\b", value, re.I):
            log("TIMELINE GROUNDING replaced unsupported imminent-transition forecast")
            return self._profile_fallback()
        return value

    def _enforce_authoritative_energy(self, text: str) -> str:
        """Prevent creative direction from contradicting the current script band."""
        timeline = self.last_live_state.get("timeline") or {}
        now = (timeline.get("now") or {}) if isinstance(timeline, dict) else {}
        energy = str(now.get("energy_band") or "").strip().lower()
        if energy != "relaxing":
            return text
        contradiction = re.search(
            r"\b(?:tighter|faster|rising|ramping|surging|spiking|cresting|peaking|"
            r"challenging|demanding|intensifying|intensity (?:is )?build(?:s|ing)?)\b",
            text or "",
            re.I,
        )
        if not contradiction:
            return text
        log(f"TIMELINE GROUNDING replaced reply contradicting authoritative relaxing band: {contradiction.group(0)!r}")
        return (
            "The script has eased into a genuine lull, and I’m not going to pretend otherwise. "
            "Let the softer rhythm settle around you while I watch for the next real change."
        )

    def _ollama_seconds(self, data: Dict[str, Any], key: str) -> float:
        try:
            return float(data.get(key) or 0) / 1_000_000_000.0
        except Exception:
            return 0.0

    def _llm_backend(self) -> str:
        backend = str(self.cfg.get("llm_backend", "ollama") or "ollama").strip().lower()
        if backend not in {"ollama", "openai"}:
            raise ValueError(f"Unsupported llm_backend: {backend!r}")
        return backend

    def _llm_model(self) -> str:
        if self._llm_backend() == "openai":
            return str(self.cfg.get("openai_model") or "qwen3-30b-a3b")
        return str(self.cfg["ollama_model"])

    def _llm_health_url(self) -> str:
        if self._llm_backend() == "openai":
            return str(self.cfg.get("openai_base") or "http://127.0.0.1:8091/v1").rstrip("/") + "/models"
        return str(self.cfg["ollama_base"]).rstrip("/") + "/api/tags"

    def _llm_chat_url(self) -> str:
        if self._llm_backend() == "openai":
            return str(self.cfg.get("openai_base") or "http://127.0.0.1:8091/v1").rstrip("/") + "/chat/completions"
        return str(self.cfg["ollama_base"]).rstrip("/") + "/api/chat"

    def _llm_headers(self) -> Dict[str, str]:
        if self._llm_backend() != "openai":
            return {}
        api_key = str(self.cfg.get("openai_api_key") or "").strip()
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _llm_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._llm_backend() == "openai":
            choices = data.get("choices") or []
            if not choices or not isinstance(choices[0], dict):
                return {}
            return choices[0].get("message") or {}
        return data.get("message") or {}

    def warm_ollama(self) -> None:
        """Load Qwen early and keep it resident before GPU-heavy TTS starts.

        v0.11 showed a sharp latency cliff after about five idle minutes, matching
        Ollama's normal model-idle lifecycle.  Keeping Qwen resident also avoids a
        late reload after Chatterbox has already occupied GPU memory.
        """
        backend = self._llm_backend()
        model = self._llm_model()
        keep_alive = self.cfg.get("ollama_keep_alive", "60m")
        timeout = max(10.0, float(self.cfg.get("ollama_startup_timeout_sec", 120)))
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "stream": False,
        }
        if backend == "openai":
            payload.update({
                "temperature": 0.0,
                "max_tokens": 2,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
                "cache_prompt": True,
            })
        else:
            payload.update({
                "think": False,
                "keep_alive": keep_alive,
                "options": {"temperature": 0.0, "num_predict": 2},
            })
        label = "llama-server" if backend == "openai" else "Ollama"
        self.emit("log", f"Warming {label} model {model}…")
        t0 = time.perf_counter()
        try:
            post_kwargs: Dict[str, Any] = {"json": payload, "timeout": timeout}
            if backend == "openai":
                post_kwargs["headers"] = self._llm_headers()
            r = self.ollama_session.post(self._llm_chat_url(), **post_kwargs)
            r.raise_for_status()
            data = r.json()
            wall = time.perf_counter() - t0
            load = self._ollama_seconds(data, "load_duration")
            log(f"LLM WARM backend={backend} wall={wall:.3f}s load={load:.3f}s")
            self.emit("log", f"{label} ready ({wall:.1f}s warm-up).")
            self.ollama_ready.set()
        except Exception as e:
            log(f"LLM WARM backend={backend} failed: {e}")
            self.emit("log", f"{label} warm-up failed; normal requests will still be attempted: {e}")
            # Do not permanently block voice startup if Ollama is temporarily unavailable.
            self.ollama_ready.set()

    def _reset_ollama_session(self) -> None:
        """Discard any pooled connection associated with a wedged Ollama request."""
        with self.ollama_session_lock:
            try:
                self.ollama_session.close()
            except Exception:
                pass
            self.ollama_session = requests.Session()
        log("OLLAMA HTTP session reset after failed request")

    def _ollama_health_after_failure(self) -> None:
        """Best-effort lightweight probe; diagnostic only and never blocks for long."""
        try:
            with self.ollama_session_lock:
                r = self.ollama_session.get(
                    self._llm_health_url(), headers=self._llm_headers(), timeout=(0.8, 1.2)
                )
            r.raise_for_status()
            log(f"OLLAMA HEALTH after failure: responsive status={r.status_code}")
        except Exception as e:
            log(f"OLLAMA HEALTH after failure: unavailable ({type(e).__name__}: {e})")

    @staticmethod
    def _prepare_openai_messages(messages: List[Dict[str, Any]], strict_alternation: bool) -> List[Dict[str, Any]]:
        """Fold system prompts and optionally satisfy strict Mistral-style chat templates."""
        system_parts = [
            str(item.get("content") or "").strip()
            for item in messages
            if item.get("role") == "system" and str(item.get("content") or "").strip()
        ]
        non_system = [dict(item) for item in messages if item.get("role") != "system"]
        if strict_alternation and all(item.get("role") in {"user", "assistant"} for item in non_system):
            alternating: List[Dict[str, Any]] = []
            for item in non_system:
                role = str(item.get("role") or "")
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                # Signal Lab publishes an assistant opening before the user's first
                # turn. Strict Mistral templates require the first conversational
                # role to be user, so retain that opening as prior context.
                if role == "assistant" and not alternating:
                    system_parts.append("PRIOR ASSISTANT OPENING (context only):\n" + content)
                    continue
                if alternating and alternating[-1].get("role") == role:
                    alternating[-1]["content"] = str(alternating[-1].get("content") or "") + "\n\n" + content
                else:
                    alternating.append({"role": role, "content": content})
            non_system = alternating
        return ([{"role": "system", "content": "\n\n".join(system_parts)}] if system_parts else []) + non_system

    def ollama_chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, phase: str = "turn") -> Dict[str, Any]:
        backend = self._llm_backend()
        keep_alive = self.cfg.get("ollama_keep_alive", "60m")
        max_predict = {
            "proactive": 150,
            "noop-reroll": 170,
            "grounding-rewrite": 220,
            "post-tool": 220,
            "signal-json-primary": 420,
            "signal-json-repair": 420,
            "signal-presence": 240,
            "vector-autonomous-narration": 240,
        }.get(phase, 480)
        configured_cap = int(self.cfg.get("llm_max_reply_tokens", max_predict))
        if phase in {"signal-json-primary", "signal-json-repair"}:
            # A complete two-lane command is longer than an ordinary short reply.
            # Do not let the conversational reply cap truncate valid JSON halfway
            # through the right lane.
            configured_cap = max(configured_cap, 420)
        max_predict = max(48, min(max_predict, configured_cap))
        temperature = 0.45 if phase == "grounding-rewrite" else 0.9
        outbound_messages = messages
        if backend == "openai":
            # Qwen3.8's native Jinja template accepts system content only in the
            # first message. Gwendolyn deliberately adds later grounding/system
            # instructions, so fold them into one leading system message while
            # retaining all user, assistant and tool messages in their order.
            outbound_messages = self._prepare_openai_messages(messages, strict_alternation=not bool(tools))
        payload: Dict[str, Any] = {
            "model": self._llm_model(),
            "messages": outbound_messages,
            "stream": False,
        }
        if backend == "openai":
            payload.update({
                "temperature": temperature,
                "max_tokens": max_predict,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
                "cache_prompt": True,
            })
        else:
            payload.update({
                "think": False,
                "keep_alive": keep_alive,
                "options": {"temperature": temperature, "num_predict": max_predict},
            })
        if tools:
            payload["tools"] = tools

        # Live voice needs bounded latency. A request that has not produced a response
        # within this window is more useful as a recoverable hiccup than a minute-long freeze.
        read_timeout = max(5.0, float(self.cfg.get("ollama_timeout_sec", 12)))
        connect_timeout = max(0.5, float(self.cfg.get("ollama_connect_timeout_sec", 2.0)))
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        tool_count = len(tools or [])
        tool_schema_chars = len(json.dumps(tools, ensure_ascii=False, separators=(",", ":"))) if tools else 0
        approx_tokens = max(1, (prompt_chars + tool_schema_chars) // 4)
        log(
            f"OLLAMA REQUEST phase={phase!r} messages={len(messages)} tools={tool_count} "
            f"prompt_chars={prompt_chars} tool_schema_chars={tool_schema_chars} "
            f"approx_tokens={approx_tokens} timeout={read_timeout:.1f}s"
        )
        t0 = time.perf_counter()
        try:
            with self.ollama_session_lock:
                if backend == "openai":
                    r = self.ollama_session.post(
                        self._llm_chat_url(), json=payload, headers=self._llm_headers(),
                        timeout=(connect_timeout, read_timeout),
                    )
                else:
                    r = self.ollama_session.post(
                        self._llm_chat_url(), json=payload,
                        timeout=(connect_timeout, read_timeout),
                    )
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.Timeout as e:
            wall = time.perf_counter() - t0
            log(f"OLLAMA TIMEOUT phase={phase!r} after={wall:.3f}s type={type(e).__name__}")
            self._reset_ollama_session()
            self._ollama_health_after_failure()
            raise
        except requests.exceptions.RequestException as e:
            wall = time.perf_counter() - t0
            log(f"OLLAMA REQUEST FAILED phase={phase!r} after={wall:.3f}s type={type(e).__name__}: {e}")
            self._reset_ollama_session()
            self._ollama_health_after_failure()
            raise

        wall = time.perf_counter()-t0
        load = self._ollama_seconds(data, "load_duration")
        prompt_s = self._ollama_seconds(data, "prompt_eval_duration")
        eval_s = self._ollama_seconds(data, "eval_duration")
        pcount = int(data.get("prompt_eval_count") or 0)
        ecount = int(data.get("eval_count") or 0)
        log(
            f"LLM backend={backend} wall={wall:.3f}s phase={phase!r} tools={bool(tools)} load={load:.3f}s "
            f"prompt={prompt_s:.3f}s/{pcount}tok eval={eval_s:.3f}s/{ecount}tok "
            f"messages={len(messages)} keep_alive={keep_alive}"
        )
        return self._llm_message(data)

    @staticmethod
    def _looks_like_approval(text: str) -> bool:
        return bool(re.search(
            r"\b(?:yes|yeah|yep|absolutely|do it|please do|please proceed|proceed|go ahead|let(?:'|’)s do (?:it|that)|that (?:sounds|seems) like a (?:good|great|nice|lovely|wonderful|excellent|perfect) idea|that would be (?:good|great|nice|lovely|wonderful)|sounds (?:really |very )?(?:good|great|fabulous|lovely|nice|wonderful|excellent|perfect|tempting|delicious)|i(?:'|’)m ready)\b",
            text or "", re.I))

    @staticmethod
    def _looks_like_positive_feedback(text: str) -> bool:
        return bool(re.search(
            r"\b(?:it|that|this) (?:feels|is feeling|felt|sounds) (?:really |very |so )?"
            r"(?:good|great|nice|lovely|wonderful|excellent|perfect|delicious|amazing)\b|"
            r"\bi (?:like|love|enjoy)(?:d|ing)? (?:it|that|this)\b",
            text or "", re.I))

    @staticmethod
    def _looks_like_finish_request(text: str) -> bool:
        t = text or ""
        return bool(
            re.search(r"\b(?:thank you|thanks)\b.{0,50}\b(?:for (?:the|this|our) session|for today|gwendolyn)\b", t, re.I)
            or re.search(r"\b(?:finish|end|close|wrap up)\s+(?:the|this|our)?\s*session\b", t, re.I)
            or re.search(r"\b(?:that(?:'|’)s|that is) (?:enough|all) for (?:today|tonight)\b", t, re.I)
            or re.search(r"\b(?:i(?:'|’)ll|i will) see you (?:later|next time|tomorrow)\b", t, re.I)
            or re.search(r"\b(?:goodbye|good night|goodnight)\b", t, re.I)
            or re.search(r"\bi(?:'|’)m (?:going|off) to (?:bed|sleep)\b", t, re.I)
            or re.search(r"\b(?:finish|done|stop) for (?:today|tonight|now)\b", t, re.I)
        )

    @staticmethod
    def _looks_like_rejection(text: str) -> bool:
        return bool(re.search(r"\b(?:no|nope|not yet|keep going|continue|carry on|don(?:'|’)t|do not)\b", text or "", re.I))

    def _emit_local_turn(self, user_text: str, reply: str, timings: Timings, voice_prefix: str = "TURN") -> None:
        """Record and speak a deterministic conversational turn without calling the LLM."""
        timings.llm_done = time.perf_counter()
        self.history.append({"role":"user", "content":user_text})
        self.history.append({"role":"assistant", "content":reply})
        self.trim_history()
        reply = self._publish_reply(reply)
        self.enqueue_voice(reply, timings, voice_id=self.next_voice_tx(voice_prefix))

    def _handle_hermes_export(self, text: str, timings: Timings, reply_prefix: str = "",
                              user_text_override: str = "") -> bool:
        """Create an explicit, local-only snapshot for later read-only Hermes review."""
        if not is_hermes_export_command(text):
            return False
        self.set_state("THINKING")
        try:
            controller_state: Dict[str, Any]
            if SESSION_MODE == "signal_lab":
                controller_state = self.signal_lab_get("/v1/state", timeout=1.5)
            else:
                try:
                    controller_state = self._refresh_authoritative_state(timeout=1.5)
                except Exception:
                    controller_state = dict(self.last_live_state)
            created = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            journal = self._journal_copy()
            session_actions = [item for item in journal if item.get("event") in
                               {"control_action_executed", "control_action_failed"}]
            snapshot = {
                "schema": "gwendolyn-hermes-handoff/v2",
                "created_local": created,
                "session_id": self.memory_session_id,
                "session_mode": SESSION_MODE,
                "director": self.director_name(),
                "purpose": "Offline analysis only; this snapshot grants no live-control authority.",
                "session_summary": {
                    "journal_started_local": self.session_journal_started_local,
                    "exported_local": created,
                    "elapsed_seconds": round(max(0.0, time.monotonic() - self.session_journal_started_monotonic), 3),
                    "narrative_arc": self.narrative_arc_template,
                    "conducted_session_active_at_export": self.conducted_session_active,
                    "conducted_session_held_at_export": self.conducted_session_held,
                    "conducted_decision_count": self.conducted_decision_count,
                    "control_action_count": len(session_actions),
                },
                "controller_state": controller_state,
                "session_journal": self._journal_copy(),
                "session_actions": session_actions,
                "session_brief_schedule": self.conductor.snapshot(),
                "recent_conversation": [
                    {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
                    for item in self.history[-12:] if isinstance(item, dict)
                ],
                "recent_actions": {
                    "executed": [
                        {"name": item.get("name"), "args": item.get("args") or {}}
                        for item in self.last_action_ledger.get("executed", [])
                    ],
                    "failed": [
                        {"name": item.get("name"), "args": item.get("args") or {}, "error": item.get("error")}
                        for item in self.last_action_ledger.get("failed", [])
                    ],
                },
            }
            HERMES_HANDOFF_INBOX.mkdir(parents=True, exist_ok=True)
            destination = HERMES_HANDOFF_INBOX / f"{stamp}-{SESSION_MODE}-{self.memory_session_id}.json"
            suffix = 1
            while destination.exists():
                destination = HERMES_HANDOFF_INBOX / f"{stamp}-{SESSION_MODE}-{self.memory_session_id}-{suffix}.json"
                suffix += 1
            temporary = destination.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
            temporary.replace(destination)
            log(f"HERMES HANDOFF exported {destination}")
            reply = (
                f"I’ve prepared a read-only Hermes handoff named {destination.name}. "
                "It contains the complete structured session journal and current controller state; Hermes has not been launched."
            )
            if not reply_prefix:
                reply += " Nothing live was changed."
        except Exception as exc:
            log(f"HERMES HANDOFF export failed: {type(exc).__name__}: {exc}")
            reply = "I couldn’t prepare the Hermes handoff, so no snapshot was created."
            if not reply_prefix:
                reply += " Nothing live was changed."
        if reply_prefix:
            reply = f"{reply_prefix} {reply}"
        self._emit_local_turn(user_text_override or text, reply, timings, "HANDOFF")
        return True

    def _handle_global_signal_lab_shutdown(self, text: str, timings: Timings) -> bool:
        """Stop Signal Lab deterministically, then export if the same request asks for Hermes."""
        if SESSION_MODE != "signal_lab" or not is_signal_lab_shutdown_command(text):
            return False
        self.reset_action_ledger()
        self.set_state("THINKING")
        self.conducted_session_active = False
        self.conducted_session_held = False
        self.conducted_start_pending = False
        result = self.execute_tool("signal_lab_neutral", {"reason": "Explicit Signal Lab shutdown"})
        stopped = bool(result.get("ok")) if isinstance(result, dict) else False
        export_requested = bool(
            is_hermes_export_command(text)
            or re.search(r"\b(?:send|write|create|prepare)\b.{0,35}\breport\b", text or "", re.I)
        )
        self._journal_event("explicit_signal_lab_shutdown", {
            "stop_confirmed": stopped,
            "hermes_export_requested": export_requested,
        })
        log(f"GLOBAL SIGNAL LAB SHUTDOWN confirmed={stopped}")
        if stopped:
            prefix = "Signal Lab is neutral and its session output is stopped."
        else:
            prefix = "Signal Lab did not confirm that it stopped; please use its manual stop control now."
        if export_requested:
            return self._handle_hermes_export(text, timings, reply_prefix=prefix)
        self._emit_local_turn(text, prefix, timings, "SIGNAL-STOP")
        return True

    def _handle_global_vector_shutdown(self, text: str, timings: Timings) -> bool:
        """Honour an explicit Vector shutdown from either fixed session UI."""
        action = direct_vector_action(text, vector_stopped=False)
        if not action or action[0] != "vector_stop":
            return False
        self.reset_action_ledger()
        self.set_state("THINKING")
        result = self.execute_tool("vector_stop", {})
        stopped = bool(result.get("ok")) if isinstance(result, dict) else False
        if getattr(self, "vector_autonomous_active", False):
            self.vector_autonomous_active = False
            self.vector_autonomous_held = False
            self._journal_event("vector_autonomous_session_disarmed", {
                "decision_count": self.vector_autonomous_decision_count,
                "completion_reason": "explicit_vector_shutdown",
            })
        self._journal_event("explicit_vector_shutdown", {
            "requested_from_session_mode": SESSION_MODE,
            "stop_confirmed": stopped,
        })
        reply = "Vector is stopped." if stopped else "Vector did not confirm that it stopped. Please use Vector’s manual stop control now."
        log(f"GLOBAL VECTOR SHUTDOWN mode={SESSION_MODE} confirmed={stopped}")
        self._emit_local_turn(text, reply, timings, "VECTOR-STOP")
        return True

    def _handle_session_finish(self, text: str, timings: Timings) -> bool:
        """Run the explicit two-stage close using the controller fixed for this session."""
        signal_mode = SESSION_MODE == "signal_lab"
        controller = "Signal Lab" if signal_mode else "Vector"
        if self.session_finish_pending:
            if re.search(r"\b(?:don(?:'|’)t finish|do not finish|keep going|not yet|continue the session)\b", text or "", re.I):
                self.session_finish_pending = False
                self.session_finish_stage = ""
                self.session_closure_reflection = ""
                log("SESSION FINISH cancelled by user")
                self._emit_local_turn(text, "Of course. We’ll keep the session running.", timings, "CLOSE")
                return True
            if self.session_finish_stage == "awaiting_reflection":
                self.session_closure_reflection = (text or "").strip()
                self.session_finish_stage = "awaiting_summary_receipt"
                actions = [item for item in self._journal_copy()
                           if item.get("event") == "control_action_executed"]
                elapsed_minutes = max(0.0, time.monotonic() - self.session_journal_started_monotonic) / 60.0
                summary = (
                    f"Your closing reflection is recorded: {self.session_closure_reflection} "
                    f"This session ran for about {elapsed_minutes:.0f} minutes and recorded {len(actions)} control actions. "
                    "That is the session summary. Please confirm that you received it; only then will I neutralise and close."
                )
                self._journal_event("session_closure_reflection", {"content": self.session_closure_reflection})
                self._journal_event("session_closure_summary_spoken", {
                    "summary": summary, "control_action_count": len(actions),
                    "elapsed_minutes": round(elapsed_minutes, 2),
                })
                self._emit_local_turn(text, summary, timings, "CLOSE-SUMMARY")
                return True
            if self.session_finish_stage == "awaiting_summary_receipt" and self._looks_like_approval(text):
                self.session_finish_pending = False
                self.session_finish_stage = ""
                self.pending_proposal_text = ""
                self.pending_proposal_action = None
                self.pending_proposal_at = 0.0
                self.reset_action_ledger()
                self.set_state("THINKING")
                if signal_mode and self.conducted_session_active:
                    self.conducted_session_active = False
                    self.conducted_session_held = False
                    self._journal_event("conducted_session_completed", {
                        "decision_count": self.conducted_decision_count,
                        "completion_reason": "user_confirmed_session_finish",
                        "neutral_confirmed": None,
                    })
                    log("CONDUCTED SESSION disarmed before confirmed session finish")
                if not signal_mode and getattr(self, "vector_autonomous_active", False):
                    self.vector_autonomous_active = False
                    self.vector_autonomous_held = False
                    self._journal_event("vector_autonomous_session_disarmed", {
                        "decision_count": self.vector_autonomous_decision_count,
                        "completion_reason": "user_confirmed_session_finish",
                    })
                    log("VECTOR AUTONOMOUS disarmed before confirmed session finish")
                action_started = time.perf_counter()
                tool_name = "signal_lab_neutral" if signal_mode else "vector_stop"
                tool_args = {"reason": "Confirmed end of session"} if signal_mode else {}
                result = self.execute_tool(tool_name, tool_args)
                timings.action_seconds += time.perf_counter() - action_started
                stopped = bool(result.get("ok")) if isinstance(result, dict) else False
                if signal_mode:
                    self._journal_event("session_finish_result", {
                        "neutral_confirmed": stopped,
                        "conducted_decision_count": self.conducted_decision_count,
                    })
                if stopped:
                    if signal_mode:
                        reply = "Signal Lab is neutral and its session output is stopped. Thank you for sharing the session with me. Goodbye for now—and until next time."
                    else:
                        reply = "Vector is stopped. Thank you for sharing the session with me. Goodbye for now—and until next time."
                    log(f"SESSION FINISH confirmed; {controller} stopped before farewell")
                    self._journal_event("session_closure_summary_received", {
                        "confirmation": text,
                        "reflection": self.session_closure_reflection,
                    })
                    # Closure order is now deterministic: reflection -> summary -> receipt
                    # -> neutral/stop confirmation -> read-only export.
                    return self._handle_hermes_export(
                        "export Hermes handoff", timings, reply_prefix=reply,
                        user_text_override=text,
                    )
                else:
                    reply = f"{controller} did not confirm that it stopped, so I won’t say goodbye just yet. Please stop it manually, and then we can finish properly."
                    log(f"SESSION FINISH {controller} stop failed; farewell withheld")
                self._emit_local_turn(text, reply, timings, "CLOSE")
                return True
            reply = ("Please confirm that you received the spoken session summary. "
                     f"After that I’ll stop {controller} and prepare the read-only export.")
            self._emit_local_turn(text, reply, timings, "CLOSE")
            return True

        if not self._looks_like_finish_request(text):
            return False
        self.session_finish_pending = True
        self.session_finish_stage = "awaiting_reflection"
        self.session_closure_reflection = ""
        self.pending_proposal_text = ""
        self.pending_proposal_action = None
        self.pending_proposal_at = 0.0
        log("SESSION FINISH requested; awaiting closing reflection")
        reply = "Before I close, give me one line of reflection about this session. I’ll record it, speak the summary separately, and wait for your confirmation before stopping anything."
        self._emit_local_turn(text, reply, timings, "CLOSE")
        return True

    @staticmethod
    def _normalized_reply(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()

    @staticmethod
    def _contains_unexecuted_action_claim(text: str) -> bool:
        t = (text or "").lower()
        patterns = [
            r"\b(?:i(?:'|’)ll|i will|i shall(?: now)?|let me|i(?:'|’)m going to|i am going to)\b.{0,100}\b(?:change|adjust|tighten|narrow|widen|increase|decrease|raise|lower|shift|set|switch|slow|speed|double|halve|reduce|restore|amplify|intensify|deepen|heighten|elevate|dial back)\b",
            r"\b(?:done|completed)\b.{0,100}\b(?:changed|adjusted|tightened|narrowed|widened|increased|decreased|raised|lowered|shifted|set|switched|slowed|sped|doubled|halved|reduced|restored|amplified)\b",
            r"\bi(?:'|’)ve\b.{0,100}\b(?:changed|adjusted|tightened|narrowed|widened|increased|decreased|raised|lowered|shifted|set|switched|slowed|sped|doubled|halved|reduced|restored|amplified)\b",
            r"\b(?:vector is|the vector is)\b.{0,60}\b(?:now|no longer)\b",
            r"\b(?:i(?:'|’)m|i am) (?:shifting|changing|moving|setting|switching|raising|lowering|increasing|decreasing)\b",
            r"\bi have (?:adjusted|changed|shifted|moved|set|switched|raised|lowered|increased|decreased)\b",
            r"\bthe change is (?:immediate|already|now|complete)\b",
            r"\b(?:the|that) shift (?:away from|from|to)\b.{0,100}\b(?:glans|shaft|root|prostate|anal|perineum|focus)\b",
            r"\bit(?:'|’)s working exactly as intended\b.{0,120}\b(?:shaft|glans|root|prostate|focus|texture|variation)\b",
            r"\b(?:left|right|stairway|moaner|lane|channel|signal|rhythm|pulse)\b.{0,100}\b(?:locked|matched|synced|synchroni[sz]ed|glued)\b.{0,80}\b(?:heart|heartbeat|heart rate|pulse)\b",
            r"\b(?:locked|matched|synced|synchroni[sz]ed|glued)\b.{0,80}\b(?:to|with)\b.{0,30}\b(?:heart|heartbeat|heart rate|pulse)\b",
            r"\b(?:both|the) (?:lanes?|electrodes?|signals?) (?:are|is) (?:now )?(?:at|on|down|up|neutral|stopped|lower|higher|gentler|softer|milder|stronger|weaker)\b",
            r"\b(?:the )?(?:signal|output) (?:is|has been) (?:now )?(?:neutral|stopped|lowered|reduced|raised|increased|gentler|softer|stronger|weaker)\b",
        ]
        return any(re.search(p, t, re.I | re.S) for p in patterns)

    @staticmethod
    def _contains_unsupported_biometric_claim(text: str) -> bool:
        """Reject causal physiological claims that the local telemetry cannot establish."""
        t = text or ""
        patterns = [
            r"\b(?:signal|lane|channel|carrier|pulse|vibration|frequency)\b.{0,120}\b(?:cause|causes|causing|make|makes|making|push|pushes|pushing|force|forces|forcing|drive|drives|driving|raise|raises|raising|increase|increases|increasing)\b.{0,45}\b(?:your )?heart rate\b",
            r"\b(?:your )?heart rate\b.{0,80}\b(?:catch up|rise|rises|rising|increase|increases|increasing|race|racing)\b.{0,80}\b(?:signal|lane|channel|carrier|pulse|vibration|frequency|on its own)\b",
            r"\b(?:tightening|fullness|pressure)\b.{0,55}\b(?:chest|ribs)\b.{0,100}\b(?:signal|lane|channel|carrier|pulse|vibration|frequency)\b",
        ]
        return any(re.search(pattern, t, re.I | re.S) for pattern in patterns)

    @staticmethod
    def _contains_heart_lock_claim(text: str) -> bool:
        return bool(re.search(
            r"\b(?:heartbeat|heart rate|pulse)\s*(?:lock|match|sync)|"
            r"\b(?:locked|matched|synced|synchroni[sz]ed|riding|following)\b.{0,45}\b(?:heartbeat|heart rate|your pulse)\b|"
            r"\b(?:lane|channel|signal|rhythm|pulse)\b.{0,65}\b(?:in time with|following|riding)\b.{0,35}\b(?:heartbeat|heart rate|your pulse)\b",
            text or "", re.I | re.S,
        ))

    @staticmethod
    def _heart_lock_verified_in_state(state: Dict[str, Any]) -> bool:
        control = (state or {}).get("heart_tempo_control") or {}
        heart = (state or {}).get("heart_tempo") or {}
        if not control.get("active") or not heart.get("fresh"):
            return False
        target = str(control.get("target") or "")
        ratio = float(control.get("ratio") or 0.0)
        signal = ((state or {}).get("live") or {}).get(f"{target}_signal") or {}
        actual = signal.get("amFreq")
        expected = float(heart.get("bpm") or 0.0) * ratio / 60.0
        return target in {"left", "right"} and isinstance(actual, (int, float)) and abs(float(actual) - expected) <= 0.02

    def _delegated_surprise_action(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Choose one bounded, non-noop semantic change when the user explicitly delegates choice.

        This is deliberately conservative: no gain increases and no tempo acceleration. The user
        asked Gwendolyn to choose, so the choice is mechanical and grounded rather than left to a
        tool-less prose turn.
        """
        if not re.search(r"\b(?:surprise me|make a change and surprise me|choose for me|your choice|pick something|do something interesting|mix it up)\b", text or "", re.I):
            return None
        try:
            self._refresh_authoritative_state(timeout=1.0)
        except Exception:
            pass
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for value in ("Smooth", "Normal", "Rough"):
            candidates.append(("vector_set_texture", {"texture": value}))
        for value in ("Subtle", "Normal", "Lively"):
            candidates.append(("vector_set_variation", {"variation": value}))
        for value in ("Glans Focus", "Shaft Focus", "Lower Shaft Focus", "Root Focus", "Top Full"):
            candidates.append(("vector_set_top_focus", {"top_focus": value}))
        for value in ("Prostate Focus", "Anal Focus", "Perineum Focus", "Bottom Full"):
            candidates.append(("vector_set_bottom_focus", {"bottom_focus": value}))
        recent = set(self.intervention_recent_actions[-6:])
        eligible = []
        for action in candidates:
            name, args = action
            label = f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True)}"
            if label in recent or self._action_is_noop(name, args):
                continue
            eligible.append(action)
        if not eligible:
            return None
        chosen = self.narrative_rng.choice(eligible)
        log(f"DELEGATED surprise action selected {chosen[0]} {chosen[1]}")
        return chosen

    def _rewrite_unsupported_execution(self, draft: str, base_messages: List[Dict[str, Any]], pending_action: Optional[Tuple[str, Dict[str, Any]]] = None) -> str:
        """Hide grounding plumbing while preserving the useful narrative around a blocked claim."""
        if pending_action:
            proposal = self._safe_proposal_phrase(pending_action)
            instruction = (
                "IMMERSION GROUNDING REWRITE: The draft below contains unsupported execution language. "
                "No Vector change happened. Preserve the tone, imagery, and conversational content, but remove only the false claim that a mechanical change already happened or is happening. "
                f"End naturally with this real pending proposal, paraphrased if needed: {proposal!r}. "
                "Do not mention Vector, tools, grounding, bookkeeping, safety rules, or that you rewrote anything."
            )
        else:
            instruction = (
                "IMMERSION GROUNDING REWRITE: The draft below contains unsupported execution language. "
                "No mechanical change happened. Preserve as much of the tone, imagery, continuity, and non-mechanical content as possible. "
                "Rewrite only the unsupported mechanical claim into observation, possibility, imagination, or ordinary conversation. "
                "Do not mention Vector, tools, grounding, bookkeeping, safety rules, or that you rewrote anything. Do not invent a new change."
            )
        msgs = list(base_messages) + [
            {"role":"system", "content": instruction},
            {"role":"assistant", "content": draft},
        ]
        try:
            msg = self.ollama_chat(msgs, None, phase="grounding-rewrite")
            raw_rewritten = msg.get("content") or ""
            rewritten = self._sanitize_model_text(raw_rewritten)
            sane_length = len(rewritten) <= max(600, int(len(draft) * 1.25))
            clean_script = not re.search(r"[\u3400-\u9fff]|</?think>", raw_rewritten, re.I)
            if rewritten and sane_length and clean_script and not self._contains_unexecuted_action_claim(rewritten):
                log("GROUNDING immersion rewrite succeeded")
                return rewritten
            log(f"GROUNDING immersion rewrite rejected sane_length={sane_length} clean_script={clean_script}")
        except Exception as e:
            log(f"GROUNDING immersion rewrite failed: {type(e).__name__}: {e}")
        if pending_action:
            return self._safe_proposal_phrase(pending_action)
        # If the rewrite model cannot answer (for example at a context boundary),
        # preserve all complete narrative sentences that do not make an unsupported
        # mechanical claim instead of collapsing the conversation to a stock line.
        pieces = re.split(r"(?<=[.!?])\s+|\n+", self._sanitize_model_text(draft))
        preserved = " ".join(
            piece.strip() for piece in pieces
            if piece.strip() and not self._contains_unexecuted_action_claim(piece)
        ).strip()
        if len(preserved) >= 40 and not self._contains_unexecuted_action_claim(preserved):
            log("GROUNDING preserved non-mechanical draft sentences after rewrite failure")
            return preserved
        return "Mm. I’ll let that idea hang in the air for a moment and see where the session takes us."

    def _safe_proposal_phrase(self, action: Optional[Tuple[str, Dict[str, Any]]]) -> str:
        if not action:
            return "How about I choose a different direction for the next move?"
        name, args = action
        if name == "vector_set_top_focus":
            return f"Mm. How about I move the top focus to {args.get('top_focus')}?"
        if name == "vector_set_bottom_focus":
            return f"How about I move the bottom focus to {args.get('bottom_focus')}?"
        if name == "vector_set_texture":
            return f"How about I make the texture {args.get('texture')}?"
        if name == "vector_set_variation":
            return f"How about I shift the variation to {args.get('variation')}?"
        if name == "vector_set_targeting":
            label = str(args.get('preset') or '').replace('_', ' ')
            return f"How about I try the {label} targeting pattern?"
        if name == "vector_top_spatial_gain":
            direction = str(args.get('action') or 'adjust')
            return f"Would you like me to {direction} the top intensity a little?"
        if name == "vector_bottom_spatial_gain":
            direction = str(args.get('action') or 'adjust')
            return f"Would you like me to {direction} the bottom intensity a little?"
        if name == "vector_adjust_stroke_range":
            action_word = str(args.get('action') or 'adjust')
            return f"Would you like me to make the stroke range {action_word}?"
        if name == "vector_tempo_window":
            return "Would you like me to try a temporary tempo change?"
        return "Would you like me to make that change?"

    def _grounded_confirmation(self, ledger: Optional[Dict[str, Any]] = None) -> str:
        ledger = self.last_action_ledger if ledger is None else ledger
        executed = ledger.get("executed", [])
        failed = ledger.get("failed", [])
        if not executed:
            if failed:
                failed_name = str(failed[-1].get("name") or "")
                if failed_name.startswith("signal_lab_"):
                    return "That Signal Lab change didn’t take, so its current state is unchanged."
                return "That change didn’t take. I’ve left Vector as it was."
            return "I haven’t changed Vector there."
        item = executed[-1]
        name = item.get("name")
        args = item.get("args") or {}
        if name == "vector_set_primary_spatial":
            label = str(args.get("primary_spatial") or "").replace("top_", "").replace("_", " ")
            return f"Done. I’ve shifted the primary spatial mode to {label}."
        if name == "vector_set_texture":
            if item.get("result", {}).get("noop"):
                return f"The texture is already {args.get('texture')}, so I’ve left it there."
            return f"Done. I’ve changed the texture to {args.get('texture')}."
        if name == "vector_set_variation":
            if item.get("result", {}).get("noop"):
                return f"The variation is already {args.get('variation')}, so I’ve left it there."
            return f"Done. I’ve changed the variation to {args.get('variation')}."
        if name == "vector_set_top_focus":
            if item.get("result", {}).get("noop"):
                return f"The top focus is already at {args.get('top_focus')}, so I’ve left it there."
            return f"Done. I’ve shifted the top focus to {args.get('top_focus')}."
        if name == "vector_set_bottom_focus":
            if item.get("result", {}).get("noop"):
                return f"The bottom focus is already at {args.get('bottom_focus')}, so I’ve left it there."
            return f"Done. I’ve shifted the bottom focus to {args.get('bottom_focus')}."
        if name == "vector_top_spatial_gain":
            return "Done. I’ve adjusted the top spatial gain."
        if name == "vector_bottom_spatial_gain":
            return "Done. I’ve adjusted the bottom spatial gain."
        if name == "vector_adjust_stroke_range":
            action = str(args.get("action") or "")
            if action == "restore":
                return "Done. I’ve restored the authored stroke range."
            if action == "narrower":
                return "Done. I’ve narrowed the stroke range a little."
            if action == "wider":
                return "Done. I’ve widened the stroke range a little."
            return "Done. I’ve adjusted the stroke range."
        if name == "vector_set_targeting":
            preset = str(args.get("preset") or "")
            if preset == "authored":
                return "Done. I’ve restored the authored stroke path."
            label = preset.replace("_", " ")
            return f"Done. I’ve applied the {label} targeting preset."
        if name == "vector_tempo_window":
            return f"Done. Tempo is at {float(args.get('scale', 1.0)):.1f}× for {int(args.get('duration_seconds', 0))} seconds, then Vector will restore the authored pace."
        if name == "vector_restore_modifiers":
            return "Done. I’ve restored the authored targeting and tempo."
        if name == "vector_stop":
            return "Vector is stopped."
        if name == "vector_resume":
            return "Vector is running again."
        if name == "vector_neutral":
            return "Vector is neutral."
        if name == "signal_lab_apply":
            left = args.get("left") or {}
            right = args.get("right") or {}
            def level(value, low, high, lower, middle, upper):
                number = float(value or 0)
                return lower if number < low else upper if number > high else middle
            li = level(left.get("intensity"), 40, 60, "restrained", "moderate", "strong")
            ri = level(right.get("intensity"), 40, 60, "restrained", "moderate", "strong")
            lt = level(left.get("texture"), 35, 65, "rougher", "balanced", "smoother")
            rt = level(right.get("texture"), 35, 65, "rougher", "balanced", "smoother")
            if li != ri or lt != rt:
                return (f"Done. I’ve set the Stairway {li} and {lt}, while the Moaner is {ri} and {rt}. "
                        "Signal Lab will introduce the character changes one at a time and bring intensity in last.")
            vibration = level((float(left.get("vibration_depth", 0)) + float(right.get("vibration_depth", 0))) / 2,
                              35, 65, "light", "defined", "deep")
            return (f"Done. Both lanes are {li} and {lt}, with a {vibration} vibration. "
                    "Signal Lab will stage the changes one parameter at a time, with intensity last.")
        if name == "signal_lab_neutral":
            return "The offline Signal Lab is neutral. No audio output is active."
        if name == "signal_lab_heart_tempo":
            mode = str(args.get("mode") or "").lower()
            if mode == "off":
                return "Done. I’ve released the heart-tempo lock and restored authored movement."
            result = item.get("result") or {}
            state = result.get("verification_state") or {}
            heart = state.get("heart_tempo") or {}
            target = str(args.get("target") or "")
            ratio = {"escalation": 1.0, "double": 2.0, "relief": 0.9}.get(mode, 1.0)
            effective = (((state.get("live") or {}).get(f"{target}_signal") or {}).get("amFreq"))
            if not isinstance(effective, (int, float)) or not isinstance(heart.get("bpm"), (int, float)):
                return (f"The {target} lane heart-tempo request was accepted, but its live rate readback is not available yet. "
                        "I won’t invent a confirmation value.")
            return (f"Done. The {target} lane is verified at {float(effective):.3f} hertz, "
                    f"following {float(heart.get('bpm')):.1f} BPM at {ratio:.0%}; the other lane remains authored.")
        if name == "signal_lab_get_state":
            result = item.get("result") or {}
            live = result.get("live") or {}
            left = live.get("left_signal") or {}
            right = live.get("right_signal") or {}
            monitor = live.get("headphone_monitor") or {}

            def number(state: Dict[str, Any], key: str, unit: str = "") -> str:
                value = state.get(key)
                if not isinstance(value, (int, float)):
                    return "unavailable"
                rendered = f"{float(value):.2f}".rstrip("0").rstrip(".")
                return f"{rendered}{unit}"

            def lane(label: str, state: Dict[str, Any]) -> str:
                if not state:
                    return f"{label} telemetry is unavailable"
                secondary = (
                    f"secondary AM {number(state, 'secondaryAmFreq', ' hertz')} at "
                    f"{number(state, 'secondaryAmDepth', ' percent')}"
                    if float(state.get("secondaryAmDepth") or 0) > 0 else
                    "no secondary AM"
                )
                fm = (
                    f"FM {number(state, 'fmFreq', ' hertz')} at {number(state, 'fmDepth', ' hertz')} depth"
                    if float(state.get("fmDepth") or 0) > 0 else
                    "no FM"
                )
                return (
                    f"{label} is {number(state, 'freq', ' hertz')}, volume {number(state, 'volume', ' percent')}, "
                    f"AM {number(state, 'amFreq', ' hertz')} at {number(state, 'amDepth', ' percent')}, "
                    f"{secondary}, and {fm}"
                )

            if not left and not right:
                return "Signal Lab is connected, but its live lane telemetry is not available yet."
            position = max(0, int(live.get("position_ms") or 0)) // 1000
            duration = max(0, int(live.get("duration_ms") or 0)) // 1000
            playback = "playing" if live.get("playing") else "paused"
            audible = bool(monitor.get("audible"))
            monitor_text = (
                f"The headphone monitor is audible at {number(monitor, 'gain_percent', ' percent')}."
                if audible else "The headphone monitor is not currently audible."
            )
            return (
                f"The live Signal Lab state is: {lane('Stairway', left)}. "
                f"{lane('Moaner', right)}. Playback is {playback} at {position} of {duration} seconds. "
                f"{monitor_text} Physical output remains disabled."
            )
        return "Done. The Vector change executed successfully."

    def _recent_signal_change_context(self) -> str:
        """Return the last validated change record for a concise comparative answer."""
        for item in reversed(self._journal_copy()):
            if item.get("event") == "conducted_decision_applied":
                data = item.get("data") or {}
                grounded = {
                    "reason": data.get("reason"),
                    "before_state": data.get("before_state"),
                    "applied_command": data.get("semantic_command"),
                    "validated_result": data.get("validated_result"),
                }
                return json.dumps(grounded, ensure_ascii=False, separators=(",", ":"))
        executed = (self.last_action_ledger or {}).get("executed") or []
        if executed and str(executed[-1].get("name") or "").startswith("signal_lab_"):
            return json.dumps(executed[-1], ensure_ascii=False, separators=(",", ":"))
        return "No recent validated Signal Lab change record is available."

    def _authoritative_signal_time_reply(self, text: str) -> str:
        if SESSION_MODE != "signal_lab" or not self.conducted_session_ends_at:
            return ""
        if not re.search(
            r"\b(?:how (?:long|much time)|time (?:left|remaining)|elapsed time|session (?:time|duration)|"
            r"(?:have we|did we|we have|we(?:'|’)ve) (?:finished|reached|completed)|"
            r"(?:finished|complete|completed).{0,30}(?:session|preset|minute|limit)|"
            r"(?:session|preset).{0,30}(?:finished|complete|completed))\b",
            str(text or ""),
            re.I,
        ):
            return ""
        configured = self.conducted_session_duration_minutes or max(
            1.0, (self.conducted_session_ends_at - self.conductor.started_at) / 60.0
        )
        if self.conducted_session_completed_at:
            neutral = (
                "Signal Lab is confirmed neutral."
                if self.conducted_session_neutral_confirmed else
                "Signal Lab did not confirm neutral, so please stop it manually."
            )
            return (
                f"Yes—the preset’s {configured:g}-minute conducted window is complete. {neutral} "
                "The journal can show a slightly longer span because it also includes setup and closing conversation."
            )
        now = time.monotonic()
        remaining = max(0.0, self.conducted_session_ends_at - now)
        elapsed = max(0.0, now - self.conductor.started_at)
        return (
            f"The authoritative preset clock is at {elapsed / 60.0:.1f} of {configured:g} minutes, "
            f"with about {remaining / 60.0:.1f} minutes remaining."
        )

    def handle_text(self, text: str, timings: Optional[Timings] = None) -> None:
        timings = timings or Timings()
        self.director_thread.allow_raw_signal_telemetry = bool(
            SESSION_MODE == "signal_lab" and asks_for_raw_signal_telemetry(text)
        )
        turn_now = time.monotonic()
        self.last_conversation_activity_at = turn_now
        if self.sensor_pending_feedback:
            self._journal_event(f"{SESSION_MODE}_sensor_feedback", {
                "user_text": text,
                "prompted_by": self.sensor_pending_feedback,
                "meaning_is_user_supplied": True,
            })
            log("RESTIM SENSOR user feedback attached to pending observation")
            self.sensor_pending_feedback = None
        if SESSION_MODE == "signal_lab":
            self.signal_user_revision += 1
            if self.conducted_session_active:
                priority_seconds = max(120.0, float(self.cfg.get("conducted_user_priority_seconds", 180.0)))
                self.conducted_user_priority_until = turn_now + priority_seconds
                self.last_conducted_decision_at = turn_now
                log(f"CONDUCTED SESSION user priority for {priority_seconds:.0f}s revision={self.signal_user_revision}")
        if self._handle_global_vector_shutdown(text, timings):
            return
        if self._handle_global_signal_lab_shutdown(text, timings):
            return
        if self._handle_hermes_export(text, timings):
            return
        if self._handle_conducted_session_command(text, timings):
            return
        authoritative_time_reply = self._authoritative_signal_time_reply(text)
        if authoritative_time_reply:
            self._journal_event("authoritative_session_time_answered", {
                "user_text": text,
                "completed": bool(self.conducted_session_completed_at),
                "neutral_confirmed": bool(self.conducted_session_neutral_confirmed),
                "clock_source": "session_conductor_started_at",
            })
            self._emit_local_turn(text, authoritative_time_reply, timings, "SESSION-TIME")
            log("SESSION TIME answered from authoritative conducted clock")
            return
        if self._handle_vector_autonomous_command(text, timings):
            return
        if self._handle_vector_heart_query(text, timings):
            return
        if self._handle_memory_command(text, timings):
            return
        if self._handle_session_finish(text, timings):
            return
        if re.search(r"\b(?:leave it|leave things|not yet|give me (?:a|a few|few) minutes|later for now)\b", text, re.I):
            interval = self.proactive_interval_seconds()
            if interval:
                self.proactive_snooze_until = time.monotonic() + interval
                log(f"PROACTIVE snoozed for {int(interval)}s by user phrase")
        approval_of_signal_proposal = bool(
            SESSION_MODE == "signal_lab"
            and self._looks_like_approval(text)
            and self.pending_signal_lab_proposal
        )
        approval_of_pending = bool(
            SESSION_MODE == "vector"
            and self._looks_like_approval(text)
            and self.pending_proposal_text
        )
        pending_before_turn = (
            self.pending_signal_lab_proposal if approval_of_signal_proposal else
            self.pending_proposal_text if approval_of_pending else ""
        )
        pending_action_before_turn = self.pending_proposal_action if approval_of_pending else None
        self.reset_action_ledger()
        try:
            self.set_state("THINKING")
            timings.llm_start = time.perf_counter()
            if SESSION_MODE == "signal_lab":
                try:
                    signal_snapshot = self.signal_lab_get("/v1/state")
                    live = json.dumps(signal_snapshot, ensure_ascii=False, separators=(",", ":"))
                except Exception as exc:
                    signal_snapshot = {"available": False, "error": str(exc)}
                    live = json.dumps(signal_snapshot, ensure_ascii=False)
                live_heading = "LIVE SIGNAL LAB STATE NOW:\n"
            else:
                live = self.live_context()
                live_heading = "LIVE VECTOR STATE NOW:\n"
            self.observe_preference_evidence(text)
            beat, narrative = self.narrative_context(proactive=False)
            system = (self.system_prompt + "\n\n" + live_heading + live +
                      "\n\n" + narrative + self._recent_language_instruction())
            if SESSION_MODE == "signal_lab" and not self.director_thread.allow_raw_signal_telemetry:
                system += (
                    "\n\nSPOKEN SIGNAL LAB PRESENTATION: Describe changes and the current signal perceptually. "
                    "Do not recite numeric lane values, slash-separated controls, parameter names, raw telemetry, or JSON. "
                    "Exact technical values remain available only when the user explicitly asks for them."
                )
            user_msg = {"role":"user","content":text}
            base_messages = [{"role":"system","content":system}] + self.history + [user_msg]

            # Fast deterministic action for clear commands; otherwise qwen chooses a tool.
            vector_stopped = str(self.last_live_state.get("engine_state", "")).lower() == "stopped"
            # Explicit Signal Lab language belongs to the offline two-lane tools.
            # Never allow legacy anatomical/intensity shortcuts (for example
            # "more on the Moaner") to turn it into a Vector spatial-gain action.
            signal_context_active = SESSION_MODE == "signal_lab"
            positive_signal_feedback = bool(
                signal_context_active
                and (
                    self._looks_like_positive_feedback(text)
                    or re.search(r"\b(?:good|great|excellent|fabulous|fantastic|lovely|really good) signal\b", text, re.I)
                    or re.search(r"\b(?:got|is|feels?) (?:much |really |very )?(?:smoother|rougher|stronger|weaker|better)\b", text, re.I)
                    or re.search(r"\b(?:increase[ds]?|enhance[ds]?|improve[ds]?) (?:the )?(?:feeling|sensation|effect)\b", text, re.I)
                    or re.search(r"\b(?:hips?|body) (?:are |is )?(?:rocking|moving|responding)\b", text, re.I)
                    or re.search(r"\b(?:you are|you(?:'|’)re) (?:so |absolutely )?right\b|\bthank you(?: again)?\b", text, re.I)
                )
                and not re.search(r"\b(?:change|adjust|create|make|set|increase|decrease|raise|lower|smooth|roughen|vary|neutral|stop)\b", text, re.I)
            )
            signal_control_cue = bool(
                approval_of_signal_proposal
                or re.search(
                    r"\b(?:please|change|adjust|create|make|design|apply|set|build|increase|decrease|raise|lower|"
                    r"smooth|roughen|vary|neutral|reset|stop|start|establish|try|use|give me|let(?:'|’)s|"
                    r"reduce|gentler|softer|milder|weaker|stronger|pull (?:it |both )?down|turn (?:it |both )?down|back off|ease|"
                    r"can you|could you|would you|shall we|how about|what(?:'s| is| are)|show|read|report|status|check)\b",
                    text,
                    re.I,
                )
            )
            signal_detail_suppression = bool(
                signal_context_active
                and re.search(r"\b(?:don(?:'|’)t|do not|stop|avoid|without|no more)\b", text, re.I)
                and re.search(r"\b(?:detail|number|numeric|parameter|telemetry|technical|readout|read out|recite|list)\w*\b", text, re.I)
            )
            explicit_signal_request = bool(
                SESSION_MODE == "signal_lab"
                and not positive_signal_feedback
                and not signal_detail_suppression
                and signal_control_cue
                and is_signal_lab_intent(f"{pending_before_turn} {text}")
            )
            heart_tempo_direct = (
                direct_signal_heart_tempo_action(f"{pending_before_turn} {text}")
                if SESSION_MODE == "signal_lab" else None
            )
            signal_relief_direct = (
                deterministic_signal_relief(signal_snapshot, text)
                if SESSION_MODE == "signal_lab" else {}
            )
            explicit_vector_request = SESSION_MODE == "vector" and bool(re.search(r"\bvector\b", text, re.I))
            sticky_signal_control = bool(
                signal_context_active
                and not positive_signal_feedback
                and not signal_detail_suppression
                and not explicit_vector_request
                and signal_control_cue
                and re.search(
                    r"\b(?:intensity|volume|texture|smooth|rough|frequency|carrier|vibration|modulation|"
                    r"left|right|other lane|both|electrode|stairway|moaner|signal|wave|stronger|weaker|gentler|softer|milder|"
                    r"increase|decrease|reduce|raise|lower|pull (?:it |both )?down|turn (?:it |both )?down|back off|ease)\w*\b",
                    text,
                    re.I,
                )
            )
            signal_request = explicit_signal_request or sticky_signal_control
            signal_state_question = bool(
                signal_context_active
                and (asks_what_changed_in_signal(text)
                     or re.search(r"\b(?:what(?:'s| is| are)|show|read|report|current|status|check)\b", text, re.I))
                and not re.search(r"\b(?:create|make|design|apply|set|build|new|adjust|increase|decrease|raise|lower)\b", text, re.I)
            )
            raw_signal_telemetry_request = bool(
                signal_context_active
                and asks_for_raw_signal_telemetry(text)
                and (signal_state_question or re.search(r"\b(?:give me|tell me|provide|list)\b", text, re.I))
                and not re.search(r"\b(?:create|make|design|apply|set|build|adjust|increase|decrease|raise|lower)\b", text, re.I)
            )
            concise_signal_summary_request = bool(
                signal_state_question and not raw_signal_telemetry_request
            )
            if positive_signal_feedback:
                log("SIGNAL LAB positive feedback holds current state; no command generated")
            if signal_detail_suppression:
                log("SIGNAL LAB presentation preference recognised; no control command generated")
            if sticky_signal_control and not explicit_signal_request:
                log("SIGNAL LAB sticky context inherited for ambiguous control request")
            direct = heart_tempo_direct or (
                ("signal_lab_apply", signal_relief_direct) if signal_relief_direct else None
            ) or (
                None if SESSION_MODE == "signal_lab" or signal_request else (
                    pending_action_before_turn
                    or self._delegated_surprise_action(text)
                    or direct_vector_action(text, vector_stopped=vector_stopped)
                )
            )
            if pending_action_before_turn:
                log(f"PROPOSAL APPROVAL direct execution {pending_action_before_turn[0]} {pending_action_before_turn[1]}")

            # Once the excursion budget is reached, an ordinary conversational turn
            # triggers one deterministic recovery before Qwen can invent another
            # excursion. Explicit user Vector commands and proposal approvals still
            # take priority over automatic recovery.
            if SESSION_MODE == "vector" and not direct and self.intervention_count >= self.intervention_target:
                if self._looks_like_positive_feedback(text):
                    positive_hold = max(0.0, float(self.cfg.get("state_arc_positive_feedback_hold_seconds", 120.0)))
                    self.recovery_snooze_until = max(self.recovery_snooze_until, time.monotonic() + positive_hold)
                    log(f"STATE ARC recovery extended by positive feedback for {positive_hold:.0f}s")
                recovery_reply = self._execute_due_recovery()
                if recovery_reply:
                    timings.llm_done = time.perf_counter()
                    self.history.append(user_msg)
                    self.history.append({"role":"assistant","content":recovery_reply})
                    self.trim_history()
                    self._record_narrative_beat("relief")
                    recovery_reply = self._publish_reply(recovery_reply)
                    self.enqueue_voice(recovery_reply, timings, voice_id=self.next_voice_tx("RECOVERY"))
                    return
            tool_calls = []
            assistant_first: Dict[str, Any] = {"role":"assistant","content":""}
            if direct:
                name, args = direct
                tool_calls = [{"type":"function","function":{"name":name,"arguments":args}}]
                assistant_first["tool_calls"] = tool_calls
                log(f"DIRECT {name} {args}")
            else:
                if concise_signal_summary_request:
                    change_context = self._recent_signal_change_context()
                    base_messages.insert(1, {"role": "system", "content":
                        "SIGNAL LAB SUMMARY REQUEST: Answer in one to three concise, natural sentences. Describe the perceptual "
                        "difference and current character. Do not dump numbers, raw lane telemetry, parameter names, JSON, or claim "
                        "a bodily effect. If no validated comparison is available, say so plainly and describe only the current character.\n"
                        "LAST VALIDATED CHANGE RECORD:\n" + change_context})
                    assistant_first = self.ollama_chat(base_messages, None, phase="first-pass")
                    tool_calls = []
                    log("OLLAMA ROUTE signal-lab-summary conversational")
                elif signal_request and re.search(r"\b(?:neutral|clear|reset)\b", text, re.I):
                    tool_calls = [{"type":"function","function":{"name":"signal_lab_neutral","arguments":{"reason":"Explicit Signal Lab neutral request"}}}]
                    assistant_first["tool_calls"] = tool_calls
                    log("OLLAMA ROUTE signal-lab-neutral direct")
                elif raw_signal_telemetry_request:
                    tool_calls = [{"type":"function","function":{"name":"signal_lab_get_state","arguments":{}}}]
                    assistant_first["tool_calls"] = tool_calls
                    log("OLLAMA ROUTE signal-lab-state direct")
                elif signal_request:
                    log("OLLAMA ROUTE signal-json-primary tools=0")
                    design_messages = [
                        {"role":"system","content":
                            "Return ONLY one JSON object, with no markdown or prose. It must have left and right objects containing intensity, texture, vibration_rate, vibration_depth, secondary_rate, secondary_depth, and modulation, all numeric 0-100; plus numeric transition_seconds from 5-15 and a short reason string. SEMANTICS: texture 0=rough/low carrier, 50=balanced, 100=smooth/high carrier; vibration_rate is primary tapping speed; vibration_depth is primary pulse strength; secondary_rate/depth add a second beating or buzzing relationship; modulation is FM movement; intensity is level, never quality. EDIT RULE: preserve every current value not explicitly requested by the user. A request to change texture must not alter intensity, vibration or modulation. Choose a new full state only when the request explicitly asks to create or surprise. The transition is performed one perceptual parameter at a time, with intensity last."},
                        {"role":"system","content":"CURRENT SIGNAL LAB STATE:\n" + live},
                        {"role":"user","content": str(
                            (pending_before_turn + "\nthe user approved this proposal: " + text)
                            if approval_of_signal_proposal else
                            (text or "Design and apply a contrasting two-lane Signal Lab state.")
                        )},
                    ]
                    designed = self.ollama_chat(design_messages, None, phase="signal-json-primary")
                    designed_text = str(designed.get("content") or "").strip()
                    designed_args = extract_signal_lab_args(designed_text)
                    if not signal_lab_apply_args_complete(designed_args):
                        log(f"SIGNAL LAB unusable primary JSON preview={designed_text[:240]!r}")
                        repair_messages = design_messages + [
                            {"role": "assistant", "content": designed_text},
                            {"role": "user", "content":
                                "REPAIR ONCE. Your previous response failed the required Signal Lab schema. Return ONLY one complete "
                                "JSON object with both full lane objects, values in their allowed ranges, numeric transition_seconds "
                                "from 5-15, and a short reason. No markdown or prose."},
                        ]
                        repaired = self.ollama_chat(repair_messages, None, phase="signal-json-repair")
                        repaired_text = str(repaired.get("content") or "").strip()
                        designed_args = extract_signal_lab_args(repaired_text)
                        if signal_lab_apply_args_complete(designed_args):
                            log("SIGNAL LAB single schema repair accepted")
                        else:
                            # A failed retry is not permission to substitute and
                            # execute a generic state. Hold the current signal instead.
                            tool_calls = []
                            assistant_first = {
                                "role": "assistant",
                                "content": "I couldn’t complete that Signal Lab design cleanly after one repair attempt, so I’ve left the current signal exactly as it is."
                            }
                            log(f"SIGNAL LAB malformed design rejected after single retry preview={repaired_text[:240]!r}; current state preserved")
                    if signal_lab_apply_args_complete(designed_args):
                        try:
                            signal_snapshot = json.loads(live)
                            designed_args = preserve_unrequested_signal_dimensions(
                                designed_args, signal_snapshot, text
                            )
                            designed_args = bound_signal_lab_step(designed_args, signal_snapshot)
                        except Exception:
                            pass
                        tool_calls = [{"type":"function","function":{"name":"signal_lab_apply","arguments":designed_args}}]
                        assistant_first = {"role":"assistant","content":"","tool_calls":tool_calls}
                    log("SIGNAL LAB primary JSON design complete" if signal_lab_apply_args_complete(designed_args) else "SIGNAL LAB primary JSON design incomplete")
                else:
                    routed_tools = select_tools_for_turn(text, pending_before_turn) if SESSION_MODE == "vector" else []
                    route_names = [item.get("function", {}).get("name") for item in routed_tools]
                    if routed_tools:
                        log(f"OLLAMA ROUTE vector-intent tools={len(routed_tools)} names={route_names}")
                    else:
                        log("OLLAMA ROUTE conversational tools=0")
                        context_rule = (
                            "SIGNAL LAB CONTEXT: Continue discussing the current Signal Lab session. Do not propose, imply, or capture any Vector change unless the user explicitly names Vector."
                            if signal_context_active else
                            "CONVERSATION-ONLY TURN: You have no Vector action tools in this turn. You may observe, discuss, tease, or PROPOSE a future change, but you MUST NOT say or imply that you are changing Vector now. Do not use execution language such as 'I'll adjust', 'let me change', 'I've shifted', or describe a new Vector state as if it happened. If you want a change, phrase it as a proposal and ask the user whether they want it."
                        )
                        base_messages.insert(1, {"role":"system","content":context_rule})
                    assistant_first = self.ollama_chat(base_messages, routed_tools or None, phase="first-pass")
                    tool_calls = assistant_first.get("tool_calls") or []

            follow_messages = list(base_messages)
            if tool_calls:
                follow_messages.append({"role":"assistant","content":assistant_first.get("content") or "", "tool_calls":tool_calls})
                for tc in tool_calls[:3]:
                    fn = tc.get("function") or {}
                    name = fn.get("name", "")
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: args = {}
                    _a0 = time.perf_counter()
                    result = self.execute_tool(name, args)
                    timings.action_seconds += time.perf_counter() - _a0
                    narration_result = self.narration_tool_result(name, args, result)
                    follow_messages.append({"role":"tool","tool_name":name,"content":json.dumps(narration_result, ensure_ascii=False)})
                # Tool recursion deliberately suppressed: after action, require prose.
                # Ground the narration in what actually executed this turn.
                follow_messages.append({
                    "role": "system",
                    "content": (
                        "GROUNDING FOR THIS REPLY — ACTION LEDGER:\n"
                        + self.compact_action_ledger()
                        + "\nDescribe only actions listed under executed as having happened. "
                          "Anything requested but not executed remains a proposal. If an action is listed under failed, say plainly that it did not happen; never narrate it as successful. "
                          "The user has already answered the preceding proposal. Do NOT repeat that proposal or ask for approval again. "
                          "Speak as Gwendolyn, not as a diagnostic report. Confirm successful changes naturally in one or two conversational sentences. "
                          "Do not claim subjective sensation or say 'feel that shift' merely because an action executed; describe the change itself or its intended effect. "
                          "Never read internal tool names, JSON, numeric strengths, weights, alpha windows, HTTP details, ledger bookkeeping, or phrases such as 'the following actions were executed successfully' unless the user explicitly asks for technical details."
                    ),
                })
                # Clear deterministic commands already tell us exactly what happened.
                # Do not pay for a second full-context Qwen pass after execution;
                # short tempo windows were previously ending before narration began.
                if direct or signal_request:
                    reply = self._grounded_confirmation()
                    if signal_request:
                        self.pending_signal_lab_proposal = ""
                    confirmation_kind = "Signal Lab" if signal_request else "direct action"
                    log(f"GROUNDING {confirmation_kind} confirmed without second Ollama pass")
                else:
                    reply_msg = self.ollama_chat(follow_messages, None, phase="post-tool")
                    reply = (reply_msg.get("content") or "").strip()
                    # A successful tool turn must never replay the proposal that triggered it.
                    if pending_before_turn and self._normalized_reply(reply) == self._normalized_reply(pending_before_turn):
                        log("GROUNDING replaced repeated proposal after tool execution")
                        reply = self._grounded_confirmation()
                if approval_of_pending:
                    self.pending_proposal_text = ""
                    self.pending_proposal_action = None
                    self.pending_proposal_at = 0.0
            else:
                reply = (assistant_first.get("content") or "").strip()
                proposal_captured = False
                self.last_proposal_rejected_noop = False
                if SESSION_MODE == "signal_lab" and not approval_of_signal_proposal:
                    proposal_captured = self._capture_signal_lab_proposal(reply)
                if not approval_of_pending and not signal_context_active:
                    proposal_captured = self._capture_conversational_proposal(reply)
                    if proposal_captured and self._proposal_mentions_multiple_actions(reply):
                        reply = self._safe_proposal_phrase(self.pending_proposal_action)
                        self.pending_proposal_text = reply
                        log("PROPOSAL stacked actions reduced to one deterministic pending action")
                if self.last_proposal_rejected_noop and not approval_of_pending:
                    # Director-generated no-ops are internal planning failures, not conversation.
                    # Silently replace them with a different valid proposal (or natural continuation).
                    rejected = self.last_rejected_noop_action
                    reply, proposal_captured = self._reroll_after_noop(base_messages, rejected, reply)
                if (not approval_of_pending and not proposal_captured and not self.last_proposal_rejected_noop
                        and not signal_context_active and self._contains_unexecuted_action_claim(reply)):
                    log("GROUNDING blocked unexecuted action claim on conversation-only turn")
                    inferred = direct_vector_action(reply, vector_stopped=False)
                    if inferred and not self._action_is_noop(inferred[0], inferred[1]):
                        self.pending_proposal_action = inferred
                        self.pending_proposal_at = time.monotonic()
                        reply = self._rewrite_unsupported_execution(reply, base_messages, inferred)
                        self.pending_proposal_text = reply
                        log(f"GROUNDING immersive proposal captured {inferred[0]} {inferred[1]}")
                    else:
                        reply = self._rewrite_unsupported_execution(reply, base_messages, None)
                if approval_of_pending:
                    # An approval that failed to resolve should remain immersive; never expose plumbing.
                    log("GROUNDING approval received but no Vector tool executed")
                    reply = self._rewrite_unsupported_execution(reply, base_messages, None)
                    self.pending_proposal_text = ""
                    self.pending_proposal_action = None
                    self.pending_proposal_at = 0.0

            # A spoken heart lock requires current controller readback, not an old
            # proposal, a failed action, or a model-authored reason string.
            if SESSION_MODE == "signal_lab" and self._contains_heart_lock_claim(reply):
                try:
                    heart_state = json.loads(live)
                except Exception:
                    heart_state = {}
                if not self._heart_lock_verified_in_state(heart_state):
                    pieces = re.split(r"(?<=[.!?])\s+|\n+", self._sanitize_model_text(reply))
                    reply = " ".join(
                        piece.strip() for piece in pieces
                        if piece.strip() and not self._contains_heart_lock_claim(piece)
                    ).strip()
                    if not reply:
                        reply = "I don’t have a verified heartbeat lock at the moment, so I’m leaving that claim out."
                    log("GROUNDING removed unverified heart-lock narration")

            if SESSION_MODE == "signal_lab" and re.search(
                    r"\bphysical (?:output|signal) (?:is )?(?:armed|enabled|active|live)\b", reply or "", re.I):
                pieces = re.split(r"(?<=[.!?])\s+|\n+", self._sanitize_model_text(reply))
                reply = " ".join(piece.strip() for piece in pieces if piece.strip() and not re.search(
                    r"\bphysical (?:output|signal) (?:is )?(?:armed|enabled|active|live)\b", piece, re.I)).strip()
                log("GROUNDING removed invented Signal Lab physical-output claim")

            # Telemetry can verify timing correlation, not a causal effect on heart rate.
            # Strip unsupported physiological predictions even when the surrounding
            # prose is otherwise conversational.
            if self._contains_unsupported_biometric_claim(reply):
                pieces = re.split(r"(?<=[.!?])\s+|\n+", self._sanitize_model_text(reply))
                reply = " ".join(
                    piece.strip() for piece in pieces
                    if piece.strip() and not self._contains_unsupported_biometric_claim(piece)
                ).strip()
                if not reply:
                    reply = (
                        "I can match one lane’s tempo to the measured heartbeat and record what changes, "
                        "but this session cannot establish that the signal changes your heart rate."
                    )
                log("GROUNDING removed unsupported biometric causation claim")

            # Final mechanical execution-language gate.  Prompt instructions are not
            # sufficient: any prose that claims a change happened must be backed by
            # a successful, non-noop action in THIS turn's ledger.
            if self._contains_unexecuted_action_claim(reply):
                executed_now = [
                    item for item in self.last_action_ledger.get("executed", [])
                    if not (item.get("result") or {}).get("noop")
                ]
                if not executed_now:
                    if self.pending_proposal_action:
                        log("GROUNDING hard ledger gate rewrote future execution language as pending proposal")
                        reply = self._rewrite_unsupported_execution(reply, base_messages, self.pending_proposal_action)
                        self.pending_proposal_text = reply
                    else:
                        log("GROUNDING hard ledger gate blocked execution claim without same-turn action")
                        reply = self._rewrite_unsupported_execution(reply, base_messages, None)

            if not reply:
                if self.last_action_ledger.get("requested"):
                    reply = self._grounded_confirmation()
                else:
                    log("LLM returned no user-visible content; retrying once with a direct conversational prompt")
                    retry_messages = list(base_messages)
                    retry_messages.append({
                        "role": "system",
                        "content": (
                            "RESPONSE RECOVERY: Your previous answer contained no user-visible text. "
                            "Reply now with one short, direct conversational sentence in character. "
                            "Do not use tools, hidden reasoning, XML tags, or action/execution claims."
                        ),
                    })
                    try:
                        retry_msg = self.ollama_chat(retry_messages, None, phase="empty-response-retry")
                        reply = self._sanitize_model_text(retry_msg.get("content") or "")
                        if reply and self._contains_unexecuted_action_claim(reply):
                            log("GROUNDING rejected execution claim from empty-response retry")
                            reply = ""
                        if reply:
                            log("LLM empty-response retry produced user-visible content")
                        else:
                            log("LLM empty-response retry also returned no user-visible content")
                    except requests.exceptions.RequestException as e:
                        log(f"LLM empty-response retry failed: {type(e).__name__}: {e}")
                    if not reply:
                        log("LLM returned no user-visible content after retry; using neutral conversational fallback")
                        reply = self._scenario_fallback()
            # Timeline replacement must pass through the same near-duplicate guard
            # as model prose; otherwise consecutive lull turns can repeat verbatim.
            reply = self._enforce_authoritative_energy(reply)
            protected_action_receipt = bool(self.last_action_ledger.get("executed") or self.last_action_ledger.get("failed"))
            repeated_draft = (not protected_action_receipt and
                              (self._near_duplicate_similarity(reply) >= 0.78 or self._repeated_signal_template(reply)))
            if repeated_draft:
                reply = self._retry_near_duplicate(reply, base_messages)
            final_fallback = (
                self._positive_feedback_fallback() if self._looks_like_positive_feedback(text) else
                self._next_repetition_fallback() if repeated_draft else None
            )
            reply = self._finalize_reply(reply, fallback=final_fallback)
            reply = self._enforce_grounded_language(reply)
            # A proposal must exist in the final user-visible reply. Previously a
            # proposal located beyond the spoken-length cutoff could remain pending,
            # causing an unrelated positive remark to be treated as approval.
            if not approval_of_pending and self.pending_proposal_action:
                visible_cue = has_proposal_cue(reply)
                visible_action = direct_vector_action(reply, vector_stopped=False) if visible_cue else None
                if visible_action != self.pending_proposal_action:
                    log("PROPOSAL cleared because action is absent from final visible reply")
                    self.pending_proposal_text = ""
                    self.pending_proposal_action = None
                    self.pending_proposal_at = 0.0
            timings.llm_done = time.perf_counter()
            self.history.append(user_msg)
            self.history.append({"role":"assistant","content":reply})
            self.trim_history()
            self._record_narrative_beat(beat)
            reply = self._publish_reply(reply)
            voice_id = self.next_voice_tx("TURN")
            self.enqueue_voice(reply, timings, voice_id=voice_id)
        except requests.exceptions.Timeout:
            # The audio/STT/Vector pipeline is still alive; fail this turn quickly and locally.
            # Do not dump a frightening traceback for a recoverable local-model stall.
            timings.llm_done = time.perf_counter()
            fallback = self._scenario_fallback()
            log("OLLAMA turn abandoned cleanly; returning to voice loop with local fallback")
            self.history.append({"role":"user","content":text})
            self.history.append({"role":"assistant","content":fallback})
            self.trim_history()
            fallback = self._publish_reply(fallback)
            voice_id = self.next_voice_tx("TURN")
            self.enqueue_voice(fallback, timings, voice_id=voice_id)
        except requests.exceptions.RequestException as e:
            timings.llm_done = time.perf_counter()
            fallback = self._scenario_fallback(connection_error=True)
            log(f"OLLAMA request error recovered without traceback: {type(e).__name__}: {e}")
            self.history.append({"role":"user","content":text})
            self.history.append({"role":"assistant","content":fallback})
            self.trim_history()
            fallback = self._publish_reply(fallback)
            voice_id = self.next_voice_tx("TURN")
            self.enqueue_voice(fallback, timings, voice_id=voice_id)
        except Exception as e:
            self.emit("error", f"Gwendolyn turn failed: {e}")
            log(traceback.format_exc())
            self.set_state("READY")

    def _fetch_tts_pcm(self, chunk_text: str, voice_id: str, idx: int,
                       sample_rate: int) -> bytes:
        with self.tts.request(chunk_text, voice_id, idx, timeout=(3, 120)) as response:
            response.raise_for_status()
            ctype = (response.headers.get("content-type") or "").lower()
            if "mpeg" in ctype or "mp3" in ctype:
                raise RuntimeError(f"TTS returned unsupported content type {ctype}")
            payload = response.content
        if payload[:4] != b"RIFF":
            return payload[:len(payload) - (len(payload) % 2)]
        with wave.open(io.BytesIO(payload), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise RuntimeError("TTS WAV must be mono 16-bit PCM")
            if wav_file.getframerate() != sample_rate:
                raise RuntimeError(
                    f"TTS WAV sample rate {wav_file.getframerate()} does not match {sample_rate}")
            return wav_file.readframes(wav_file.getnframes())

    def _speak_now(self, text: str, timings: Timings, voice_id: str, generation: int,
                   announcement_receipt: int = 0) -> None:
        """Prefetch synthesis while the preceding semantic chunk is playing."""
        spoken_text = prepare_spoken_text(text, self.effective_director_profile())
        chunk_mode = self.profile.get("speech_chunking", "Off")
        backend = self.tts.backend()
        chunks = (semantic_tts_chunks(spoken_text)
                  if backend.key == "chatterbox"
                  else split_spoken_reply(spoken_text, chunk_mode))
        if not chunks:
            return
        self.set_state("SPEAKING")
        self.stop_audio.clear()
        sample_rate = int(backend.sample_rate)
        edge_fade_ms = float(self.cfg.get("audio_edge_fade_ms", 12))
        block_bytes = max(2, int(sample_rate * int(self.cfg.get("audio_block_ms", 20)) / 1000.0) * 2)
        pause_frames = int(sample_rate * max(0.0, float(self.cfg.get("tts_chunk_pause_ms", 45))) / 1000.0)
        result_queue: queue.Queue = queue.Queue(maxsize=2)
        producer_stop = threading.Event()
        log(f"VOICE {voice_id} DISPLAY={text!r}")
        effective_profile = self.effective_director_profile()
        log(f"VOICE {voice_id} DELIVERY={chunk_mode!r} verbosity={effective_profile.get('speech_verbosity', 4)}")
        log(
            f"VOICE {voice_id} CHUNK_MODE={('Semantic prefetch 75-170' if backend.key == 'chatterbox' else chunk_mode)!r} "
            f"chunks={len(chunks)} chars={[len(chunk) for chunk in chunks]} prefetch=True "
            f"long_form_degradation_avoided={backend.key == 'chatterbox' and len(chunks) > 1}"
        )
        log(f"VOICE {voice_id} BACKEND={backend.key!r} label={backend.label!r} voice={backend.voice!r}")

        def produce() -> None:
            for idx, chunk_text in enumerate(chunks, 1):
                if producer_stop.is_set() or generation != self.current_voice_generation():
                    return
                started = time.perf_counter()
                log(f"VOICE {voice_id} TTS_REQUEST={idx}/{len(chunks)} backend={backend.key} chars={len(chunk_text)} spoken={chunk_text!r}")
                try:
                    pcm = self._fetch_tts_pcm(chunk_text, voice_id, idx, sample_rate)
                    item = (idx, chunk_text, pcm, time.perf_counter() - started, None)
                except Exception as exc:
                    item = (idx, chunk_text, b"", time.perf_counter() - started, exc)
                while not producer_stop.is_set():
                    try:
                        result_queue.put(item, timeout=.1)
                        break
                    except queue.Full:
                        continue
                if item[-1] is not None:
                    return

        producer = threading.Thread(target=produce, daemon=True, name=f"TTSPrefetch-{voice_id}")
        request_started = time.perf_counter()
        timings.tts_request = request_started
        producer.start()
        try:
            out = self._ensure_output_stream()
            for expected in range(1, len(chunks) + 1):
                wait_started = time.perf_counter()
                idx, chunk_text, pcm, synth_latency, error = result_queue.get(timeout=125)
                wait_latency = time.perf_counter() - wait_started
                if error:
                    raise error
                if idx != expected:
                    raise RuntimeError(f"TTS prefetch order error: expected {expected}, received {idx}")
                if self.stop_audio.is_set() or generation != self.current_voice_generation():
                    log(f"VOICE {voice_id} chunk {idx} discarded — barged-in/stale")
                    break
                if not timings.tts_first_audio:
                    timings.tts_first_audio = time.perf_counter()
                    base = timings.ptt_release or request_started
                    self.emit("latency", {
                        "stt": timings.stt_done - timings.ptt_release if timings.ptt_release and timings.stt_done else None,
                        "llm": timings.llm_done - timings.llm_start if timings.llm_start and timings.llm_done else None,
                        "action": timings.action_seconds,
                        "tts_first": timings.tts_first_audio - request_started,
                        "release_to_audio": timings.tts_first_audio - base,
                    })
                    log(f"VOICE {voice_id} PLAYBACK_START first_chunk={timings.tts_first_audio-request_started:.3f}s")
                pcm = fade_pcm_edge(pcm, sample_rate, edge_fade_ms, fade_in=True, fade_out=True)
                playback_started = time.perf_counter()
                for offset in range(0, len(pcm), block_bytes):
                    if self.stop_audio.is_set() or generation != self.current_voice_generation():
                        break
                    block = pcm[offset:offset + block_bytes]
                    usable = len(block) - (len(block) % 2)
                    if usable:
                        out.write(block[:usable])
                if expected < len(chunks) and pause_frames and not self.stop_audio.is_set():
                    out.write(np.zeros(pause_frames, dtype=np.int16).tobytes())
                log(
                    f"VOICE {voice_id} CHUNK_COMPLETE {idx}/{len(chunks)} chars={len(chunk_text)} "
                    f"synthesis={synth_latency:.3f}s prefetch_wait={wait_latency:.3f}s "
                    f"playback={time.perf_counter()-playback_started:.3f}s pause={pause_frames/sample_rate*1000:.0f}ms"
                )
            if not self.stop_audio.is_set() and generation == self.current_voice_generation():
                tail_frames = int(sample_rate * float(self.cfg.get("audio_tail_ms", 22)) / 1000.0)
                if tail_frames:
                    out.write(np.zeros(tail_frames, dtype=np.int16).tobytes())
                if self.stop_audio.is_set() or generation != self.current_voice_generation():
                    return
                self.conductor.acknowledge(text, announcement_receipt)
                for milestone in self.conductor.due(time.monotonic()):
                    matched_evidence = self._brief_milestone_complete(milestone["label"], text)
                    journal_citation = self._latest_journal_evidence("assistant", text) if matched_evidence else None
                    if not milestone.get("control") and matched_evidence and journal_citation:
                        if self.conductor.transition(milestone["id"], "attempted"):
                            self.conductor.transition(milestone["id"], "confirmed", {
                                "spoken": text,
                                "basis": "matched verbatim journal evidence",
                                "journal_evidence": journal_citation,
                                "elapsed_seconds": round(max(0.0, time.monotonic() - self.conductor.started_at), 3),
                            })
        except Exception as exc:
            self.emit("error", f"TTS failed: {exc}")
            log("TTS ERROR\n" + traceback.format_exc())
        finally:
            producer_stop.set()
            producer.join(timeout=.2)
            log(f"VOICE {voice_id} PLAYBACK_END generation={generation}")
            if generation == self.current_voice_generation():
                self.last_conversation_activity_at = time.monotonic()
                self.stop_audio.clear()
                if self.stt_ready.is_set():
                    self.set_state("READY")


class VoiceActivationMonitor(threading.Thread):
    """Lightweight local speech-trigger capture.

    This deliberately does not touch faster-whisper or the Director path. It only
    decides when a microphone utterance begins/ends, then hands the captured
    float32 audio to the existing _process_audio() method.

    Detection is adaptive RMS with pre-roll and hysteresis. faster-whisper's own
    VAD remains enabled during transcription, providing a second speech filter.
    """
    def __init__(self, core: GwendolynCore):
        super().__init__(daemon=True, name="VoiceActivation")
        self.core = core
        self.stop_event = threading.Event()
        self.suspend_event = threading.Event()
        self._stream_lock = threading.Lock()
        self._stream = None
        self._ptt_suspend_until = 0.0
        self._last_mode = None
        self._reset_requested = threading.Event()

    def reset_after_empty_captures(self) -> None:
        """Reopen PortAudio and discard queued noise after repeated empty STT."""
        self._reset_requested.set()

    def mode(self) -> str:
        return str(self.core.cfg.get("input_mode", "Xbox PTT"))

    def enabled(self) -> bool:
        return self.mode() in ("Voice activation", "Both")

    def suspend_for_ptt(self) -> None:
        self.suspend_event.set()
        self._close_stream()
        # Give PortAudio a moment to release the device before the PTT stream opens.
        time.sleep(0.06)

    def resume_after_ptt(self) -> None:
        self._ptt_suspend_until = time.monotonic() + 0.25
        self.suspend_event.clear()

    def _close_stream(self) -> None:
        with self._stream_lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass

    def _sensitivity_factor(self) -> float:
        return {"Low": 4.2, "Normal": 3.2, "High": 2.5}.get(
            str(self.core.cfg.get("voice_activation_sensitivity", "Normal")), 3.2
        )

    def run(self) -> None:
        q: queue.Queue = queue.Queue(maxsize=160)
        sr = int(self.core.cfg.get("sample_rate", 16000))
        block_ms = 20
        blocksize = max(160, int(sr * block_ms / 1000))

        def cb(indata, frames, time_info, status):
            if status:
                log(f"Voice activation input status: {status}")
            try:
                q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                try: q.get_nowait()
                except queue.Empty: pass
                try: q.put_nowait(indata[:, 0].copy())
                except queue.Full: pass

        noise_floor = 0.004
        speaking = False
        voiced_blocks = 0
        silent_blocks = 0
        utterance = []
        preroll = []
        utterance_samples = 0

        while not self.stop_event.is_set():
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                self._close_stream()
                while True:
                    try: q.get_nowait()
                    except queue.Empty: break
                speaking = False; voiced_blocks = 0; silent_blocks = 0; utterance = []; preroll = []; utterance_samples = 0
                noise_floor = 0.004
                self._ptt_suspend_until = time.monotonic() + 0.5
                log("VOICE ACTIVITY microphone stream reset after repeated empty captures")
            if not self.enabled() or self.suspend_event.is_set() or not self.core.stt_ready.is_set():
                self._close_stream()
                speaking = False; voiced_blocks = 0; silent_blocks = 0; utterance = []; preroll = []; utterance_samples = 0
                time.sleep(0.10)
                continue

            # Do not let Gwendolyn transcribe her own TTS. LB remains available for deliberate barge-in.
            if self.core.current_state in ("SPEAKING", "TRANSCRIBING", "THINKING", "OPENING", "LOADING STT"):
                speaking = False; voiced_blocks = 0; silent_blocks = 0; utterance = []; preroll = []; utterance_samples = 0
                while True:
                    try: q.get_nowait()
                    except queue.Empty: break
                time.sleep(0.05)
                continue

            if time.monotonic() < self._ptt_suspend_until or time.monotonic() < getattr(self.core, "vad_debounce_until", 0.0):
                time.sleep(0.05)
                continue

            with self._stream_lock:
                stream = self._stream
            if stream is None:
                try:
                    stream = sd.InputStream(
                        samplerate=sr, channels=1, dtype="float32", blocksize=blocksize,
                        device=self.core.cfg.get("microphone_device"), callback=cb
                    )
                    stream.start()
                    with self._stream_lock:
                        self._stream = stream
                    log(f"VOICE ACTIVITY ready mode={self.mode()!r} sensitivity={self.core.cfg.get('voice_activation_sensitivity','Normal')!r}")
                    self.core.emit("vad", "ready")
                except Exception as e:
                    log(f"VOICE ACTIVITY microphone open failed: {e}")
                    self.core.emit("error", f"Voice activation microphone failed: {e}")
                    time.sleep(1.0)
                    continue

            try:
                block = q.get(timeout=0.12)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64) + 1e-12))
            factor = self._sensitivity_factor()
            adaptive = float(getattr(self.core, "vad_adaptive_multiplier", 1.0))
            min_rms = float(self.core.cfg.get("voice_activation_min_rms", 0.012))
            if SESSION_MODE == "signal_lab":
                # Continuous Signal Lab output can leak into the microphone, so it
                # retains the stronger commissioning floor.
                min_rms = max(min_rms, 0.018)
            elif SESSION_MODE == "vector":
                # Vector is normally quieter. Use a modest floor that rejects room
                # noise without suppressing the user's ordinary speaking level.
                min_rms = max(min_rms, 0.014)
            threshold = max(min_rms, noise_floor * factor * adaptive)
            is_voice = rms >= threshold

            if not speaking:
                # Slowly learn the room when it is quiet; cap prevents a sudden noise becoming the new normal.
                if rms < max(0.02, threshold * 1.15):
                    noise_floor = 0.985 * noise_floor + 0.015 * max(0.001, rms)
                preroll.append(block)
                max_pre = max(1, int(int(self.core.cfg.get("voice_activation_preroll_ms", 320)) / block_ms))
                if len(preroll) > max_pre:
                    preroll.pop(0)
                if is_voice:
                    voiced_blocks += 1
                else:
                    voiced_blocks = max(0, voiced_blocks - 1)
                required_voiced_blocks = 6 if SESSION_MODE == "signal_lab" else 4
                if voiced_blocks >= required_voiced_blocks:
                    speaking = True
                    utterance = list(preroll)
                    utterance_samples = sum(len(x) for x in utterance)
                    silent_blocks = 0
                    self.core.invalidate_voice("voice activation")
                    self.core.set_state("LISTENING")
                    self.core.emit("vad", "listening")
                    log(f"VOICE ACTIVITY speech start rms={rms:.4f} threshold={threshold:.4f} noise={noise_floor:.4f}")
            else:
                utterance.append(block)
                utterance_samples += len(block)
                if is_voice:
                    silent_blocks = 0
                else:
                    silent_blocks += 1
                base_end_ms = int(self.core.cfg.get("voice_activation_end_silence_ms", 650))
                short_pause_ms = int(self.core.cfg.get("voice_activation_short_pause_ms", 1100))
                short_utterance_sec = float(self.core.cfg.get("voice_activation_short_utterance_sec", 2.6))
                spoken_sec = max(0.0, (utterance_samples / sr) - (silent_blocks * block_ms / 1000.0))
                target_end_ms = short_pause_ms if spoken_sec <= short_utterance_sec else base_end_ms
                end_blocks = max(5, int(target_end_ms / block_ms))
                max_samples = int(sr * float(self.core.cfg.get("voice_activation_max_seconds", 20)))
                if silent_blocks >= end_blocks or utterance_samples >= max_samples:
                    audio = np.concatenate(utterance).astype(np.float32, copy=False)
                    duration = len(audio) / sr
                    log(f"VOICE ACTIVITY speech end duration={duration:.2f}s silence={silent_blocks*block_ms}ms target={target_end_ms}ms")
                    speaking = False; voiced_blocks = 0; silent_blocks = 0; utterance = []; preroll = []; utterance_samples = 0
                    while True:
                        try: q.get_nowait()
                        except queue.Empty: break
                    if duration >= 0.25:
                        threading.Thread(target=self.core._process_audio, args=(audio,), daemon=True).start()
                    else:
                        self.core.set_state("READY")

        self._close_stream()

class XInputReader:
    """Small dependency-free Xbox reader for Signal Lab's local LB PTT."""
    LB = 0x0100
    DPAD = 0x000F
    DPAD_DIRECTIONS = ((0x0001, "up"), (0x0002, "down"), (0x0004, "left"), (0x0008, "right"))

    class Gamepad(ctypes.Structure):
        _fields_ = [
            ("wButtons", ctypes.c_ushort),
            ("bLeftTrigger", ctypes.c_ubyte),
            ("bRightTrigger", ctypes.c_ubyte),
            ("sThumbLX", ctypes.c_short),
            ("sThumbLY", ctypes.c_short),
            ("sThumbRX", ctypes.c_short),
            ("sThumbRY", ctypes.c_short),
        ]

    class State(ctypes.Structure):
        pass

    State._fields_ = [("dwPacketNumber", ctypes.c_uint), ("Gamepad", Gamepad)]

    def __init__(self):
        self.get_state = None
        if os.name != "nt":
            return
        for name in ("xinput1_4.dll", "xinput9_1_0.dll", "xinput1_3.dll"):
            try:
                dll = ctypes.WinDLL(name)
                fn = dll.XInputGetState
                fn.argtypes = [ctypes.c_uint, ctypes.POINTER(self.State)]
                fn.restype = ctypes.c_uint
                self.get_state = fn
                log(f"Xbox PTT direct XInput loaded from {name}")
                break
            except (OSError, AttributeError):
                continue
        if self.get_state is None:
            log("Xbox PTT direct XInput unavailable")

    def read(self) -> Dict[str, Any]:
        if self.get_state is None:
            return {"connected": False, "lb_pressed": False,
                    "lb_dpad_active": False, "ptt_candidate": False, "dpad_direction": None}
        for index in range(4):
            state = self.State()
            if self.get_state(index, ctypes.byref(state)) == 0:
                buttons = int(state.Gamepad.wButtons)
                lb = bool(buttons & self.LB)
                combo = bool(lb and (buttons & self.DPAD))
                direction = next((name for mask, name in self.DPAD_DIRECTIONS if buttons & mask), None)
                return {"connected": True, "lb_pressed": lb,
                        "lb_dpad_active": combo, "ptt_candidate": lb and not combo,
                        "dpad_direction": direction}
        return {"connected": False, "lb_pressed": False,
                "lb_dpad_active": False, "ptt_candidate": False, "dpad_direction": None}


class SDLControllerReader:
    """Read Bluetooth Xbox controllers through SDL's semantic gamepad map."""
    LEFT_SHOULDER = 9
    DPAD_BUTTONS = (11, 12, 13, 14)
    DPAD_DIRECTIONS = ((11, "up"), (12, "down"), (13, "left"), (14, "right"))

    def __init__(self):
        self.pygame = None
        self.controller_api = None
        self.device = None
        self.initialized = False
        self.read_error_logged = False

    def _initialize(self) -> None:
        """Initialize SDL on the controller-monitor thread that will poll it."""
        if self.initialized:
            return
        self.initialized = True
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
            import pygame
            from pygame._sdl2 import controller
            pygame.display.init()
            controller.init()
            self.pygame = pygame
            self.controller_api = controller
            self._connect()
        except Exception as exc:
            log(f"Xbox PTT SDL unavailable: {type(exc).__name__}: {exc}")

    def _connect(self) -> None:
        if self.controller_api is None:
            return
        try:
            count = self.controller_api.get_count()
            self.device = self.controller_api.Controller(0) if count else None
            if self.device is not None:
                log(f"Xbox PTT SDL connected: {self.device.name}")
        except Exception:
            self.device = None

    def read(self) -> Dict[str, Any]:
        self._initialize()
        if self.pygame is None or self.controller_api is None:
            return {"connected": False, "lb_pressed": False,
                    "lb_dpad_active": False, "ptt_candidate": False, "dpad_direction": None}
        try:
            self.pygame.event.pump()
            if self.device is None:
                self._connect()
            if self.device is None:
                raise RuntimeError("no SDL controller")
            lb = bool(self.device.get_button(self.LEFT_SHOULDER))
            combo = bool(lb and any(self.device.get_button(button) for button in self.DPAD_BUTTONS))
            direction = next((name for button, name in self.DPAD_DIRECTIONS
                              if self.device.get_button(button)), None)
            return {"connected": True, "lb_pressed": lb,
                    "lb_dpad_active": combo, "ptt_candidate": lb and not combo,
                    "dpad_direction": direction}
        except Exception as exc:
            if not self.read_error_logged:
                log(f"Xbox PTT SDL read failed: {type(exc).__name__}: {exc}")
                self.read_error_logged = True
            self.device = None
            return {"connected": False, "lb_pressed": False,
                    "lb_dpad_active": False, "ptt_candidate": False, "dpad_direction": None}


class ControllerMonitor(threading.Thread):
    def __init__(self, core: GwendolynCore):
        super().__init__(daemon=True)
        self.core = core
        self.stop_event = threading.Event()
        self.lb_since: Optional[float] = None
        self.ptt_started = False
        self.suppressed = False
        self.last_connected = None
        self.last_dpad_direction: Optional[str] = None
        self.last_dpad_at = 0.0
        self.gamepad = SDLControllerReader() if SESSION_MODE == "signal_lab" else None
        self.xinput = XInputReader() if SESSION_MODE == "signal_lab" else None

    def run(self) -> None:
        interval = max(0.03, int(self.core.cfg["controller_poll_ms"])/1000)
        hold = max(0.05, int(self.core.cfg["ptt_hold_ms"])/1000)
        while not self.stop_event.is_set():
            try:
                # Signal Lab is deliberately independent of Vector. Its Xbox PTT
                # reads Windows XInput directly, while Vector retains the existing
                # bridge endpoint and its LB+D-pad command suppression semantics.
                if self.gamepad is not None:
                    state = self.gamepad.read()
                    if not state.get("connected") and self.xinput is not None:
                        state = self.xinput.read()
                else:
                    state = self.core.vector_get("/v1/controller", timeout=0.65)
                connected = bool(state.get("connected"))
                if connected != self.last_connected:
                    self.core.emit("controller", connected)
                    self.last_connected = connected
                lb = bool(state.get("lb_pressed"))
                combo = bool(state.get("lb_dpad_active"))
                candidate = bool(state.get("ptt_candidate", lb and not combo))
                dpad_direction = state.get("dpad_direction")
                now = time.monotonic()

                # A bare D-pad press is silent session feedback. LB+D-pad remains
                # reserved/suppressed so it can never accidentally become speech.
                if (not lb and dpad_direction and dpad_direction != self.last_dpad_direction
                        and now - self.last_dpad_at >= 0.35):
                    self.last_dpad_at = now
                    threading.Thread(
                        target=self.core.handle_controller_feedback,
                        args=(str(dpad_direction),), daemon=True,
                        name=f"DpadFeedback-{dpad_direction}",
                    ).start()
                self.last_dpad_direction = str(dpad_direction) if dpad_direction else None

                if lb and combo:
                    self.suppressed = True
                    if self.ptt_started:
                        self.core.cancel_recording()
                        self.ptt_started = False
                ptt_enabled = str(self.core.cfg.get("input_mode", "Xbox PTT")) in ("Xbox PTT", "Both")
                if candidate and not self.suppressed and ptt_enabled:
                    if self.lb_since is None:
                        self.lb_since = now
                    elif not self.ptt_started and now - self.lb_since >= hold:
                        self.core.start_recording()
                        self.ptt_started = True
                if not lb:
                    if self.ptt_started:
                        self.core.stop_recording_and_process()
                    self.lb_since = None
                    self.ptt_started = False
                    self.suppressed = False
            except Exception:
                if self.last_connected is not False:
                    self.core.emit("controller", False)
                    self.last_connected = False
                self.lb_since = None
            time.sleep(interval)


class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(f"Gwendolyn {SESSION_LABEL} v{APP_VERSION}")
        self.cfg = load_config()
        if SESSION_MODE == "signal_lab":
            # Signal Lab defaults to deliberate LB hold-to-talk so natural pauses
            # cannot prematurely submit a partial utterance. Vector retains its
            # independently saved voice-input choice.
            self.cfg["input_mode"] = str(self.cfg.get("signal_lab_input_mode", "Xbox PTT"))
        if SESSION_MODE == "signal_lab" and self.cfg.get("voice_activation_sensitivity") == "High":
            # High is retained for Vector sessions, but continuous Signal Lab
            # audio made its old threshold prone to background activations.
            self.cfg["voice_activation_sensitivity"] = "Normal"
            log("VOICE ACTIVITY Signal Lab session reduced High sensitivity to Normal")
        self.root.geometry(self.cfg.get("window_geometry", "900x720"))
        self.events: queue.Queue = queue.Queue()
        self.core = GwendolynCore(self.cfg, self.events)
        self.controller = ControllerMonitor(self.core)
        self.voice_activation = VoiceActivationMonitor(self.core)
        self.core.voice_activation_monitor = self.voice_activation

        top = ttk.Frame(self.root, padding=8); top.pack(fill="x")
        self.state_var = tk.StringVar(value="STARTING")
        self.svc_var = tk.StringVar(value=f"{SESSION_LABEL} ?   LLM ?   TTS ?   Xbox ?")
        ttk.Label(top, textvariable=self.state_var, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.svc_var).pack(side="right")

        notebook = ttk.Notebook(self.root); notebook.pack(fill="both", expand=True, padx=8, pady=(0,8))
        chat_tab = ttk.Frame(notebook); control_tab = ttk.Frame(notebook); memory_tab = ttk.Frame(notebook); setup_tab = ttk.Frame(notebook)
        notebook.add(chat_tab, text="Conversation"); notebook.add(control_tab, text=f"{SESSION_LABEL} Control"); notebook.add(memory_tab, text="Memory"); notebook.add(setup_tab, text="Director Setup")

        self.chat = tk.Text(chat_tab, wrap="word", state="disabled", font=("Segoe UI", 11), padx=10, pady=10)
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_configure("user", font=("Segoe UI",11,"bold")); self.chat.tag_configure("gw", font=("Segoe UI",11))
        self.chat.tag_configure("sys", foreground="#666666"); self.chat.tag_configure("tool", foreground="#555588")

        bottom = ttk.Frame(chat_tab, padding=(0,8,0,0)); bottom.pack(fill="x")
        self.entry = ttk.Entry(bottom); self.entry.pack(side="left", fill="x", expand=True); self.entry.bind("<Return>", lambda e: self.send_text())
        ttk.Button(bottom, text="Send", command=self.send_text).pack(side="left", padx=(6,0))
        ptt = ttk.Button(bottom, text="Hold PTT (mouse)"); ptt.pack(side="left", padx=(6,0))
        self._mouse_ptt_active = False
        def _mouse_ptt_start(_event):
            self._mouse_ptt_active = True
            self.core.start_recording()
        def _mouse_ptt_stop(_event):
            if not self._mouse_ptt_active:
                return
            self._mouse_ptt_active = False
            self.core.stop_recording_and_process()
        ptt.bind("<ButtonPress-1>", _mouse_ptt_start)
        # Bind release at application level so dragging off the button cannot
        # strand Gwendolyn in LISTENING with an open microphone stream.
        self.root.bind_all("<ButtonRelease-1>", _mouse_ptt_stop, add="+")
        ttk.Button(bottom, text="Stop voice", command=lambda: self.core.stop_audio.set()).pack(side="left", padx=(6,0))

        self.latency_var = tk.StringVar(value="Xbox LB = hold to talk; release to send")
        ttk.Label(chat_tab, textvariable=self.latency_var, padding=(2,6,2,0)).pack(fill="x")

        if SESSION_MODE == "vector":
            self.build_vector_control_tab(control_tab)
        else:
            self.build_signal_lab_control_tab(control_tab)
        self.build_memory_tab(memory_tab)

        profile = load_profile(); self.profile_vars = {}
        private_lexicon = load_private_lexicon(); self.lexicon_vars = {}
        # Director Setup can outgrow a laptop-sized window. Keep the whole form in
        # a vertically scrollable canvas so Save is always reachable. Mouse wheel
        # scrolling is active while the pointer is over this tab.
        setup_shell = ttk.Frame(setup_tab)
        setup_shell.pack(fill="both", expand=True)
        setup_canvas = tk.Canvas(setup_shell, highlightthickness=0)
        setup_scroll = ttk.Scrollbar(setup_shell, orient="vertical", command=setup_canvas.yview)
        setup_canvas.configure(yscrollcommand=setup_scroll.set)
        setup_scroll.pack(side="right", fill="y")
        setup_canvas.pack(side="left", fill="both", expand=True)
        form = ttk.Frame(setup_canvas, padding=12)
        form_window = setup_canvas.create_window((0,0), window=form, anchor="nw")
        form.bind("<Configure>", lambda _e: setup_canvas.configure(scrollregion=setup_canvas.bbox("all")))
        setup_canvas.bind("<Configure>", lambda e: setup_canvas.itemconfigure(form_window, width=e.width))
        def _setup_wheel(e):
            setup_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        setup_canvas.bind("<Enter>", lambda _e: setup_canvas.bind_all("<MouseWheel>", _setup_wheel))
        setup_canvas.bind("<Leave>", lambda _e: setup_canvas.unbind_all("<MouseWheel>"))
        row=0
        def add_entry(label, key):
            nonlocal row
            ttk.Label(form, text=label).grid(row=row,column=0,sticky="nw",padx=(0,10),pady=5)
            v=tk.StringVar(value=profile.get(key,"")); self.profile_vars[key]=v
            e=ttk.Entry(form,textvariable=v); e.grid(row=row,column=1,sticky="ew",pady=5); row+=1
        def add_text(label,key,height=4):
            nonlocal row
            ttk.Label(form,text=label).grid(row=row,column=0,sticky="nw",padx=(0,10),pady=5)
            w=tk.Text(form,height=height,wrap="word"); w.insert("1.0",profile.get(key,"")); w.grid(row=row,column=1,sticky="nsew",pady=5); self.profile_vars[key]=w; row+=1
        add_entry("Character name", "character_name")
        add_entry("User name / salutation", "user_name")
        add_entry("Opening spoken line", "opening_line")
        add_text("Character / personality", "character_description",4)
        add_text("Role-play / relationship", "roleplay_description",4)
        add_text("Director role", "director_role",4)
        add_text("Conversation style", "conversation_style",4)
        add_text("Persistent narrative preferences", "persistent_narrative_preferences",5)
        add_text("Session brief — flexible phases", "session_brief",4)

        ttk.Separator(form, orient="horizontal").grid(row=row,column=0,columnspan=2,sticky="ew",pady=(8,10)); row+=1
        ttk.Label(form,text="Electrode hardware map",font=("Segoe UI",10,"bold")).grid(row=row,column=0,columnspan=2,sticky="w",pady=(0,4)); row+=1
        ttk.Label(form,text="Authoritative physical context supplied on every model turn; unlike the Session brief, this is not treated as optional atmosphere.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1
        add_entry("Top electrode name", "top_electrode_name")
        add_text("Top contact map", "top_electrode_map",4)
        add_entry("Bottom electrode name", "bottom_electrode_name")
        add_text("Bottom contact map", "bottom_electrode_map",5)

        ttk.Separator(form, orient="horizontal").grid(row=row,column=0,columnspan=2,sticky="ew",pady=(8,10)); row+=1
        ttk.Label(form,text="Private Language & Sensation Lexicon",font=("Segoe UI",10,"bold")).grid(row=row,column=0,columnspan=2,sticky="w",pady=(0,4)); row+=1
        ttk.Label(form,text="Saved only in private_lexicon.json beside this local Gwendolyn build. It is not part of Vector state or telemetry.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1
        ttk.Label(form,text="Language level").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        llv=tk.StringVar(value=private_lexicon.get("language_level","Natural")); self.lexicon_vars["language_level"]=llv
        ttk.Combobox(form,textvariable=llv,values=["Reserved","Natural","Explicit"],state="readonly",width=15).grid(row=row,column=1,sticky="w",pady=5); row+=1
        def add_lexicon_text(label,key,height=4):
            nonlocal row
            ttk.Label(form,text=label).grid(row=row,column=0,sticky="nw",padx=(0,10),pady=5)
            w=tk.Text(form,height=height,wrap="word"); w.insert("1.0",private_lexicon.get(key,"")); w.grid(row=row,column=1,sticky="nsew",pady=5); self.lexicon_vars[key]=w; row+=1
        add_lexicon_text("Anatomy / slang vocabulary", "anatomy_terms", 3)
        add_lexicon_text("User sensation vocabulary", "sensation_terms", 5)
        add_lexicon_text("Signal-to-language associations", "signal_associations", 7)
        add_lexicon_text("Language notes / principles", "language_notes", 5)
        ttk.Label(form,text="Signal associations guide wording only. They may suggest a sensation as possible; Gwendolyn should treat it as occurring only after you report it.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1

        ttk.Label(form,text="Director autonomy").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        av=tk.StringVar(value=profile.get("autonomy","Reactive")); self.profile_vars["autonomy"]=av
        cb=ttk.Combobox(form,textvariable=av,values=["Reactive","Suggestive","Delegated"],state="readonly"); cb.grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Reactive = asked changes only. Suggestive = may propose. Delegated = may make bounded autonomous Vector changes.",foreground="#666666").grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1
        ttk.Label(form,text="Proactive Director").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        pdv=tk.StringVar(value=profile.get("proactive_director","5 min")); self.profile_vars["proactive_director"]=pdv
        pdcb=ttk.Combobox(form,textvariable=pdv,values=["Off","5 min","10 min"],state="readonly",width=12); pdcb.grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="After this long without a meaningful Vector change, Gwendolyn may propose one. Timeline context can make her defer instead. Proposals never auto-execute.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1
        ttk.Label(form,text="Narrative Director").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        ndv=tk.StringVar(value=profile.get("narrative_director","Dynamic")); self.profile_vars["narrative_director"]=ndv
        ndcb=ttk.Combobox(form,textvariable=ndv,values=["Off","Dynamic"],state="readonly",width=12); ndcb.grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Dynamic adds a loose session arc, varied narrative beats, callbacks and restrained reuse of motifs from the Session brief. Narrative state never bypasses Vector grounding.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1
        ttk.Label(form,text="Narrative randomness").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        nrv=tk.StringVar(value=profile.get("narrative_randomness","Medium")); self.profile_vars["narrative_randomness"]=nrv
        nrcb=ttk.Combobox(form,textvariable=nrv,values=["Low","Medium","High"],state="readonly",width=12); nrcb.grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Higher randomness makes repeated runs of the same script diverge more strongly in tone, motifs and beat selection; Vector actions remain deterministic and bounded.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1
        ttk.Label(form,text="Narrative arc").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        nav=tk.StringVar(value=profile.get("narrative_arc","Automatic")); self.profile_vars["narrative_arc"]=nav
        nacb=ttk.Combobox(form,textvariable=nav,values=["Automatic","Slow burn","Deceptive calm","Challenge / relief","Escalating control","Contrast / restraint"],state="readonly",width=20); nacb.grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Automatic chooses one arc per launch. Arc templates bias the story beats only; they never alter Vector limits or bypass approval/grounding.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1

        ttk.Separator(form, orient="horizontal").grid(row=row,column=0,columnspan=2,sticky="ew",pady=(8,10)); row+=1
        ttk.Label(form,text="Reference behaviour tuning",font=("Segoe UI",10,"bold")).grid(row=row,column=0,columnspan=2,sticky="w",pady=(0,4)); row+=1
        ttk.Label(form,text="Expression preset").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        epv=tk.StringVar(value=profile.get("expression_preset","Reference")); self.expression_preset_var=epv; self.profile_vars["expression_preset"]=epv
        epf=ttk.Frame(form); epf.grid(row=row,column=1,sticky="w",pady=5)
        epcb=ttk.Combobox(epf,textvariable=epv,values=["Reference","Restrained","Expressive","Sparse / confident","Custom"],state="readonly",width=20); epcb.pack(side="left")
        ttk.Button(epf,text="Load preset",command=self.apply_expression_preset).pack(side="left",padx=(8,0)); row+=1
        ttk.Label(form,text="Presets load starting values into the sliders. Save Director Profile to make them the reference for later comparisons.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1

        def add_tuning_scale(label,key,default):
            nonlocal row
            ttk.Label(form,text=label).grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
            var=tk.IntVar(value=tuning_value(profile,key,default)); self.profile_vars[key]=var
            frame=ttk.Frame(form); frame.grid(row=row,column=1,sticky="ew",pady=5)
            scale=ttk.Scale(frame,from_=1,to=5,orient="horizontal",variable=var,command=lambda _v,k=key,v=var: self.update_tuning_label(k,v))
            scale.pack(side="left",fill="x",expand=True)
            lbl=tk.StringVar(value=str(var.get())); self.tuning_label_vars[key]=lbl
            ttk.Label(frame,textvariable=lbl,width=3).pack(side="left",padx=(10,0)); row+=1

        self.tuning_label_vars = {}
        add_tuning_scale("Metaphor density","metaphor_density",4)
        add_tuning_scale("Director pressure","director_pressure",4)
        add_tuning_scale("Question frequency","question_frequency",3)
        add_tuning_scale("Proposal frequency","proposal_frequency",3)
        add_tuning_scale("Sensation carryover","sensation_carryover",4)
        ttk.Label(form,text="All five tune language/planning only. They do not change Vector limits, approval requirements or deterministic State Arc execution.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1

        ttk.Separator(form, orient="horizontal").grid(row=row,column=0,columnspan=2,sticky="ew",pady=(8,10)); row+=1
        ttk.Label(form,text="Vector state arc",font=("Segoe UI",10,"bold")).grid(row=row,column=0,columnspan=2,sticky="w",pady=(0,4)); row+=1
        ttk.Label(form,text="Top baseline").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        btv=tk.StringVar(value=profile.get("baseline_top_focus","Top Full")); self.profile_vars["baseline_top_focus"]=btv
        ttk.Combobox(form,textvariable=btv,values=["Top Sweep","Top Full","Glans Focus","Shaft Focus","Lower Shaft Focus","Root Focus"],state="readonly",width=22).grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Bottom baseline").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        bbv=tk.StringVar(value=profile.get("baseline_bottom_focus","Bottom Full")); self.profile_vars["baseline_bottom_focus"]=bbv
        ttk.Combobox(form,textvariable=bbv,values=["Bottom Full","Bottom Sweep","Prostate Focus","Anal Focus","Perineum Focus"],state="readonly",width=22).grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Texture baseline").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        btx=tk.StringVar(value=profile.get("baseline_texture","Normal")); self.profile_vars["baseline_texture"]=btx
        ttk.Combobox(form,textvariable=btx,values=["Normal","Smooth","Smoothest","Rough","Roughest"],state="readonly",width=22).grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Variation baseline").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        bvv=tk.StringVar(value=profile.get("baseline_variation","Normal")); self.profile_vars["baseline_variation"]=bvv
        ttk.Combobox(form,textvariable=bvv,values=["Normal","Still","Subtle","Lively","Wild"],state="readonly",width=22).grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Changes before recovery").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        isv=tk.StringVar(value=profile.get("intervention_sequence","2–4 changes")); self.profile_vars["intervention_sequence"]=isv
        ttk.Combobox(form,textvariable=isv,values=["2–3 changes","2–4 changes","3–4 changes"],state="readonly",width=22).grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="After this many meaningful excursions, Gwendolyn is biased toward restoring one lane or control toward baseline before beginning a new sequence. Top and Bottom can recover independently.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1
        ttk.Label(form,text="Speech pace").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        pv=tk.StringVar(value=profile.get("speech_pace","Deliberate")); self.profile_vars["speech_pace"]=pv
        pcb=ttk.Combobox(form,textvariable=pv,values=["Natural","Deliberate","Slow"],state="readonly"); pcb.grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Pace uses wording/punctuation/prosody cues; audio is not time-stretched.",foreground="#666666").grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1
        ttk.Label(form,text="Speech chunking").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        cv=tk.StringVar(value=profile.get("speech_chunking","Off")); self.profile_vars["speech_chunking"]=cv
        ccb=ttk.Combobox(form,textvariable=cv,values=["Off","First sentence"],state="readonly"); ccb.grid(row=row,column=1,sticky="w",pady=5); row+=1
        ttk.Label(form,text="Fast start: synthesize the first sentence first so expressive replies begin speaking sooner; the remainder follows sequentially.",foreground="#666666").grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1

        ttk.Label(form,text="Response verbosity").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        vv=tk.IntVar(value=int(profile.get("speech_verbosity",4))); self.verbosity_var=vv; self.profile_vars["speech_verbosity"]=vv
        vframe=ttk.Frame(form); vframe.grid(row=row,column=1,sticky="ew",pady=5)
        vscale=ttk.Scale(vframe,from_=1,to=5,orient="horizontal",variable=vv,command=self.update_verbosity_label); vscale.pack(side="left",fill="x",expand=True)
        self.verbosity_label_var=tk.StringVar(value=self.verbosity_label(vv.get())); ttk.Label(vframe,textvariable=self.verbosity_label_var,width=11).pack(side="left",padx=(10,0)); row+=1
        ttk.Label(form,text="1 = concise · 3 = natural · 5 = expansive. Saved with the Director profile.",foreground="#666666").grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1

        ttk.Separator(form, orient="horizontal").grid(row=row,column=0,columnspan=2,sticky="ew",pady=(8,10)); row+=1
        ttk.Label(form,text="Voice input").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        iv=tk.StringVar(value=self.cfg.get("input_mode","Both")); self.input_mode_var=iv
        icb=ttk.Combobox(form,textvariable=iv,values=["Xbox PTT","Voice activation","Both"],state="readonly",width=22)
        icb.grid(row=row,column=1,sticky="w",pady=5); icb.bind("<<ComboboxSelected>>", lambda _e: self.save_voice_input_from_ui()); row+=1
        ttk.Label(form,text="Xbox PTT uses the left bumper: hold while speaking, then release to send. Both also enables voice activation.",foreground="#666666").grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1
        ttk.Label(form,text="Voice sensitivity").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        sv=tk.StringVar(value=self.cfg.get("voice_activation_sensitivity","Normal")); self.voice_sensitivity_var=sv
        scb=ttk.Combobox(form,textvariable=sv,values=["Low","Normal","High"],state="readonly",width=15)
        scb.grid(row=row,column=1,sticky="w",pady=5); scb.bind("<<ComboboxSelected>>", lambda _e: self.save_voice_input_from_ui()); row+=1
        ttk.Label(form,text="Normal is recommended. High triggers more easily; Low is better in a noisy room.",foreground="#666666").grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1
        ttk.Label(form,text="End-of-speech silence (ms)").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        ev=tk.StringVar(value=str(self.cfg.get("voice_activation_end_silence_ms",650))); self.voice_end_silence_var=ev
        ecb=ttk.Combobox(form,textvariable=ev,values=["400","500","650","800","1000"],state="readonly",width=10)
        ecb.grid(row=row,column=1,sticky="w",pady=5); ecb.bind("<<ComboboxSelected>>", lambda _e: self.save_voice_input_from_ui()); row+=1

        ttk.Separator(form, orient="horizontal").grid(row=row,column=0,columnspan=2,sticky="ew",pady=(8,10)); row+=1
        ttk.Label(form,text="TTS backend").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        choices = self.core.tts.backend_choices()
        self.tts_label_to_key = {label:key for key,label in choices}
        current_backend = self.core.tts.backend()
        tv=tk.StringVar(value=current_backend.label); self.tts_backend_var=tv
        tcb=ttk.Combobox(form,textvariable=tv,values=[label for _key,label in choices],state="readonly",width=28)
        tcb.grid(row=row,column=1,sticky="w",pady=5)
        tcb.bind("<<ComboboxSelected>>", lambda _event: self.save_tts_backend_from_ui()); row+=1
        ttk.Label(form,text="Selection applies immediately. Installed Kokoro and Chatterbox bridges auto-start with Gwendolyn.",foreground="#666666").grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1

        ttk.Label(form,text="Director voice").grid(row=row,column=0,sticky="w",padx=(0,10),pady=5)
        self.director_voice_labels = {data["label"]:key for key,data in DIRECTOR_VOICES.items()}
        current_voice = DIRECTOR_VOICES[self.core.director_voice_key()]["label"]
        self.director_voice_var = tk.StringVar(value=current_voice)
        voice_frame = ttk.Frame(form); voice_frame.grid(row=row,column=1,sticky="w",pady=5)
        voice_cb = ttk.Combobox(voice_frame,textvariable=self.director_voice_var,values=list(self.director_voice_labels),state="readonly",width=30)
        voice_cb.pack(side="left"); voice_cb.bind("<<ComboboxSelected>>", lambda _e: self.save_director_voice_from_ui())
        ttk.Button(voice_frame,text="Test voice",command=self.test_director_voice).pack(side="left",padx=(8,0)); row+=1
        self.director_personality_var = tk.BooleanVar(value=bool(self.cfg.get("director_personality_profile", True)))
        ttk.Checkbutton(form,text="Apply voice-specific Director Profile",variable=self.director_personality_var,command=self.save_director_personality_mode).grid(row=row,column=1,sticky="w",pady=(0,4)); row+=1
        ttk.Label(form,text="When enabled, the selected character supplies her own pacing, verbosity, pressure, question/proposal balance, sensation carryover and narrative arc. Shared Vector grounding and limits never change.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1
        if SESSION_MODE == "vector":
            self.vector_generated_motion_var = tk.BooleanVar(value=bool(self.cfg.get("vector_autonomous_generate_tcode", False)))
            ttk.Checkbutton(form,text="Autonomous Vector: Gwendolyn generates incoming motion",variable=self.vector_generated_motion_var,command=self.save_vector_generated_motion_mode).grid(row=row,column=1,sticky="w",pady=(0,4)); row+=1
            ttk.Label(form,text="Off keeps authored T-code. On lets Gwendolyn choose bounded motion plans; Vector validates and generates the continuous L0 stream.",foreground="#666666",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,8)); row+=1
        self.director_personality_summary_var = tk.StringVar(value=self.personality_summary())
        ttk.Label(form,textvariable=self.director_personality_summary_var,foreground="#3f4c5a",wraplength=690).grid(row=row,column=1,sticky="w",pady=(0,10)); row+=1

        ttk.Button(form,text="Save Director Profile",command=self.save_profile_from_ui).grid(row=row,column=1,sticky="w",pady=8)
        form.columnconfigure(1,weight=1)
        for r in range(1,row): form.rowconfigure(r,weight=0)
        self.append("sys", "Gwendolyn Direct is starting. SillyTavern is not used.\n")
        # Warm Qwen first so it remains GPU-resident before Chatterbox allocates VRAM.
        threading.Thread(target=self.core.warm_ollama, daemon=True, name="OllamaWarmup").start()
        threading.Thread(target=self.core.autostart_voice_services, daemon=True, name="VoiceAutostart").start()
        threading.Thread(target=self.core.load_stt, daemon=True).start()
        threading.Thread(target=self.service_loop, daemon=True).start()
        self.voice_activation.start()
        # Both modes need the monitor: Vector polls its bridge endpoint, while
        # Signal Lab reads the Bluetooth controller locally through SDL/XInput.
        self.controller.start()
        self.root.after(50, self.pump_events)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_vector_control_tab(self, tab) -> None:
        from tkinter import ttk
        tk = self.tk
        shell = ttk.Frame(tab, padding=12); shell.pack(fill="both", expand=True)
        self.vector_control_status = tk.StringVar(value="Waiting for authoritative Vector state…")
        ttk.Label(shell, textvariable=self.vector_control_status, font=("Segoe UI",11,"bold"), wraplength=820).pack(fill="x", pady=(0,10))
        self.vector_control_buttons = {}

        def section(title, description):
            box = ttk.LabelFrame(shell, text=title, padding=10); box.pack(fill="x", pady=5)
            ttk.Label(box, text=description, foreground="#555555", wraplength=810).pack(anchor="w", pady=(0,7))
            return box

        def action_button(parent, label, name, args, dimension=None, value=None):
            button = tk.Button(
                parent, text=label, padx=9, pady=5, relief="raised", borderwidth=1,
                bg="#e9eef5", activebackground="#d7e2ef",
                command=lambda: self.manual_vector_action(name, dict(args)),
            )
            button.pack(side="left", padx=(0,6), pady=2)
            if dimension and value:
                self.vector_control_buttons[(dimension, value)] = button
            return button

        top_name = self.core.profile.get("top_electrode_name", "Top electrode")
        top = section(top_name, "E1 glans/corona · E2 mid-shaft · E3 lower shaft · E4 penis base/root")
        row = ttk.Frame(top); row.pack(fill="x")
        for value, label in (("Glans Focus","E1 Glans"),("Shaft Focus","E2 Mid-shaft"),("Lower Shaft Focus","E3 Lower shaft"),("Root Focus","E4 Root"),("Top Sweep","Sweep"),("Top Full","Full")):
            action_button(row,label,"vector_set_top_focus",{"top_focus":value},"top_focus",value)
        gain = ttk.Frame(top); gain.pack(fill="x", pady=(5,0))
        action_button(gain,"Less top","vector_top_spatial_gain",{"action":"decrease"})
        action_button(gain,"Restore top","vector_top_spatial_gain",{"action":"restore"})
        action_button(gain,"More top","vector_top_spatial_gain",{"action":"increase"})

        bottom_name = self.core.profile.get("bottom_electrode_name", "Bottom electrode")
        bottom = section(bottom_name, "A prostate · B ass · C perineum/balls · Full stroke ABCBA · Prostate Focus ABA")
        row = ttk.Frame(bottom); row.pack(fill="x")
        for value, label in (("Prostate Focus","ABA Prostate"),("Anal Focus","B Ass"),("Perineum Focus","C Perineum/balls"),("Bottom Sweep","ABCBA Sweep"),("Bottom Full","Full")):
            action_button(row,label,"vector_set_bottom_focus",{"bottom_focus":value},"bottom_focus",value)
        gain = ttk.Frame(bottom); gain.pack(fill="x", pady=(5,0))
        action_button(gain,"Less bottom","vector_bottom_spatial_gain",{"action":"decrease"})
        action_button(gain,"Restore bottom","vector_bottom_spatial_gain",{"action":"restore"})
        action_button(gain,"More bottom","vector_bottom_spatial_gain",{"action":"increase"})

        shared = section("Shared character", "Active choices are highlighted in green. Every other button is an available change.")
        row = ttk.Frame(shared); row.pack(fill="x")
        for value in ("Smoothest","Smooth","Normal","Rough","Roughest"):
            action_button(row,value,"vector_set_texture",{"texture":value},"texture",value)
        row = ttk.Frame(shared); row.pack(fill="x", pady=(4,0))
        for value in ("Still","Subtle","Normal","Lively","Wild"):
            action_button(row,value,"vector_set_variation",{"variation":value},"variation",value)

        timing = section("Tempo and engine", "Temporary tempo windows restore automatically; Stop and Neutral remain explicit.")
        row = ttk.Frame(timing); row.pack(fill="x")
        action_button(row,"½ speed · 30s","vector_tempo_window",{"scale":0.5,"duration_seconds":30})
        action_button(row,"2× speed · 15s","vector_tempo_window",{"scale":2.0,"duration_seconds":15})
        action_button(row,"Restore authored","vector_restore_modifiers",{})
        action_button(row,"Resume","vector_resume",{})
        action_button(row,"Neutral","vector_neutral",{})
        action_button(row,"STOP","vector_stop",{})

    def build_signal_lab_control_tab(self, tab) -> None:
        from tkinter import ttk
        shell = ttk.Frame(tab, padding=18); shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Signal Lab session", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 10))
        ttk.Label(
            shell,
            text=("This Gwendolyn session is permanently connected to Signal Lab. Vector commands are disabled. "
                  "Signal generation, the latched audio device, presets and operator limits remain in the separate "
                  "Gwendolyn Signal Lab window."),
            wraplength=800, justify="left",
        ).pack(anchor="w")
        ttk.Label(
            shell,
            text=("Ask Gwendolyn to create, smooth, roughen, strengthen, weaken, vary or neutralise the two-lane signal. "
                  "Left controls Stairway E1 ↔ E4; right controls Moaner tip ↔ base."),
            wraplength=800, justify="left", foreground="#555555",
        ).pack(anchor="w", pady=(12, 0))

    def build_memory_tab(self, tab) -> None:
        from tkinter import ttk
        shell = ttk.Frame(tab, padding=12); shell.pack(fill="both", expand=True)
        ttk.Label(
            shell,
            text=("Memory is local and reviewable. Ordinary praise creates low-confidence evidence only. "
                  "Evidence must recur across three separate sessions before it becomes learned; “remember that…” learns immediately."),
            foreground="#555555", wraplength=830,
        ).pack(fill="x", pady=(0,10))
        columns = ("status", "scope", "confidence", "memory")
        self.memory_tree = ttk.Treeview(shell, columns=columns, show="headings", selectmode="browse")
        self.memory_tree.heading("status", text="Level")
        self.memory_tree.heading("scope", text="Personality")
        self.memory_tree.heading("confidence", text="Confidence")
        self.memory_tree.heading("memory", text="Preference or evidence")
        self.memory_tree.column("status", width=85, stretch=False)
        self.memory_tree.column("scope", width=95, stretch=False)
        self.memory_tree.column("confidence", width=80, stretch=False)
        self.memory_tree.column("memory", width=540, stretch=True)
        scroll = ttk.Scrollbar(shell, orient="vertical", command=self.memory_tree.yview)
        self.memory_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.memory_tree.pack(fill="both", expand=True)
        actions = ttk.Frame(shell); actions.pack(fill="x", pady=(10,0))
        ttk.Button(actions, text="Pin selected", command=self.pin_selected_memory).pack(side="left")
        ttk.Button(actions, text="Forget selected", command=self.forget_selected_memory).pack(side="left", padx=(7,0))
        ttk.Button(actions, text="Refresh", command=self.refresh_memory_view).pack(side="left", padx=(7,0))
        self.memory_status_var = self.tk.StringVar(value="")
        ttk.Label(actions, textvariable=self.memory_status_var, foreground="#555555").pack(side="left", padx=(12,0))
        self.refresh_memory_view()

    def refresh_memory_view(self) -> None:
        if not hasattr(self, "memory_tree"):
            return
        for item in self.memory_tree.get_children():
            self.memory_tree.delete(item)
        rows = self.core.memory_rows()
        rank = {"pinned": 0, "learned": 1, "evidence": 2}
        rows.sort(key=lambda x: (rank.get(str(x.get("status")), 9), -float(x.get("confidence", 0))))
        for row in rows:
            memory_id = str(row.get("id") or "")
            self.memory_tree.insert("", "end", iid=memory_id, values=(
                str(row.get("status") or ""), str(row.get("scope") or "shared"),
                f"{float(row.get('confidence', 0)):.0%}", str(row.get("text") or ""),
            ))
        if hasattr(self, "memory_status_var"):
            learned = sum(1 for x in rows if x.get("status") in ("learned", "pinned"))
            evidence = sum(1 for x in rows if x.get("status") == "evidence")
            self.memory_status_var.set(f"{learned} durable · {evidence} evidence")

    def _selected_memory_id(self) -> str:
        selected = self.memory_tree.selection() if hasattr(self, "memory_tree") else ()
        return str(selected[0]) if selected else ""

    def pin_selected_memory(self) -> None:
        memory_id = self._selected_memory_id()
        if memory_id and self.core.pin_memory(memory_id):
            self.refresh_memory_view()

    def forget_selected_memory(self) -> None:
        memory_id = self._selected_memory_id()
        if memory_id and self.core.forget_memory(memory_id):
            self.refresh_memory_view()

    def manual_vector_action(self, name: str, args: Dict[str, Any]) -> None:
        if self.core.current_state in ("STARTING", "OPENING"):
            self.append("sys", "Vector controls will be available after startup completes.")
            return
        def run():
            result = self.core.execute_tool(name, args)
            if not result.get("ok"):
                self.core.emit("error", f"Vector control failed: {result.get('error','unknown error')}")
            self.core.emit("control_state", self.core.control_snapshot())
        threading.Thread(target=run, daemon=True, name="VectorControl").start()

    def update_vector_control(self, snapshot: Dict[str, Any]) -> None:
        if snapshot.get("error"):
            self.vector_control_status.set("Vector state unavailable: " + str(snapshot["error"]))
            return
        active = snapshot.get("active") or {}
        top = active.get("top_focus") or "?"; bottom = active.get("bottom_focus") or "?"
        texture = active.get("texture") or "?"; variation = active.get("variation") or "?"
        engine = snapshot.get("engine_state") or "?"
        self.vector_control_status.set(f"{engine}  ·  Top: {top}  ·  Bottom: {bottom}  ·  Texture: {texture}  ·  Variation: {variation}")
        for (dimension, value), button in self.vector_control_buttons.items():
            selected = str(active.get(dimension) or "").strip().lower() == value.lower()
            button.configure(
                bg="#2e7d32" if selected else "#e9eef5",
                fg="white" if selected else "black",
                disabledforeground="white",
                relief="sunken" if selected else "raised",
                font=("Segoe UI",9,"bold" if selected else "normal"),
                state="disabled" if selected else "normal",
            )

    def update_tuning_label(self, key: str, var) -> None:
        if hasattr(self, "tuning_label_vars") and key in self.tuning_label_vars:
            try:
                self.tuning_label_vars[key].set(str(max(1,min(5,int(round(float(var.get())))))))
            except Exception:
                self.tuning_label_vars[key].set("3")
        if hasattr(self, "expression_preset_var") and self.expression_preset_var.get() not in ("Custom", ""):
            self.expression_preset_var.set("Custom")

    def apply_expression_preset(self) -> None:
        name = self.expression_preset_var.get().strip() if hasattr(self, "expression_preset_var") else "Reference"
        values = TUNING_PRESETS.get(name)
        if not values:
            self.append("sys", "Custom expression tuning left unchanged.")
            return
        for key, value in values.items():
            widget = self.profile_vars.get(key)
            if widget is not None:
                try: widget.set(value)
                except Exception: pass
            if hasattr(self, "tuning_label_vars") and key in self.tuning_label_vars:
                self.tuning_label_vars[key].set(str(value))
        self.expression_preset_var.set(name)
        if hasattr(self, "verbosity_label_var") and "speech_verbosity" in values:
            self.verbosity_label_var.set(self.verbosity_label(values["speech_verbosity"]))
        self.append("sys", f"Loaded expression preset: {name}. Save Director Profile to keep it.")

    def verbosity_label(self, value) -> str:
        try:
            n=max(1,min(5,int(round(float(value)))))
        except Exception:
            n=4
        return {1:"Concise",2:"Brief",3:"Natural",4:"Expressive",5:"Expansive"}[n]

    def update_verbosity_label(self, value=None) -> None:
        if hasattr(self, "verbosity_label_var"):
            self.verbosity_label_var.set(self.verbosity_label(self.verbosity_var.get()))

    def save_profile_from_ui(self) -> None:
        profile = {}
        for key, widget in self.profile_vars.items():
            if isinstance(widget, self.tk.Text):
                profile[key] = widget.get("1.0", "end").strip()
            else:
                value = widget.get()
                profile[key] = value.strip() if isinstance(value, str) else value
        try:
            profile["speech_verbosity"] = max(1, min(5, int(round(float(profile.get("speech_verbosity", 4))))))
        except Exception:
            profile["speech_verbosity"] = 4
        for key, default in (("metaphor_density",4),("director_pressure",4),("question_frequency",3),("proposal_frequency",3),("sensation_carryover",4)):
            profile[key] = tuning_value(profile, key, default)
        self.core.update_profile(profile)
        lexicon = {}
        for key, widget in self.lexicon_vars.items():
            if isinstance(widget, self.tk.Text):
                lexicon[key] = widget.get("1.0", "end").strip()
            else:
                value = widget.get()
                lexicon[key] = value.strip() if isinstance(value, str) else value
        self.core.update_private_lexicon(lexicon)

    def save_voice_input_from_ui(self) -> None:
        self.cfg["input_mode"] = self.input_mode_var.get().strip()
        if SESSION_MODE == "signal_lab":
            self.cfg["signal_lab_input_mode"] = self.cfg["input_mode"]
        self.cfg["voice_activation_sensitivity"] = self.voice_sensitivity_var.get().strip()
        try:
            self.cfg["voice_activation_end_silence_ms"] = int(self.voice_end_silence_var.get())
        except Exception:
            self.cfg["voice_activation_end_silence_ms"] = 650
        save_config(self.cfg)
        self.append("sys", f"Voice input: {self.cfg['input_mode']} / sensitivity {self.cfg['voice_activation_sensitivity']} / end silence {self.cfg['voice_activation_end_silence_ms']} ms")

    def save_tts_backend_from_ui(self) -> None:
        label = self.tts_backend_var.get().strip()
        key = self.tts_label_to_key.get(label)
        if not key:
            self.append("sys", f"Unknown TTS backend: {label}")
            return
        if key == self.cfg.get("tts_backend"):
            self.append("sys", f"TTS backend remains {label}.")
            return
        self.core.invalidate_voice("TTS backend change")
        self.core.close_output_stream()
        self.cfg["tts_backend"] = key
        save_config(self.cfg)
        self.core.tts = TTSAdapter(self.cfg, self.core.session)
        self.append("sys", f"TTS backend selected: {label}. Next utterance will use it.")
        self.core.check_services()

    def save_director_voice_from_ui(self) -> None:
        label = self.director_voice_var.get().strip()
        key = self.director_voice_labels.get(label)
        ok, message = self.core.select_director_voice(key or "")
        if ok:
            self.append("sys", f"Director voice selected: {message}.")
            if hasattr(self, "director_personality_summary_var"):
                self.director_personality_summary_var.set(self.personality_summary())
        else:
            self.append("sys", message)
            self.director_voice_var.set(DIRECTOR_VOICES[self.core.director_voice_key()]["label"])

    def test_director_voice(self) -> None:
        name = self.core.director_name()
        line = self.core.director_voice_profile().get("opening") or f"Good evening. This is {name}. I am listening carefully."
        self.core.invalidate_voice("Director voice test")
        self.core.emit("assistant", line)
        self.core.enqueue_voice(line, Timings(), voice_id=self.core.next_voice_tx("VOICE-TEST"))

    def save_director_personality_mode(self) -> None:
        enabled = bool(self.director_personality_var.get())
        self.core.set_director_personality_profile(enabled)
        if hasattr(self, "director_personality_summary_var"):
            self.director_personality_summary_var.set(self.personality_summary())
        self.append("sys", "Voice-specific Director Profile enabled." if enabled else "Using the shared Director Setup tuning for every voice.")

    def save_vector_generated_motion_mode(self) -> None:
        enabled = bool(self.vector_generated_motion_var.get())
        self.cfg["vector_autonomous_generate_tcode"] = enabled
        save_config(self.cfg)
        self.append("sys", "Autonomous Vector generated motion enabled." if enabled else "Autonomous Vector will retain authored T-code.")

    def personality_summary(self) -> str:
        selected = self.core.director_voice_profile()
        policy = selected.get("operating_profile") or {}
        enabled = bool(self.director_personality_var.get()) if hasattr(self, "director_personality_var") else bool(self.cfg.get("director_personality_profile", True))
        if not enabled:
            return "Personality policy is disabled; only the selected voice is used."
        archetype = policy.get("archetype", "Director")
        doctrine = policy.get("doctrine", selected.get("personality", ""))
        return f"{selected.get('name','Director')} — {archetype}: {doctrine}"

    def append(self, tag: str, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", text + ("\n" if not text.endswith("\n") else ""), tag)
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def send_text(self) -> None:
        text = self.entry.get().strip(); self.entry.delete(0,"end")
        if text: self.core.typed_turn(text)

    def service_loop(self) -> None:
        while True:
            self.core.check_services()
            self.core.proactive_tick()
            try:
                self.core.restim_sensor_checkin_tick()
            except Exception as exc:
                log(f"RESTIM SENSOR check-in failed: {exc}")
            if SESSION_MODE == "vector":
                try:
                    self.core.vector_heart_tempo_tick()
                except Exception as exc:
                    log(f"HEART TEMPO Vector pacing rejected: {exc}")
                self.core.emit("control_state", self.core.control_snapshot())
            time.sleep(3)

    def pump_events(self) -> None:
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "state": self.state_var.set(str(data))
                elif kind == "user": self.append("user", f"You: {data}")
                elif kind == "assistant": self.append("gw", f"{self.core.director_name()}: {data}\n")
                elif kind == "log": self.append("sys", str(data))
                elif kind == "error": self.append("sys", "ERROR: " + str(data))
                elif kind == "tool": self.append("tool", f"Director: {data['name']} {data['args']}")
                elif kind == "control_state" and SESSION_MODE == "vector": self.update_vector_control(data)
                elif kind == "memory_changed": self.refresh_memory_view()
                elif kind == "vad":
                    if data == "listening": self.latency_var.set("Voice activation: listening…")
                    elif data == "ready" and self.cfg.get("input_mode") in ("Voice activation","Both"):
                        self.latency_var.set("Voice activation ready — speak naturally; LB PTT remains available in Both mode")
                elif kind == "controller":
                    self._controller_ok = bool(data); self.update_services()
                elif kind == "services":
                    self._services = data; self.update_services()
                elif kind == "latency":
                    stt = data.get("stt")
                    llm=data.get("llm"); action=data.get("action",0.0)
                    parts=[]
                    if stt is not None: parts.append(f"STT {stt:.2f}s")
                    if llm is not None: parts.append(f"LLM {llm:.2f}s")
                    if action: parts.append(f"Vector {action:.2f}s")
                    parts.append(f"TTS→first audio {data['tts_first']:.2f}s")
                    parts.append(f"release→audio {data['release_to_audio']:.2f}s")
                    self.latency_var.set("Latency: " + "   ".join(parts))
        except queue.Empty:
            pass
        self.root.after(50, self.pump_events)

    def update_services(self) -> None:
        s = getattr(self, "_services", {})
        x = getattr(self, "_controller_ok", False)
        yes = lambda v: "●" if v else "○"
        tts_label = s.get("TTS_label") or "TTS"
        llm_label = s.get("LLM_label") or "LLM"
        self.svc_var.set(f"Vector {yes(s.get('Vector'))}   {llm_label} {yes(s.get('Ollama'))}   {tts_label} {yes(s.get('TTS'))}   Xbox {yes(x)}")

    def on_close(self) -> None:
        self.controller.stop_event.set()
        self.voice_activation.stop_event.set()
        self.voice_activation._close_stream()
        self.core.stop_audio.set()
        try: self.core.cancel_recording()
        except Exception: pass
        try: self.core.close_output_stream()
        except Exception: pass
        self.core.close_heart_tempo_listener()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


_SINGLETON_SOCKET = None
_SINGLETON_PORT = 18765

def acquire_single_instance() -> bool:
    """Own a loopback TCP port for the lifetime of the app.

    This prevents multiple Gwendolyn Direct processes from responding to the
    same Vector/Xbox PTT gesture at the same time.
    """
    global _SINGLETON_SOCKET
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind(("127.0.0.1", _SINGLETON_PORT))
        s.listen(1)
        _SINGLETON_SOCKET = s
        log(f"Single-instance lock acquired on 127.0.0.1:{_SINGLETON_PORT}")
        return True
    except OSError:
        try:
            s.close()
        except Exception:
            pass
        return False



def main() -> int:
    if not acquire_single_instance():
        msg = (
            "Gwendolyn Direct is already running.\n\n"
            "Only one instance is allowed so one Xbox PTT command cannot create "
            "multiple simultaneous replies."
        )
        log("Second Gwendolyn Direct launch blocked by singleton lock.")
        try:
            import tkinter.messagebox as _mb
            _mb.showinfo("Gwendolyn Direct", msg)
        except Exception:
            print(msg)
        return 2
    try:
        App().run()
        return 0
    except Exception as e:
        log(traceback.format_exc())
        print(f"Fatal error: {e}")
        input("Press Enter to close…")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


DIRECTOR_STYLE_V031 = (
    "\n\nConversation style: Be an active, observant Director rather than a help-desk assistant. "
    "Avoid repeatedly ending replies with phrases such as 'let me know', 'what would you like next?', "
    "or 'shall we continue?'. Prefer grounded observations, confident continuity, teasing when appropriate, "
    "and references to the actual current Vector state. Do not invent actions that were not executed."
)
