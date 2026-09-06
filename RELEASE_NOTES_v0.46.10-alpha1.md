# Gwendolyn Direct v0.46.10-alpha1

This is the first sanitized public Windows alpha of Gwendolyn Direct.

## Included

- Local text and microphone conversation with faster-whisper speech recognition.
- Ollama and OpenAI-compatible model-server backends, including llama.cpp.
- Vector Director API state, bounded semantic actions, timeline grounding and immediate stop handling.
- Optional Kokoro and Chatterbox-compatible local speech bridges.
- Local, reviewable persistent memory and editable Director/profile settings.
- Heart-tempo following, Signal Lab support and ReStim sensor auto-reconnect from the v0.46.10 development line.
- Read-only setup checker plus complete model-server and Vector commissioning guides.

## Public-package changes

- Removed personal names, machine paths, configuration, memories, logs, handoffs and private vocabulary.
- No model, Python environment, device configuration or voice recording is bundled.
- The supplied defaults bind services to localhost, use Ollama, and leave autonomous generated Vector motion disabled.
- Voice selection accepts a profile without a reference WAV, so text chat and Kokoro can work before a personal voice is configured.

## Compatibility

Use Vector 1A v1.6.0-alpha76 or newer. This remains alpha software; commission the model-to-Vector path at low output and verify Neutral/stop before a normal session.
