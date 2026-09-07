# Gwendolyn Direct

Gwendolyn is a local voice-and-text AI Director for [Vector 1A](https://github.com/Nephet2/restim-vector-live). It reads Vector's Director API, discusses the live state with a locally hosted language model, and can perform the bounded Vector actions exposed by that API.

This is an early public alpha for Windows. It contains the v0.46.11 application line with Vector integration, local persistent memory, multiple model backends, optional local speech, heart-tempo following, curated temporary Vector events, sensor auto-reconnect and Signal Lab conducted-session refinements. Personal configuration, memories, logs, handoffs, model files and voice recordings are not included.

Signal Lab is published separately at [Nephet2/gwendolyn-signal-lab](https://github.com/Nephet2/gwendolyn-signal-lab). Vector control uses [Vector 1A](https://github.com/Nephet2/restim-vector-live); v0.46.11 is intended for Vector alpha78 or newer.

## What you need

- Windows 10 or 11
- 64-bit Python 3.11 or 3.12
- A local model server: Ollama is the easiest starting point; llama.cpp is supported through its OpenAI-compatible server
- [Vector 1A v1.6.0-alpha77](https://github.com/Nephet2/restim-vector-live/releases/tag/v1.6.0-alpha77) or newer
- A microphone for voice input; text input works without one
- Optional: Kokoro or Chatterbox for spoken replies

## First run

1. Extract the release ZIP to a normal writable folder.
2. Run `SETUP_GWENDOLYN.bat`.
3. Set up and start a model server using [docs/MODEL_SERVER_SETUP.md](docs/MODEL_SERVER_SETUP.md).
4. Install and start Vector using [docs/VECTOR_SETUP.md](docs/VECTOR_SETUP.md).
5. Run `CHECK_SETUP.bat`. It only performs read-only health checks.
6. Run `START_GWENDOLYN_VECTOR.bat`.

For the separate audio-signal path, start Signal Lab first and then run
`START_GWENDOLYN_SIGNAL_LAB.bat`. The two launchers deliberately select distinct
session modes so a session cannot control both applications by accident.

The setup script creates local copies of the four `*.example.json` files. Edit those local JSON files to change the model, name, personality, speech backend and private vocabulary. They are deliberately ignored by Git.

## Speech

Text chat and model/Vector control work when a speech bridge is unavailable. For a simple local voice, run `SETUP_KOKORO.bat`, then start `START_KOKORO.bat` before Gwendolyn. See [docs/VOICE_SETUP.md](docs/VOICE_SETUP.md) for details and the optional Chatterbox bridge.

## Responsible operation

Gwendolyn can request real output changes through Vector. Configure and test Vector's own output limits first, retain immediate access to Neutral/stop and the physical master level, and begin each new setup at a low level. Autonomous generated motion is off by default and must be deliberately enabled in Gwendolyn.

Custom events also require a separate Vector opt-in. Their names describe authored recipes, not measured sensations. Vector limits Gwendolyn to the curated catalogue and a 2–30 second duration; Neutral and Stop clear active events.

The project is intended for consenting adults. It is not a medical device and cannot infer subjective sensation from telemetry. Keep Vector, model and speech servers bound to localhost unless you have secured them yourself.

## Privacy

Conversation memory and profile data stay in local JSON files beside the app. See [SECURITY.md](SECURITY.md) before sharing logs or diagnostics. No cloud model is required by the supplied configuration.

## Development

Run:

```powershell
python -m unittest discover -p "test_*.py"
python check_setup.py --offline
```

Gwendolyn Direct is released under the MIT License. Models, voice engines, Python packages and Vector have their own licenses.
