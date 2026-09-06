# Optional speech output

Gwendolyn can run as text chat when no speech server is active. Speech bridges bind to localhost and implement the small `/v1/audio/speech` interface used by the app.

## Kokoro

1. Run `SETUP_KOKORO.bat` once.
2. Run `START_KOKORO.bat` and leave its window open.
3. Keep `tts_backend` set to `kokoro` in `config.json`.
4. Select one of the voice names supported by your installed Kokoro release in the `kokoro` backend entry.

Kokoro does not require a personal reference recording.

## Chatterbox Turbo

The Chatterbox bridge is included for advanced users, but its large Torch/CUDA dependencies are not installed by the main setup and GPU package choice varies by hardware.

Create a separate `.venv-chatterbox`, install Chatterbox according to its current upstream instructions, and run:

```powershell
.\.venv-chatterbox\Scripts\python.exe .\bridges\chatterbox_bridge.py
```

Place a clean reference WAV that you own or have permission to use in `voices`, then set its path in the Chatterbox backend and `director_voices.json`. No reference voice is distributed with Gwendolyn.
