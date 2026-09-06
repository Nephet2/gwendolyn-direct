# Gwendolyn v0.46 — Offline Signal Director

This experimental variant is copied from the preserved v0.45 Director Personalities baseline.

## Added

- Three model tools for the localhost-only Signal Lab bridge:
  - read offline state;
  - apply independent Stairway and Moaner semantic states;
  - offline neutral.
- Signal-specific tool routing so Signal Lab requests do not compete with Vector tools.
- Explicit Signal Lab language bypasses the legacy deterministic Vector parser, preventing Moaner/Stairway intensity descriptions from being misrouted as Vector spatial-gain commands.
- Grounded requested/executed/failed logging through the existing action ledger.
- Explicit offline/no-audio facts in tool results and narration guidance.
- Awareness of Signal Lab v0.4 operator-owned presets, cascaded secondary AM and optional FM movement controls.
- Variant-safe TTS autostart: shared Chatterbox/Kokoro environments and bridge scripts are resolved from the main application runtime while profiles, memory and configuration remain isolated in v0.46.

## Unchanged

- The production v0.45 folder and launchers.
- Vector/ReStim controls.
- Persistent memory and Director personalities.
- LLM, STT and TTS configuration.

This version does not synthesize live audio and cannot select an audio device.

## Local environment note

At build verification time, the pre-existing `.venv`, `.venv-chatterbox`, and `.venv-kokoro` launchers referenced a removed `Python312` installation. The new launcher detects this and displays a persistent explanation instead of opening and closing. The environments were not modified by this release.
