# Using Gwendolyn with Vector

## Install Vector

1. Download and extract [Vector 1A v1.6.0-alpha77](https://github.com/Nephet2/restim-vector-live/releases/tag/v1.6.0-alpha77) or a newer compatible release.
2. Start Vector and complete its normal ReStim/device setup.
3. Enable the local Director API in Vector. The supplied Gwendolyn configuration expects `http://127.0.0.1:11436`.
4. In Vector, confirm your limits and Neutral/stop controls before allowing any external Director action.
5. To let Gwendolyn use temporary named recipes, enable **Allow curated Director custom events** in the same Vector window. Leave it off if you only want the existing focus, texture, motion and modifier actions.

If Vector uses another port, change `vector_base` in Gwendolyn's `config.json`.

## Start a session

Use this order so each dependency is ready before Gwendolyn checks it:

1. Start the local model server.
2. Start the optional speech bridge.
3. Start Vector, ReStim and any device connection you intend to use.
4. Run `CHECK_SETUP.bat` and confirm the Model server and Vector Director API lines.
5. Run `START_GWENDOLYN_VECTOR.bat`.

Gwendolyn reads `GET /v1/state` and uses only the bounded Director actions Vector publishes. It does not connect directly to your device. A failed or unavailable API should be reported rather than treated as a successful action.

## First-session commissioning

- Begin with Vector neutral and physical output low.
- Use text input first and ask Gwendolyn to describe the current Vector state.
- Make one harmless focus or tempo request and verify that Vector shows the same change.
- If custom events are enabled, first request a short Calm or Tranquil event and confirm that Vector reports it as active and then returns to the underlying signal.
- Test the spoken or typed Neutral/stop path.
- Add microphone and speech output after the control path is confirmed.
- Leave **Autonomous Vector: Gwendolyn generates incoming motion** off until you understand the reactive workflow. The public defaults keep it off.

## Troubleshooting

- **Vector API is not reachable:** check that Vector is open, its Director API is enabled and the port matches `vector_base`.
- **Gwendolyn describes a change but Vector did not change:** inspect the console launcher and Vector status. Treat Vector's visible state as authoritative.
- **A custom event is rejected:** confirm that alpha77 or newer is running and **Allow curated Director custom events** is enabled. Events outside the curated catalogue are intentionally rejected.
- **Model replies with JSON or tool syntax:** use a model with stronger tool calling, or fix the llama.cpp chat template and `--jinja` configuration.
- **Voice input does not work:** select the correct microphone in Gwendolyn and review the console for audio-device errors. Text input remains available.
