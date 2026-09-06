# Gwendolyn Direct v0.46.10-alpha2

This alpha adds support for Vector 1A alpha77's curated temporary custom events.

## Custom events

Gwendolyn can request one of ten Vector-owned recipes when the operator has enabled
**Allow curated Director custom events** in Vector: Tease, Throb, Calm, Intensity build,
Release, Tranquil, Pulse wobble, Pulse-frequency shift, Pulse-width shift and Volume
shift. She may select a 2–30 second duration or use the recipe default.

The active and recent event state is included in her live context. Successful actions
are described naturally, while event names are never treated as proof of the user's
physical sensation. A separate action cancels every active event and returns to the
underlying Vector signal.

Vector remains the enforcement boundary: it owns the allowlist, reduced amplitudes,
duration validation, output clamping, opt-in and immediate cancellation. Neutral and
Stop clear active events.

## Compatibility and validation

Use Vector 1A v1.6.0-alpha77 or newer for custom events. Other Gwendolyn functions remain
compatible with alpha76. The full Gwendolyn suite passes 159 tests.
