# Gwendolyn Direct Voice v0.43.6

Uncensored Qwen3.8 27B Q3_K_M is now the preferred production model, using the matched NGL54, 6144-context and q4 KV-cache runtime that won the A/B trial.

## Maintenance changes

- Descriptive sensation reports containing anatomical terms no longer trigger deterministic focus changes. Named focus fast-paths now require command or proposal language.
- Natural approval such as “That sounds like a good idea” executes the pending proposal without asking twice.
- State Arc recovery waits at least 90 seconds after the latest excursion.
- Positive feedback extends the current setting for another 120 seconds before automatic recovery is considered.
- Natural closing phrases such as “I’ll see you later,” “goodnight,” and “I’m going to bed” enter the existing confirmed finish flow. Vector still stops only after explicit confirmation.
- Timeline-grounding replacements now pass through the near-duplicate response guard.
- Added a dedicated production launcher for the winning model.

## Preserved fallback

The Base 27B Q3_K_XL benchmark remains archived as the reasoning/reference fallback.
