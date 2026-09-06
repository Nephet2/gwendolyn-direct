# Gwendolyn Direct Voice v0.44 — Persistent Memory

This release adds local, cross-session preference learning without storing or replaying complete conversations.

## Memory levels

- **Evidence:** Ordinary positive or negative feedback is stored with the active Vector semantic state, recent narrative theme and selected personality. It does not instruct the model.
- **Learned:** Matching positive evidence must occur across three separately launched sessions before it becomes a durable preference.
- **Pinned:** The Memory page can promote selected evidence or learned preferences into an operator-confirmed permanent memory.

Saying “remember that …” creates a high-confidence learned preference immediately. Saying “what do you remember?” produces an honest summary of durable memory.

## Review and control

The new Memory page lists durable preferences and low-confidence evidence separately, including personality scope and confidence. Selected items can be pinned or forgotten. All data remains local in `gwendolyn_memory.json` and is written atomically.

## Guard against accidental learning

A generic remark such as “that feels good” may still be used naturally to elicit a response. It creates evidence only; repetition within the same session cannot promote it. Automatic promotion requires the same contextual combination to recur across three different application launches.

## Model grounding

Only learned or pinned preferences are included in the model prompt. Raw evidence is never presented as fact, and the model is explicitly told not to claim memory of unlisted sessions or dialogue.
