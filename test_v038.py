import io
import inspect
import sys
import threading
import time
import types
import unittest
import wave
from pathlib import Path

sys.modules.setdefault("sounddevice", types.SimpleNamespace())
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
fake_whisper = types.ModuleType("faster_whisper")
fake_whisper.WhisperModel = object
sys.modules.setdefault("faster_whisper", fake_whisper)

import gwendolyn_direct as gd


def bare_core():
    core = gd.GwendolynCore.__new__(gd.GwendolynCore)
    core.conductor = gd.SessionConductor([], time.monotonic(), lambda *a: None)
    core.director_thread = threading.local()
    return core


def wav_bytes(seconds=.10, rate=24000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b"\x01\x00" * int(rate * seconds))
    return buffer.getvalue()


class FakeResponse:
    headers = {"content-type": "audio/wav"}
    content = wav_bytes()

    def __enter__(self): return self
    def __exit__(self, *_): return False
    def raise_for_status(self): return None


class FakeTTS:
    def __init__(self): self.requests = []
    def backend(self):
        return types.SimpleNamespace(key="chatterbox", label="test", voice="test", sample_rate=24000)
    def request(self, text, voice_id, idx, timeout):
        self.requests.append((idx, time.perf_counter()))
        time.sleep(.05)
        return FakeResponse()


class FakeOutput:
    def write(self, data):
        time.sleep((len(data) // 2) / 24000.0)


class FakeOllamaResponse:
    def raise_for_status(self): return None
    def json(self):
        return {"message": {"role": "assistant", "content": "Hello."}}


class FakeOllamaSession:
    def __init__(self): self.payload = None
    def post(self, url, json, timeout):
        self.payload = json
        return FakeOllamaResponse()


class V038Tests(unittest.TestCase):
    def test_change_receipts_require_successful_uninterrupted_playback(self):
        for outcome in ('success', 'barge-in', 'tts-failure'):
            with self.subTest(outcome=outcome):
                core = bare_core()
                core.cfg = {'audio_edge_fade_ms': 0, 'audio_block_ms': 20,
                            'tts_chunk_pause_ms': 0, 'audio_tail_ms': 0}
                core.profile = {'speech_chunking': 'Off', 'speech_verbosity': 4}
                core.tts = FakeTTS()
                core.stop_audio = threading.Event()
                core.stt_ready = threading.Event()
                core.voice_generation = 0
                core.voice_generation_lock = threading.Lock()
                core.set_state = lambda *a: None
                core.emit = lambda *a: None
                text = 'I’ve changed the texture to Smooth.'
                core.conductor.announce(text)
                receipt = core.conductor.receipt(text)
                def write(data):
                    if outcome == 'barge-in':
                        core.voice_generation = 1
                core._ensure_output_stream = lambda: types.SimpleNamespace(write=write)
                if outcome == 'tts-failure':
                    def fail(*a):
                        raise RuntimeError('test synthesis failure')
                    core._fetch_tts_pcm = fail
                core._speak_now(text, gd.Timings(), 'RECEIPT', 0, receipt)
                self.assertEqual(bool(core.conductor.pending_text()), outcome != 'success')

    def test_phase_specific_ollama_options_are_defined_and_sent(self):
        core = bare_core()
        core.cfg = {"ollama_model": "test", "ollama_base": "http://test",
                    "ollama_keep_alive": "1m", "ollama_timeout_sec": 12,
                    "ollama_connect_timeout_sec": 2}
        core.ollama_session = FakeOllamaSession()
        core.ollama_session_lock = threading.Lock()
        result = core.ollama_chat([{"role": "user", "content": "Hello"}], phase="proactive")
        self.assertEqual(result["content"], "Hello.")
        self.assertEqual(core.ollama_session.payload["options"],
                         {"temperature": 0.9, "num_predict": 150})

    def test_sanitizer_removes_reasoning_prefix_and_duplicate_paragraph(self):
        raw = "old leaked material</think>Fresh line.\n\nFresh line.\n\nSecond line."
        self.assertEqual(gd.GwendolynCore._sanitize_model_text(raw),
                         "Fresh line.\n\nSecond line.")

    def test_prefetch_requests_next_chunk_during_current_playback(self):
        core = bare_core()
        core.cfg = {"audio_edge_fade_ms": 0, "audio_block_ms": 20,
                    "tts_chunk_pause_ms": 0, "audio_tail_ms": 0}
        core.profile = {"speech_chunking": "Off", "speech_verbosity": 4,
                        "user_name": "the user", "speech_pace": "Deliberate"}
        core.tts = FakeTTS()
        core.stop_audio = threading.Event()
        core.stt_ready = threading.Event()
        core.voice_generation = 0
        core.voice_generation_lock = threading.Lock()
        core.set_state = lambda *_: None
        core.emit = lambda *_: None
        core._ensure_output_stream = lambda: FakeOutput()
        text = " ".join([
            "This is a complete sentence with enough detail to sound natural and deliberate."
            for _ in range(7)
        ])
        core._speak_now(text, gd.Timings(), "TEST", 0)
        self.assertGreaterEqual(len(core.tts.requests), 3)
        # Chunk one takes about 100 ms to play. Request two starts immediately
        # after chunk one is synthesized, while that playback is underway.
        self.assertLess(core.tts.requests[1][1] - core.tts.requests[0][1], .09)

    def test_chunker_prefers_complete_sentences(self):
        text = " ".join(["A natural sentence carries one complete thought." for _ in range(8)])
        chunks = gd.semantic_tts_chunks(text)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.endswith(".") for chunk in chunks))

    def test_named_focus_beats_directional_gain_word(self):
        self.assertEqual(
            gd.direct_vector_action("Drop the top focus to Lower Shaft Focus"),
            ("vector_set_top_focus", {"top_focus": "Lower Shaft Focus"}),
        )
        self.assertEqual(
            gd.direct_vector_action("Widen the focus toward the perineum"),
            ("vector_set_bottom_focus", {"bottom_focus": "Perineum Focus"}),
        )

    def test_sensation_report_is_not_mistaken_for_focus_command(self):
        self.assertIsNone(gd.direct_vector_action(
            "It's feeling very nice pushing against my prostate."))
        self.assertIsNone(gd.direct_vector_action(
            "The current focus on my prostate feels excellent."))
        self.assertEqual(
            gd.direct_vector_action("Please move the bottom focus to Prostate Focus"),
            ("vector_set_bottom_focus", {"bottom_focus": "Prostate Focus"}),
        )

    def test_relaxing_timeline_blocks_invented_rise(self):
        core = bare_core()
        core.last_live_state = {"timeline": {"now": {"energy_band": "relaxing"}}}
        reply = core._enforce_authoritative_energy("The script is getting faster and more demanding now.")
        self.assertIn("genuine lull", reply)
        self.assertNotIn("faster", reply)

    def test_silent_proactive_exits_restore_ready_state(self):
        source = inspect.getsource(gd.GwendolynCore._run_proactive_proposal)
        suppression_paths = [
            'NARRATIVE proactive quiet beat; remaining silent',
            'PROACTIVE suppressed: spoken proposal had no deterministic executable action',
            'PROACTIVE suppressed repeated/no-op action',
            'PROACTIVE suppressed by repetition guard',
        ]
        for marker in suppression_paths:
            tail = source[source.index(marker):]
            return_block = tail[:tail.index('return')]
            self.assertIn('self.set_state("READY")', return_block, marker)

    def test_hardware_map_is_authoritative_prompt_context(self):
        prompt = gd.build_system_prompt(gd.DEFAULT_PROFILE, gd.DEFAULT_PRIVATE_LEXICON)
        self.assertIn("AUTHORITATIVE ELECTRODE HARDWARE MAP", prompt)
        self.assertIn("Top electrode", prompt)
        self.assertIn("Describe the operator's top electrode layout", prompt)

    def test_control_context_excludes_active_semantic_values(self):
        core = bare_core()
        core.profile = gd.DEFAULT_PROFILE
        core.current_vector_semantic = {
            "top_focus": "Glans Focus", "bottom_focus": "Prostate Focus",
            "texture": "Smooth", "variation": "Still",
        }
        core.last_live_state = {"engine_state": "Running"}
        snapshot = core.director_control_context(core.last_live_state)
        self.assertEqual(snapshot["active"]["top_focus"], "Glans Focus")
        self.assertNotIn("Glans Focus", snapshot["meaningful_alternatives"]["top_focus"])
        self.assertNotIn("Smooth", snapshot["meaningful_alternatives"]["texture"])

    def test_informal_proposal_language_is_detected(self):
        reply = "I'm thinking we tighten the top spatial gain down. Does that sound right?"
        self.assertTrue(gd.has_proposal_cue(reply))
        self.assertEqual(gd.direct_vector_action(reply), ("vector_top_spatial_gain", {"action": "decrease"}))

    def test_public_voice_supplies_identity_and_personality_prompt(self):
        core = bare_core()
        core.cfg = {"director_voice": "gwendolyn"}
        core.profile = dict(gd.DEFAULT_PROFILE)
        core.private_lexicon = dict(gd.DEFAULT_PRIVATE_LEXICON)
        self.assertEqual(core.director_name(), "Gwendolyn")
        prompt = core.build_effective_system_prompt()
        self.assertIn("You are Gwendolyn", prompt)
        self.assertIn("playfully theatrical", prompt)

    def test_voice_director_profile_applies_without_mutating_shared_profile(self):
        core = bare_core()
        core.cfg = {"director_voice": "gwendolyn", "director_personality_profile": True}
        core.profile = dict(gd.DEFAULT_PROFILE)
        core.private_lexicon = dict(gd.DEFAULT_PRIVATE_LEXICON)
        effective = core.effective_director_profile()
        self.assertEqual(effective["speech_verbosity"], 4)
        self.assertEqual(effective["narrative_arc"], "Automatic")
        self.assertEqual(core.profile["speech_verbosity"], gd.DEFAULT_PROFILE["speech_verbosity"])

    def test_noop_recovery_is_local_and_personality_specific(self):
        core = bare_core()
        core.cfg = {"director_voice": "gwendolyn"}
        core.recent_spoken_replies = []
        core.last_proposal_rejected_noop = True
        core.last_rejected_noop_action = ("vector_set_top_focus", {"top_focus": "Glans Focus"})
        core.ollama_chat = lambda *args, **kwargs: self.fail("no-op recovery must not call the LLM")
        reply, captured = core._reroll_after_noop([], core.last_rejected_noop_action)
        self.assertTrue(reply)
        self.assertFalse(captured)

    def test_grounding_blocks_unsupported_imminent_transition(self):
        core = bare_core()
        core.cfg = {"director_voice": "anna"}
        core.last_live_state = {"timeline": {"now": {"energy_band": "testing"}}}
        reply = core._enforce_grounded_language("The script is about to surge in just a few seconds.")
        self.assertNotIn("about to", reply)
        self.assertIn("hold steady", reply)

    def test_grounding_removes_unreported_shift_claim(self):
        core = bare_core()
        core.cfg = {"director_voice": "gwendolyn"}
        core.last_live_state = {}
        reply = core._enforce_grounded_language("I want you to feel that shift fully.")
        self.assertNotIn("feel that shift", reply.lower())

    def test_optimized_27b_prompt_preserves_character_hardware_and_expression(self):
        prompt = gd.build_optimized_27b_prompt(gd.DEFAULT_PROFILE, gd.DEFAULT_PRIVATE_LEXICON)
        self.assertLess(len(prompt), 9000)
        self.assertIn("Top electrode", prompt)
        self.assertIn("grounding rules constrain control claims", prompt)
        self.assertIn("direct, explicit", prompt)
        self.assertNotIn("Core rules:", prompt)

    def test_public_model_guide_covers_both_backends(self):
        root = Path(gd.__file__).resolve().parent
        guide = (root / "docs" / "MODEL_SERVER_SETUP.md").read_text(encoding="utf-8")
        self.assertIn("Ollama", guide)
        self.assertIn("llama.cpp", guide)
        self.assertIn("--jinja", guide)

    def test_natural_approval_and_implicit_focus_claims_are_grounded(self):
        self.assertTrue(gd.GwendolynCore._looks_like_approval("That sounds really good"))
        self.assertTrue(gd.GwendolynCore._looks_like_approval("Sounds very tempting"))
        self.assertTrue(gd.GwendolynCore._contains_unexecuted_action_claim(
            "The shift away from pinpoint glans intensity lets the current spread down the shaft."))
        self.assertTrue(gd.GwendolynCore._contains_unexecuted_action_claim(
            "It’s working exactly as intended, spreading that heavy hum down the shaft."))
        self.assertTrue(gd.GwendolynCore._looks_like_approval("I love that thought, please proceed"))
        self.assertTrue(gd.GwendolynCore._contains_unexecuted_action_claim(
            "I’m shifting the top focus to Glans Focus."))
        self.assertTrue(gd.GwendolynCore._contains_unexecuted_action_claim(
            "The change is immediate: the pressure has moved to the glans."))
        self.assertTrue(gd.GwendolynCore._proposal_mentions_multiple_actions(
            "Let's shift the top to Glans Focus and ramp the texture to Rough."))
        self.assertTrue(gd.GwendolynCore._proposal_mentions_multiple_actions(
            "Let's shift the top to Glans Focus and bump spatial gain to 120%."))
        self.assertFalse(gd.GwendolynCore._proposal_mentions_multiple_actions(
            "Let's shift the top to Glans Focus while keeping the prostate steady."))

    def test_noop_resolution_preserves_nonproposal_conversation(self):
        core = bare_core()
        core.cfg = {"director_voice": "gwendolyn"}
        core.recent_spoken_replies = []
        core.last_proposal_rejected_noop = True
        core.last_rejected_noop_action = ("vector_set_top_focus", {"top_focus": "Glans Focus"})
        draft = "I like the way you are moving against it. Shall I shift the top to Glans Focus?"
        reply, captured = core._reroll_after_noop([], core.last_rejected_noop_action, draft)
        self.assertEqual(reply, "I like the way you are moving against it.")
        self.assertFalse(captured)

    def test_local_hardware_stt_corrections_and_full_stroke_routing(self):
        self.assertEqual(gd.normalize_stt_text("The stay away to heaven feels good."),
                         "the Stairway to Heaven feels good.")
        self.assertEqual(gd.normalize_stt_text("Full sweep on the Mona."),
                         "Full sweep on the Moaner.")
        self.assertEqual(gd.direct_vector_action("Full stroke A B C B A"),
                         ("vector_set_bottom_focus", {"bottom_focus": "Bottom Full"}))
        self.assertEqual(gd.direct_vector_action("Full sweep through the Moaner"),
                         ("vector_set_bottom_focus", {"bottom_focus": "Bottom Full"}))

    def test_additional_approval_and_execution_claim_language(self):
        self.assertTrue(gd.GwendolynCore._looks_like_approval("That would be nice"))
        self.assertTrue(gd.GwendolynCore._looks_like_approval(
            "That sounds like a good idea, thank you."))
        self.assertTrue(gd.GwendolynCore._contains_unexecuted_action_claim(
            "I have adjusted the Stairway gain down to 80%."))

    def test_recovery_waits_for_dwell_and_positive_feedback(self):
        core = bare_core()
        core.cfg = {"state_arc_minimum_dwell_seconds": 90}
        core.intervention_count = 4
        core.intervention_target = 4
        core.last_excursion_change_at = time.monotonic()
        core.recovery_snooze_until = 0.0
        core._refresh_authoritative_state = lambda *args, **kwargs: self.fail(
            "Recovery must not inspect or change Vector during minimum dwell")
        self.assertIsNone(core._deterministic_recovery_action())
        self.assertTrue(core._looks_like_positive_feedback("It feels very nice."))

    def test_natural_farewells_begin_confirmed_finish_flow(self):
        self.assertTrue(gd.GwendolynCore._looks_like_finish_request(
            "That sounds delicious. I'll see you later. Thank you so much."))
        self.assertTrue(gd.GwendolynCore._looks_like_finish_request(
            "I'm going to bed now."))

    def _memory_core(self):
        core = bare_core()
        core.cfg = {"persistent_memory_enabled": True, "director_voice": "natasha"}
        core.memory = {"version": 1, "preferences": [], "evidence": []}
        core.memory_lock = threading.Lock()
        core.memory_session_id = "session-1"
        core.current_vector_semantic = {"top_focus": "Glans Focus", "bottom_focus": "Prostate Focus"}
        core.history = [{"role": "assistant", "content": "The slow rhythm is relentless."}]
        core.emit = lambda *args, **kwargs: None
        core._save_memory_and_refresh = lambda: None
        return core

    def test_generic_praise_is_evidence_not_immediate_preference(self):
        core = self._memory_core()
        core.observe_preference_evidence("That feels really good.")
        core.observe_preference_evidence("That feels really good.")
        self.assertEqual(len(core.memory["evidence"]), 1)
        self.assertEqual(len(core.memory["evidence"][0]["sessions"]), 1)
        self.assertEqual(core.memory["preferences"], [])

    def test_repeated_context_across_three_sessions_promotes_memory(self):
        core = self._memory_core()
        for session_id in ("session-1", "session-2", "session-3"):
            core.memory_session_id = session_id
            core.observe_preference_evidence("That feels really good.")
        self.assertEqual(len(core.memory["preferences"]), 1)
        self.assertEqual(core.memory["preferences"][0]["status"], "learned")
        self.assertEqual(core.memory["preferences"][0]["scope"], "gwendolyn")

    def test_explicit_teaching_creates_high_confidence_memory(self):
        core = self._memory_core()
        item = core.remember_explicitly("I prefer slow, rhythmic passages")
        self.assertEqual(item["status"], "learned")
        self.assertGreater(item["confidence"], 0.95)
        self.assertEqual(item["source"], "explicit")

    def test_public_director_has_behavioral_policy(self):
        selected = gd.DIRECTOR_VOICES["gwendolyn"]
        self.assertEqual(selected["operating_profile"]["archetype"], "The Conductor")
        self.assertIn("anti_caricature", selected["operating_profile"])
        self.assertTrue(selected.get("beat_bias"))
        self.assertTrue(selected.get("opening"))

    def test_personality_policy_enters_prompt_and_changes_initiative(self):
        core = bare_core()
        core.cfg = {"director_voice": "gwendolyn", "director_personality_profile": True, "optimized_27b_prompt": True}
        core.profile = dict(gd.DEFAULT_PROFILE)
        core.profile["proactive_director"] = "5 min"
        core.private_lexicon = dict(gd.DEFAULT_PRIVATE_LEXICON)
        core.memory = {"version": 1, "preferences": [], "evidence": []}
        prompt = core.build_effective_system_prompt()
        self.assertIn("The Conductor", prompt)
        self.assertIn("continuity cues", prompt)
        self.assertAlmostEqual(core.proactive_interval_seconds(), 300.0)

    def test_session_finish_requires_confirmation_and_stops_before_farewell(self):
        core = bare_core()
        core.session_finish_pending = False
        core.pending_proposal_text = ""
        core.pending_proposal_action = None
        core.pending_proposal_at = 0.0
        core.last_action_ledger = {"requested": [], "executed": [], "failed": []}
        core.history = []
        core.session_journal_lock = threading.Lock()
        core.session_journal = []
        core.session_journal_started_monotonic = time.monotonic()
        events = []
        core.trim_history = lambda: None
        core.next_voice_tx = lambda prefix: prefix
        core.emit = lambda kind, value: events.append((kind, value))
        core.enqueue_voice = lambda reply, timings, voice_id: events.append(("voice", reply))
        core.set_state = lambda state: events.append(("state", state))
        core.execute_tool = lambda name, args: events.append(("tool", name)) or {"ok": True}

        self.assertTrue(core._handle_session_finish("Thank you for the session", gd.Timings()))
        self.assertTrue(core.session_finish_pending)
        self.assertFalse(any(item[0] == "tool" for item in events))

        events.clear()
        self.assertTrue(core._handle_session_finish("The session felt balanced.", gd.Timings()))
        self.assertTrue(core.session_finish_pending)
        events.clear()
        self.assertTrue(core._handle_session_finish("Yes, I received it.", gd.Timings()))
        self.assertFalse(core.session_finish_pending)
        tool_index = next(i for i, item in enumerate(events) if item[0] == "tool")
        farewell_index = next(i for i, item in enumerate(events) if item[0] == "assistant")
        self.assertLess(tool_index, farewell_index)
        self.assertIn("until next time", events[farewell_index][1].lower())

    def test_session_finish_can_be_cancelled_without_stopping_vector(self):
        core = bare_core()
        core.session_finish_pending = True
        core.history = []
        core.trim_history = lambda: None
        core.next_voice_tx = lambda prefix: prefix
        core.emit = lambda *args: None
        core.enqueue_voice = lambda *args, **kwargs: None
        core.execute_tool = lambda *args, **kwargs: self.fail("Vector must not stop when finish is cancelled")
        self.assertTrue(core._handle_session_finish("Not yet, keep going", gd.Timings()))
        self.assertFalse(core.session_finish_pending)


if __name__ == "__main__":
    unittest.main()
