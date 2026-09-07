import ast
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "gwendolyn_direct.py").read_text(encoding="utf-8")


class SignalDirectorVariantTests(unittest.TestCase):
    def test_source_parses(self):
        ast.parse(SOURCE)

    def test_version_and_tools_are_present(self):
        self.assertIn('APP_VERSION = "0.46.11-alpha1"', SOURCE)
        self.assertIn('def signal_lab_preset_start_args', SOURCE)
        self.assertIn('conducted_preset_baseline_confirmed', SOURCE)
        self.assertIn('Signal Lab did not verify the active preset baseline', SOURCE)
        self.assertIn('def vector_heart_tempo_tick', SOURCE)
        self.assertIn('http://127.0.0.1:18767/v1/heart-tempo', SOURCE)
        self.assertIn('def restim_sensor_checkin_tick', SOURCE)
        self.assertIn('"restim_sensor_response": self.vector_sensor_snapshot()', SOURCE)
        self.assertIn('data["restim_sensor_response"] = self.restim_sensor_snapshot()', SOURCE)
        self.assertIn('SESSION_MODE == "signal_lab" and self.conducted_session_active', SOURCE)
        self.assertIn('self._journal_event(f"{SESSION_MODE}_sensor_feedback"', SOURCE)
        self.assertIn('self.core.restim_sensor_checkin_tick()', SOURCE)
        self.assertIn('Do not repeatedly ask them to classify routine readings.', SOURCE)
        for name in ("signal_lab_get_state", "signal_lab_apply", "signal_lab_heart_tempo", "signal_lab_neutral"):
            self.assertIn(f'"name":"{name}"', SOURCE)

    def test_curated_vector_events_are_bounded_and_grounded(self):
        self.assertIn('"name":"vector_trigger_event"', SOURCE)
        self.assertIn('"name":"vector_cancel_events"', SOURCE)
        self.assertIn('"minimum":2,"maximum":30', SOURCE)
        self.assertIn('"/v1/event/trigger"', SOURCE)
        self.assertIn('"/v1/event/cancel"', SOURCE)
        self.assertIn('"custom_events": s.get("custom_events")', SOURCE)
        self.assertIn('never imply that its name proves a particular physical sensation', SOURCE)

    def test_director_voices_are_loaded_from_editable_json(self):
        voices_path = ROOT / "director_voices.example.json"
        voices = json.loads(voices_path.read_text(encoding="utf-8"))
        self.assertEqual({"gwendolyn"}, set(voices))
        for key, profile in voices.items():
            for field in ("label", "name", "reference", "personality", "defaults"):
                self.assertIn(field, profile, f"{key} missing {field}")
            self.assertEqual(profile["reference"], "", key)
        self.assertIn("DIRECTOR_VOICES_PATH = ROOT / \"director_voices.json\"", SOURCE)
        self.assertIn("DIRECTOR_VOICES = load_director_voices()", SOURCE)
        self.assertIn('DEFAULT_DIRECTOR_VOICES = {', SOURCE)

    def test_public_voice_profile_has_no_bundled_recording(self):
        voices = json.loads((ROOT / "director_voices.example.json").read_text(encoding="utf-8"))
        self.assertEqual(voices["gwendolyn"]["name"], "Gwendolyn")
        self.assertEqual(voices["gwendolyn"]["reference"], "")

    def test_signal_routing_precedes_vector_routing(self):
        self.assertIn("def is_signal_lab_intent", SOURCE)
        self.assertIn("if is_signal_lab_intent(raw):", SOURCE)
        self.assertIn('and is_signal_lab_intent(f"{pending_before_turn} {text}")', SOURCE)

    def test_natural_signal_control_phrases_are_recognised(self):
        tree = ast.parse(SOURCE)
        parser_node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "is_signal_lab_intent"
        )
        namespace = {"re": re}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[parser_node], type_ignores=[])),
                     "<signal-intent>", "exec"), namespace)
        parser = namespace["is_signal_lab_intent"]
        for phrase in (
            "Can you make it a lot gentler?",
            "Reduce both electrodes to something very low and gentle.",
            "Pull it down completely and start again.",
            "Make the other lane softer.",
            "Give me a single slow wave.",
        ):
            self.assertTrue(parser(phrase), phrase)

    def test_signal_truth_has_user_priority_and_live_playback_gates(self):
        self.assertIn("self.signal_user_revision += 1", SOURCE)
        self.assertIn("CONDUCTED SESSION cancelled before apply because a newer user turn has priority", SOURCE)
        self.assertIn('if not live.get("playing") or not monitor.get("audible"):', SOURCE)
        self.assertIn("AUDIBLE LIVE STATE AFTER THE VERIFIED CHANGE", SOURCE)

    def test_q6_polish_protects_control_and_receipts(self):
        self.assertIn("def deterministic_signal_relief", SOURCE)
        self.assertIn("signal_relief_direct = (", SOURCE)
        self.assertIn("protected_action_receipt", SOURCE)
        self.assertIn("def _repeated_signal_template", SOURCE)
        self.assertIn("I won’t invent a confirmation value", SOURCE)
        self.assertIn("GROUNDING removed invented Signal Lab physical-output claim", SOURCE)

    def test_state_claims_without_execution_are_detected(self):
        tree = ast.parse(SOURCE)
        core = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GwendolynCore")
        method = next(node for node in core.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_contains_unexecuted_action_claim")
        mini = ast.ClassDef(name="Mini", bases=[], keywords=[], body=[method], decorator_list=[])
        namespace = {"re": re}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[mini], type_ignores=[])),
                     "<claim-gate>", "exec"), namespace)
        detector = namespace["Mini"]._contains_unexecuted_action_claim
        self.assertTrue(detector("Both lanes are now at the floor and neutral."))
        self.assertTrue(detector("The signal has been lowered."))

    def test_heart_tempo_has_direct_and_autonomous_routes(self):
        self.assertIn("def direct_signal_heart_tempo_action", SOURCE)
        self.assertIn("heart_tempo_direct = (", SOURCE)
        self.assertIn('You may optionally include heart_tempo', SOURCE)
        self.assertIn('applied_name = "signal_lab_heart_tempo"', SOURCE)
        self.assertIn('direct_signal_heart_tempo_action(f"{pending_before_turn} {text}")', SOURCE)
        self.assertIn('heart_tempo_requested_without_fresh_telemetry', SOURCE)
        self.assertIn('heart_tempo.fresh=true', SOURCE)

    def test_signal_commands_use_controller_owned_step_guard(self):
        self.assertIn('def bound_signal_lab_step', SOURCE)
        self.assertGreaterEqual(SOURCE.count('bound_signal_lab_step('), 3)
        self.assertIn('current_volume * 1.10', SOURCE)
        self.assertIn('current_volume * 0.90', SOURCE)
        self.assertIn('bounded step and modulation-complexity guard applied', SOURCE)

    def test_explicit_signal_request_bypasses_direct_vector_parser(self):
        self.assertIn('SESSION_MODE == "signal_lab" or signal_request', SOURCE)
        self.assertIn('select_tools_for_turn(text, pending_before_turn) if SESSION_MODE == "vector" else []', SOURCE)

    def test_signal_action_uses_fast_grounded_confirmation(self):
        self.assertIn('OLLAMA ROUTE signal-json-primary tools=0', SOURCE)
        self.assertIn('if direct or signal_request or exact_vector_confirmation:', SOURCE)
        self.assertIn('"exact Vector action" if exact_vector_confirmation', SOURCE)

    def test_incomplete_signal_action_is_rejected_and_named_correctly(self):
        self.assertIn('Incomplete Signal Lab command; both lanes and transition are required', SOURCE)
        self.assertIn('That Signal Lab change didn’t take, so its current state is unchanged.', SOURCE)
        self.assertIn('SIGNAL LAB action rejected', SOURCE)

    def test_incomplete_signal_action_gets_compact_repair(self):
        self.assertIn('def signal_lab_apply_args_complete', SOURCE)
        self.assertIn('phase="signal-json-primary"', SOURCE)
        self.assertIn('phase="signal-json-repair"', SOURCE)
        self.assertIn('SIGNAL LAB single schema repair accepted', SOURCE)
        self.assertIn('SIGNAL LAB primary JSON design complete', SOURCE)
        self.assertIn('def extract_signal_lab_args', SOURCE)
        self.assertIn('SIGNAL LAB malformed design rejected after single retry', SOURCE)

    def test_signal_context_suppresses_vector_proposals(self):
        self.assertIn('signal_context_active = SESSION_MODE == "signal_lab"', SOURCE)
        self.assertIn('SIGNAL LAB CONTEXT: Continue discussing the current Signal Lab session.', SOURCE)
        self.assertIn('if not approval_of_pending and not signal_context_active:', SOURCE)

    def test_signal_context_inherits_ambiguous_controls(self):
        self.assertIn('sticky_signal_control = bool(', SOURCE)
        self.assertIn('explicit_vector_request = SESSION_MODE == "vector"', SOURCE)
        self.assertIn('signal_request = explicit_signal_request or sticky_signal_control', SOURCE)
        self.assertIn('SIGNAL LAB sticky context inherited for ambiguous control request', SOURCE)
        self.assertIn('one at a time and bring intensity in last.', SOURCE)

    def test_positive_feedback_holds_signal_state(self):
        self.assertIn('positive_signal_feedback = bool(', SOURCE)
        self.assertIn('SIGNAL LAB positive feedback holds current state; no command generated', SOURCE)
        self.assertIn('signal_control_cue = bool(', SOURCE)

    def test_signal_finish_uses_fixed_session_controller(self):
        self.assertIn('tool_name = "signal_lab_neutral" if signal_mode else "vector_stop"', SOURCE)
        self.assertIn('tool_name = "signal_lab_neutral" if signal_mode else "vector_stop"', SOURCE)
        self.assertIn('session_finish_stage == "awaiting_summary_receipt"', SOURCE)

    def test_signal_json_has_adequate_budget_and_no_executed_fallback(self):
        self.assertIn('"signal-json-primary": 420', SOURCE)
        self.assertIn('configured_cap = max(configured_cap, 420)', SOURCE)
        self.assertNotIn('designed_args = bounded_signal_lab_fallback()', SOURCE)

    def test_signal_memory_uses_signal_state(self):
        self.assertIn('Signal Lab left carrier=', SOURCE)
        self.assertIn('live.get("left_signal")', SOURCE)
        self.assertIn('Signal Lab state temporarily unavailable', SOURCE)

    def test_live_signal_telemetry_overrides_preset_defaults(self):
        self.assertIn('the live object is authoritative', SOURCE)
        self.assertIn('Never substitute active_preset starting values', SOURCE)
        self.assertIn('never collapse unequal lanes into one carrier', SOURCE)
        self.assertIn('"live": result.get("live")', SOURCE)
        self.assertIn('physical_output_enabled', SOURCE)

    def test_state_confirmation_reports_exact_live_lanes(self):
        self.assertIn("The live Signal Lab state is:", SOURCE)
        self.assertIn("lane('Stairway', left)", SOURCE)
        self.assertIn("lane('Moaner', right)", SOURCE)
        self.assertIn("secondaryAmFreq", SOURCE)
        self.assertIn("Physical output remains disabled.", SOURCE)

    def test_signal_lab_voice_activation_defaults_to_normal(self):
        self.assertIn('"voice_activation_sensitivity": "Normal"', SOURCE)

    def test_hermes_handoff_is_explicit_local_and_non_controlling(self):
        self.assertIn("def _handle_hermes_export", SOURCE)
        self.assertIn('"schema": "gwendolyn-hermes-handoff/v2"', SOURCE)
        self.assertIn('"Offline analysis only; this snapshot grants no live-control authority."', SOURCE)
        self.assertIn("temporary.replace(destination)", SOURCE)
        self.assertIn("Hermes has not been launched", SOURCE)
        self.assertIn('if not reply_prefix:', SOURCE)
        self.assertIn('reply += " Nothing live was changed."', SOURCE)
        self.assertIn("if self._handle_hermes_export(text, timings):", SOURCE)

    def test_signal_lab_xbox_ptt_is_mode_specific_and_visible(self):
        self.assertIn('"signal_lab_input_mode": "Xbox PTT"', SOURCE)
        self.assertIn('self.cfg["input_mode"] = str(self.cfg.get("signal_lab_input_mode", "Xbox PTT"))', SOURCE)
        self.assertIn("Xbox LB = hold to talk; release to send", SOURCE)

    def test_signal_lab_ptt_uses_direct_xinput_without_vector(self):
        self.assertIn("class XInputReader", SOURCE)
        self.assertIn('self.xinput = XInputReader() if SESSION_MODE == "signal_lab" else None', SOURCE)
        self.assertIn('state = self.xinput.read()', SOURCE)
        self.assertIn('state = self.core.vector_get("/v1/controller", timeout=0.65)', SOURCE)
        self.assertIn('LB = 0x0100', SOURCE)
        self.assertIn('DPAD = 0x000F', SOURCE)

    def test_signal_lab_bluetooth_ptt_prefers_sdl_with_xinput_fallback(self):
        self.assertIn("class SDLControllerReader", SOURCE)
        self.assertIn("self.controller.start()", SOURCE)
        self.assertNotIn('if SESSION_MODE == "vector":\n            self.controller.start()', SOURCE)
        self.assertIn("LEFT_SHOULDER = 9", SOURCE)
        self.assertIn("DPAD_BUTTONS = (11, 12, 13, 14)", SOURCE)
        self.assertIn('self.gamepad = SDLControllerReader() if SESSION_MODE == "signal_lab" else None', SOURCE)
        self.assertIn('if not state.get("connected") and self.xinput is not None:', SOURCE)
        self.assertIn("state = self.xinput.read()", SOURCE)

    def test_signal_lab_dpad_feedback_contract_is_explicit(self):
        self.assertIn('CONTROLLER_FEEDBACK = {', SOURCE)
        self.assertIn('"up": "That feels good.', SOURCE)
        self.assertIn('"down": "That does not feel good.', SOURCE)
        self.assertIn('"left": "This is too intense.', SOURCE)
        self.assertIn('"right": "Ha—you call this a challenge?', SOURCE)
        self.assertIn('def handle_controller_feedback', SOURCE)
        self.assertIn('"controller_feedback"', SOURCE)
        self.assertIn('target_volume = max(vmin, current_volume * 0.90)', SOURCE)
        self.assertIn('"transition_seconds": 5', SOURCE)
        self.assertIn('self.last_conducted_decision_at = time.monotonic()', SOURCE)

    def test_dpad_is_debounced_and_does_not_conflict_with_lb_ptt(self):
        self.assertIn('"dpad_direction": direction', SOURCE)
        self.assertIn('not lb and dpad_direction', SOURCE)
        self.assertIn('now - self.last_dpad_at >= 0.35', SOURCE)
        self.assertIn('if lb and combo:', SOURCE)

    def test_signal_lab_shutdown_precedes_export_and_conducted_start(self):
        self.assertIn("def is_signal_lab_shutdown_command", SOURCE)
        self.assertIn("def is_hermes_export_command", SOURCE)
        self.assertIn("def _handle_global_signal_lab_shutdown", SOURCE)
        shutdown = SOURCE.index("if self._handle_global_signal_lab_shutdown(text, timings):")
        export = SOURCE.index("if self._handle_hermes_export(text, timings):")
        conducted = SOURCE.index("if self._handle_conducted_session_command(text, timings):")
        self.assertLess(shutdown, export)
        self.assertLess(export, conducted)
        self.assertIn('self.execute_tool("signal_lab_neutral", {"reason": "Explicit Signal Lab shutdown"})', SOURCE)
        self.assertIn("return self._handle_hermes_export(text, timings, reply_prefix=prefix)", SOURCE)

    def test_send_report_to_hermes_wording_is_recognised(self):
        self.assertIn("|send", SOURCE)
        self.assertIn(r"\breport\b.{0,35}\b(?:to|for)\s+(?:hermes|whom is|her miss)\b", SOURCE)

    def test_shutdown_report_survives_stt_name_error(self):
        self.assertIn('(?:hermes|whom is|her miss)', SOURCE)
        self.assertIn('or re.search(r"\\b(?:send|write|create|prepare)\\b.{0,35}\\breport\\b"', SOURCE)
        self.assertIn('if export_requested:', SOURCE)

    def test_handoff_v2_exports_complete_structured_session_journal(self):
        self.assertIn('"schema": "gwendolyn-hermes-handoff/v2"', SOURCE)
        self.assertIn("def _journal_event", SOURCE)
        self.assertIn("def _journal_copy", SOURCE)
        self.assertIn('"session_journal": self._journal_copy()', SOURCE)
        self.assertIn('"conducted_decision_applied"', SOURCE)
        self.assertIn('"before_state": snapshot', SOURCE)
        self.assertIn('"feedback_context": recent_feedback', SOURCE)
        self.assertIn('"conducted_session_completed"', SOURCE)
        self.assertIn("CONDUCTED SESSION disarmed before confirmed session finish", SOURCE)
        self.assertIn('"session_finish_result"', SOURCE)
        self.assertIn("complete structured session journal", SOURCE)
        self.assertIn('"session_actions": session_actions', SOURCE)
        self.assertIn('"control_action_count": len(session_actions)', SOURCE)

    def test_conducted_session_has_explicit_lifecycle(self):
        self.assertIn("def _handle_conducted_session_command", SOURCE)
        self.assertIn("Conducted Session is armed", SOURCE)
        self.assertIn("CONDUCTED SESSION held", SOURCE)
        self.assertIn("CONDUCTED SESSION resumed", SOURCE)
        self.assertIn("CONDUCTED SESSION disarmed", SOURCE)
        self.assertIn("if self._handle_conducted_session_command(text, timings):", SOURCE)
        self.assertIn("self.proactive_snooze_until = 0.0", SOURCE)
        self.assertIn("self.conducted_start_pending = True", SOURCE)

    def test_conducted_session_accepts_proceed_wording_from_voice(self):
        tree = ast.parse(SOURCE)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "is_conducted_session_start_command")
        namespace = {"re": re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<parser-test>", "exec"), namespace)
        parser = namespace["is_conducted_session_start_command"]
        self.assertTrue(parser("I do please proceed with an autonomous session. You are in charge."))
        self.assertTrue(parser("Go ahead with the conducted session."))
        self.assertFalse(parser("The autonomous session was useful to review."))

    def test_vector_autonomy_requires_explicit_lifecycle(self):
        self.assertIn("def _handle_vector_autonomous_command", SOURCE)
        self.assertIn('(?:start|begin|launch|arm|conduct)', SOURCE)
        self.assertIn('self.vector_autonomous_active = True', SOURCE)
        self.assertIn('"authority": "explicit_user_grant"', SOURCE)
        self.assertIn("if self._handle_vector_autonomous_command(text, timings):", SOURCE)

    def test_vector_autonomy_accepts_launch_wording_from_voice(self):
        tree = ast.parse(SOURCE)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "is_vector_autonomous_start_command")
        namespace = {"re": re}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<parser-test>", "exec"), namespace)
        self.assertTrue(namespace["is_vector_autonomous_start_command"](
            "Can you launch an autonomous vector session, please?"
        ))

    def test_vector_autonomy_keeps_shutdown_actions_user_owned(self):
        self.assertIn('proposal_action[0] in {"vector_stop", "vector_resume", "vector_neutral"}', SOURCE)
        self.assertIn("All other recognised actions remain inside Vector’s own operator-configured bounds", SOURCE)

    def test_vector_finish_disarms_autonomy_before_stop(self):
        self.assertIn('VECTOR AUTONOMOUS disarmed before confirmed session finish', SOURCE)
        self.assertIn('tool_name = "signal_lab_neutral" if signal_mode else "vector_stop"', SOURCE)

    def test_explicit_vector_shutdown_is_global_but_other_cross_controls_stay_blocked(self):
        self.assertIn("def _handle_global_vector_shutdown", SOURCE)
        self.assertIn('name.startswith("vector_") and name != "vector_stop"', SOURCE)
        self.assertIn('if self._handle_global_vector_shutdown(text, timings):', SOURCE)
        self.assertIn('self.execute_tool("vector_stop", {})', SOURCE)
        self.assertIn('"requested_from_session_mode": SESSION_MODE', SOURCE)

    def test_vector_launcher_pins_vector_mode(self):
        launcher = (ROOT / "START_GWENDOLYN_VECTOR.bat").read_text(encoding="utf-8")
        self.assertIn('set "GWENDOLYN_SESSION_MODE=vector"', launcher)
        self.assertIn("pending start acknowledged; rechecking live monitor", SOURCE)

    def test_conducted_session_reuses_bounded_signal_path(self):
        self.assertIn("def _run_conducted_signal_decision", SOURCE)
        self.assertIn('self.execute_tool("signal_lab_apply", args)', SOURCE)
        self.assertIn("signal_lab_apply_args_complete(args)", SOURCE)
        self.assertIn("active_preset", SOURCE)
        self.assertIn("monitor.get(\"audible\")", SOURCE)
        self.assertIn("Conducted Session preset duration complete", SOURCE)
        self.assertIn("duration_due", SOURCE)

    def test_signal_mode_cannot_capture_or_approve_vector_proposal(self):
        self.assertIn('SESSION_MODE == "vector"\n            and self._looks_like_approval(text)', SOURCE)
        self.assertIn("and not signal_context_active and self._contains_unexecuted_action_claim(reply)", SOURCE)

    def test_conducted_interval_is_configured(self):
        config = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        self.assertEqual(180, config["conducted_session_interval_seconds"])

    def test_signal_lab_has_state_aware_idle_presence(self):
        self.assertIn('def _run_signal_lab_presence', SOURCE)
        self.assertIn('target = self._run_signal_lab_presence if SESSION_MODE == "signal_lab"', SOURCE)
        self.assertIn('LIVE SIGNAL LAB STATE NOW:', SOURCE)
        self.assertIn('SIGNAL LAB PRESENCE MOMENT.', SOURCE)
        self.assertIn('Most presence moments must leave the signal unchanged.', SOURCE)
        self.assertIn('self.last_conversation_activity_at = time.monotonic()', SOURCE)
        self.assertIn('VOICE {voice_id} PLAYBACK_END', SOURCE)

    def test_signal_presence_preserves_conversational_length(self):
        self.assertIn('"signal-presence": 240', SOURCE)
        self.assertIn('Normally use two to four expressive sentences', SOURCE)
        self.assertIn('max_chars=650', SOURCE)

    def test_local_stt_polish(self):
        self.assertIn('"current feeling"', SOURCE)

    def test_signal_designer_has_sensory_map_and_current_state(self):
        self.assertIn('texture 0=rough/low carrier', SOURCE)
        self.assertIn('CURRENT SIGNAL LAB STATE:', SOURCE)
        self.assertIn('preserve every current value not explicitly requested', SOURCE)

    def test_narrow_signal_edits_are_preserved_mechanically(self):
        self.assertIn('def preserve_unrequested_signal_dimensions', SOURCE)
        self.assertIn('unrequested dimensions preserved', SOURCE)

    def test_signal_lab_voice_activation_is_hardened(self):
        self.assertIn('initial_prompt=stt_initial_prompt()', SOURCE)
        self.assertIn('def accepted_stt_text', SOURCE)
        self.assertIn('STT low-confidence segment suppressed', SOURCE)
        self.assertIn('def is_repeated_stt_hallucination', SOURCE)
        self.assertIn('STT repeated-word hallucination suppressed', SOURCE)
        self.assertIn('min_rms = max(min_rms, 0.018)', SOURCE)
        self.assertIn('min_rms = max(min_rms, 0.014)', SOURCE)
        self.assertIn('required_voiced_blocks = 6 if SESSION_MODE == "signal_lab" else 4', SOURCE)
        self.assertIn('Signal Lab session reduced High sensitivity to Normal', SOURCE)

    def test_mouse_ptt_release_cannot_be_lost(self):
        self.assertIn('self._mouse_ptt_active = False', SOURCE)
        self.assertIn('self.root.bind_all("<ButtonRelease-1>"', SOURCE)
        self.assertIn('strand Gwendolyn in LISTENING', SOURCE)

    def test_session_brief_has_evidence_based_conductor(self):
        self.assertIn('def _build_session_brief_milestones', SOURCE)
        self.assertIn('def _brief_milestone_complete', SOURCE)
        self.assertIn('SESSION BRIEF CONDUCTOR (higher priority than optional narrative beats)', SOURCE)
        self.assertIn('"completed" if state == "confirmed"', (ROOT / 'session_conductor.py').read_text())
        self.assertIn('CURRENT FLEXIBLE PHASE', SOURCE)
        self.assertIn('Treat a flexible phase as direction rather than a checklist', SOURCE)

    def test_signal_lab_proposals_survive_until_approval(self):
        self.assertIn('self.pending_signal_lab_proposal = ""', SOURCE)
        self.assertIn('def _capture_signal_lab_proposal', SOURCE)
        self.assertIn('approval_of_signal_proposal = bool(', SOURCE)
        self.assertIn('the user approved this proposal:', SOURCE)

    def test_session_mode_is_fixed_and_cross_tools_are_blocked(self):
        self.assertIn('GWENDOLYN_SESSION_MODE', SOURCE)
        self.assertIn('Vector controls are unavailable in this Signal Lab session', SOURCE)
        self.assertIn('Signal Lab controls are unavailable in this Vector session', SOURCE)
        for launcher, mode in (("START_GWENDOLYN_VECTOR.bat", "vector"),):
            text = (ROOT / launcher).read_text(encoding="utf-8")
            self.assertIn(f'GWENDOLYN_SESSION_MODE={mode}', text)
        self.assertIn('.venv\\Scripts\\python.exe', text)

    def test_create_state_is_not_misclassified_as_state_query(self):
        self.assertIn('create|make|design|apply|set|build|new|adjust|increase|decrease|raise|lower', SOURCE)

    def test_raw_signal_telemetry_requires_explicit_technical_ask(self):
        self.assertIn("def asks_for_raw_signal_telemetry", SOURCE)
        self.assertIn("def asks_what_changed_in_signal", SOURCE)
        self.assertIn("raw_signal_telemetry_request", SOURCE)
        self.assertIn("OLLAMA ROUTE signal-lab-summary conversational", SOURCE)
        self.assertIn("Do not dump numbers, raw lane telemetry", SOURCE)
        self.assertIn("def _filter_spoken_signal_telemetry", SOURCE)
        self.assertIn("SIGNAL LAB spoken telemetry filtered", SOURCE)
        self.assertIn("signal_detail_suppression", SOURCE)
        self.assertIn("presentation preference recognised; no control command generated", SOURCE)

    def test_conducted_clock_is_logged_and_survives_completion(self):
        self.assertIn('"clock_source": "session_conductor_started_at"', SOURCE)
        self.assertIn("conducted_session_completed_at", SOURCE)
        self.assertIn("conducted_session_neutral_confirmed", SOURCE)
        self.assertIn("def _authoritative_signal_time_reply", SOURCE)
        self.assertIn("SESSION TIME answered from authoritative conducted clock", SOURCE)

    def test_config_points_only_to_local_bridge(self):
        config = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        self.assertEqual("http://127.0.0.1:18766", config["signal_lab_base"])

    def test_ab_model_override_is_process_local(self):
        self.assertIn('GWENDOLYN_OPENAI_BASE', SOURCE)
        self.assertIn('GWENDOLYN_OPENAI_MODEL', SOURCE)

    def test_openai_messages_support_strict_mistral_alternation(self):
        self.assertIn("def _prepare_openai_messages", SOURCE)
        self.assertIn("PRIOR ASSISTANT OPENING (context only)", SOURCE)
        self.assertIn("strict_alternation=not bool(tools)", SOURCE)
        self.assertIn('LLM A-B override', SOURCE)
        self.assertNotIn('save_config(data)', SOURCE)

    def test_offline_claims_are_explicit(self):
        self.assertIn('"physical_output_enabled": False', SOURCE)
        self.assertIn('it never controls physical output', SOURCE)
        self.assertIn('audible headphone monitor', SOURCE)

    def test_vector_generated_motion_is_explicit_and_bounded(self):
        self.assertIn('"name":"vector_generated_motion_plan"', SOURCE)
        self.assertIn('"minimum":0.0', SOURCE)
        self.assertIn('"maximum":1.0', SOURCE)
        self.assertIn('stroke_duration_ms', SOURCE)
        self.assertIn('duration_seconds (30-600 review horizon)', SOURCE)
        self.assertIn('Do not emit raw T-code samples.', SOURCE)
        self.assertIn('vector_autonomous_generate_tcode', SOURCE)

    def test_vector_generated_motion_follows_autonomous_lifecycle(self):
        self.assertIn('self.execute_tool("vector_generated_motion_hold", {})', SOURCE)
        self.assertIn('self.execute_tool("vector_generated_motion_resume", {})', SOURCE)
        self.assertIn('self.execute_tool("vector_use_authored_tcode", {})', SOURCE)
        self.assertIn('/v1/generated-motion/plan', SOURCE)

    def test_autonomous_changes_are_narrated_while_reactive_confirmations_remain(self):
        self.assertIn('def _autonomous_change_narration', SOURCE)
        self.assertIn('vector-autonomous-narration', SOURCE)
        self.assertIn("This was your autonomous ", SOURCE)
        self.assertIn("decision, so do not say 'Done'", SOURCE)
        self.assertIn('spoken = self._autonomous_change_narration(', SOURCE)
        self.assertIn('return f"Done. I’ve changed the texture', SOURCE)


class SessionConductorBehaviorTests(unittest.TestCase):
    """Exercise real core methods without loading audio, model or physical controls."""
    @classmethod
    def setUpClass(cls):
        import threading
        import time
        from difflib import SequenceMatcher
        from session_conductor import SessionConductor, due_seconds, valid_time, matches
        tree = ast.parse(SOURCE)
        core = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GwendolynCore')
        names = {'_verified_control', '_action_state_key_value', '_semantic_snapshot_from_state',
                 '_semantic_dimension_value', 'execute_tool', '_publish_reply', '_brief_control_authorized',
                 '_run_due_brief_control', '_autonomous_interrupted', '_grounded_confirmation', '_build_session_brief_milestones',
                 '_session_brief_conductor_context', '_brief_milestone_complete',
                 '_finalize_reply', '_sanitize_model_text', '_normalized_reply',
                 '_contains_unexecuted_action_claim', '_contains_unsupported_biometric_claim',
                 '_contains_heart_lock_claim', '_heart_lock_verified_in_state',
                 'handle_controller_feedback',
                 '_near_duplicate_similarity', '_next_repetition_fallback',
                 '_control_state_snapshot', 'compact_state',
                 '_filter_spoken_signal_telemetry', '_prepare_openai_messages'}
        core.body = [n for n in core.body if isinstance(n, ast.FunctionDef) and n.name in names]
        parsers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                   n.name in {'direct_vector_action', 'direct_signal_heart_tempo_action', 'norm_words', 'semantic_top_focus', 'semantic_bottom_focus',
                              '_signal_semantic_baseline', 'bound_signal_lab_step', 'signal_lab_preset_start_args'}]
        ns = dict(re=re, time=time, json=json, threading=threading, SequenceMatcher=SequenceMatcher,
                  SessionConductor=SessionConductor, due_seconds=due_seconds, valid_time=valid_time,
                  matches=matches, SESSION_MODE='vector', log=lambda *a: None, Timings=lambda: None)
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                                 *parsers, core], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<core-behavior>', 'exec'), ns)
        cls.ns = ns
        cls.Core = ns['GwendolynCore']

    def setUp(self):
        import threading
        from session_conductor import SessionConductor
        self.ns['SESSION_MODE'] = 'vector'
        self.events = []
        self.core = self.Core.__new__(self.Core)
        c = self.core
        c.conductor = SessionConductor([], 0, lambda *event: self.events.append(event))
        c.director_thread = threading.local()
        c.current_voice_generation = lambda: 1
        c.recording = False
        c.current_state = 'READY'
        c.history = []
        c.emit = lambda *event: self.events.append(event)
        c.trim_history = lambda: None
        c._profile_fallback = lambda: 'Steady. I’m watching…'
        c.last_action_ledger = {'executed': [], 'failed': []}
        c.vector_autonomous_active = True
        c.vector_autonomous_held = False
        c.conducted_session_active = True
        c.conducted_session_held = False
        c.conducted_session_ends_at = 0
        c.reset_action_ledger = lambda: None
        c._refresh_authoritative_state = lambda **kw: {'engine_state': 'Running'}
        c.enqueue_voice = lambda *a, **kw: None
        c.next_voice_tx = lambda *a: 'test'
        c.cfg = {}
        c.recent_spoken_replies = []
        c.repetition_fallback_index = 0

    def test_dpad_up_records_feedback_without_control(self):
        self.ns['SESSION_MODE'] = 'signal_lab'
        self.ns['CONTROLLER_FEEDBACK'] = {
            'up': 'That feels good. Preserve or develop the current quality.',
            'down': 'That does not feel good.', 'left': 'Too intense.', 'right': 'Challenge.'}
        c = self.core
        c.signal_lab_get = lambda *a, **kw: {'last_command': {}}
        c._journal_event = lambda event, data: self.events.append((event, data))
        c.last_conversation_activity_at = 0.0
        c.handle_controller_feedback('up')
        self.assertTrue(any(event == 'controller_feedback' for event, _ in self.events if isinstance(event, str)))
        self.assertIn('That feels good', c.history[-1]['content'])

    def test_dpad_left_reduces_only_intensity_by_ten(self):
        self.ns['SESSION_MODE'] = 'signal_lab'
        self.ns['CONTROLLER_FEEDBACK'] = {
            'up': 'Good.', 'down': 'Not good.', 'left': 'Too intense.', 'right': 'Challenge.'}
        c = self.core
        semantic = {'intensity': 55, 'texture': 70, 'vibration_rate': 30,
                    'vibration_depth': 40, 'secondary_rate': 5,
                    'secondary_depth': 6, 'modulation': 7}
        state = {
            'active_preset': {'volume_min': 40, 'volume_max': 100},
            'live': {'left_signal': {'volume': 73}, 'right_signal': {'volume': 73}},
            'last_command': {'left_semantic': dict(semantic), 'right_semantic': dict(semantic)}}
        c.signal_lab_get = lambda *a, **kw: state
        c._journal_event = lambda *a: None
        c.last_conversation_activity_at = 0.0
        captured = {}
        c.execute_tool = lambda name, args: captured.update(name=name, args=args) or {'ok': True}
        c._verified_control = lambda *a: True
        c.handle_controller_feedback('left')
        self.assertEqual(captured['name'], 'signal_lab_apply')
        resulting_left_volume = 40 + 60 * captured['args']['left']['intensity'] / 100
        resulting_right_volume = 40 + 60 * captured['args']['right']['intensity'] / 100
        self.assertAlmostEqual(resulting_left_volume, 65.7)
        self.assertAlmostEqual(resulting_right_volume, 65.7)
        self.assertEqual(captured['args']['transition_seconds'], 5)
        self.assertEqual(captured['args']['left']['texture'], 70)

    def schedule(self, seconds=0):
        from session_conductor import SessionConductor
        self.core.conductor = SessionConductor([
            {'label': 'Texture', 'detail': 'Set texture to Smooth', 'due_seconds': seconds,
             'control': True, 'action': {'name': 'vector_set_texture', 'args': {'texture': 'Smooth'}}}
        ], self.ns['time'].monotonic(), lambda *e: self.events.append(e))

    def test_explicit_left_heartbeat_command_is_deterministic(self):
        action = self.ns['direct_signal_heart_tempo_action']("Please match the left channel to my heartbeat")
        self.assertEqual(action[0], 'signal_lab_heart_tempo')
        self.assertEqual(action[1]['target'], 'left')
        self.assertEqual(action[1]['mode'], 'escalation')

    def test_explicit_double_heartbeat_command_is_deterministic(self):
        action = self.ns['direct_signal_heart_tempo_action']("Please set the left lane to double my heart rate")
        self.assertEqual(action[0], 'signal_lab_heart_tempo')
        self.assertEqual(action[1]['target'], 'left')
        self.assertEqual(action[1]['mode'], 'double')

    def test_conducted_preset_start_uses_declared_frequency_volume_and_minimum_transition(self):
        snapshot = {
            'active_preset': {
                'name': 'Test envelope', 'frequency_min_hz': 700, 'frequency_max_hz': 1500,
                'frequency_start_hz': 900, 'volume_min': 40, 'volume_max': 100,
                'volume_start': 52, 'am_rate_min_hz': .2, 'am_rate_max_hz': 5,
                'am_depth_min_percent': 4, 'am_depth_max_percent': 12,
                'secondary_am_rate_min_hz': .05, 'secondary_am_rate_max_hz': 2,
                'secondary_am_depth_min_percent': 4, 'secondary_am_depth_max_percent': 12,
                'fm_depth_min_percent': 4, 'fm_depth_max_percent': 12,
                'transition_min_seconds': 7,
            },
            'live': {
                'left_signal': {'volume': 90, 'freq': 3000, 'amFreq': 8, 'amDepth': 20},
                'right_signal': {'volume': 20, 'freq': 500, 'amFreq': .1, 'amDepth': 1},
            },
        }
        args = self.ns['signal_lab_preset_start_args'](snapshot)
        self.assertAlmostEqual(args['left']['intensity'], 20.0)
        self.assertAlmostEqual(args['right']['intensity'], 20.0)
        self.assertAlmostEqual(args['left']['texture'], 25.0)
        self.assertAlmostEqual(args['right']['texture'], 25.0)
        self.assertEqual(args['transition_seconds'], 7.0)
        for lane in ('left', 'right'):
            for value in args[lane].values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 100.0)

    def test_relief_heartbeat_command_selects_ninety_percent(self):
        action = self.ns['direct_signal_heart_tempo_action']("Use the right electrode for a slower relief heartbeat pace")
        self.assertEqual(action[1]['target'], 'right')
        self.assertEqual(action[1]['mode'], 'relief')

    def test_short_approval_resolves_pending_heartbeat_proposal(self):
        combined = "I propose matching the left lane to your heartbeat. Shall I do that? Yes, please."
        action = self.ns['direct_signal_heart_tempo_action'](combined)
        self.assertEqual(action[0], 'signal_lab_heart_tempo')
        self.assertEqual((action[1]['target'], action[1]['mode']), ('left', 'escalation'))

    def test_heart_observation_is_not_treated_as_a_command(self):
        self.assertIsNone(self.ns['direct_signal_heart_tempo_action']("My heart rate is 46 BPM"))

    def test_unsupported_heart_rate_causation_is_rejected(self):
        self.assertTrue(self.core._contains_unsupported_biometric_claim(
            "The lane can push your heart rate up a few beats on its own."
        ))
        self.assertFalse(self.core._contains_unsupported_biometric_claim(
            "The left lane is verified at 1.3 hertz, following 78 BPM."
        ))

    def test_unverified_heartbeat_narration_is_detected(self):
        self.assertTrue(self.core._contains_heart_lock_claim("The left lane is deepening its heartbeat lock."))
        self.assertTrue(self.core._contains_heart_lock_claim("The signal is riding your pulse."))
        self.assertFalse(self.core._heart_lock_verified_in_state({
            'heart_tempo': {'fresh': False}, 'heart_tempo_control': {'active': False}}))

    def test_verified_heart_state_requires_effective_rate_match(self):
        state = {'heart_tempo': {'fresh': True, 'bpm': 60},
                 'heart_tempo_control': {'active': True, 'target': 'left', 'ratio': 1.0},
                 'live': {'left_signal': {'amFreq': 1.0}}}
        self.assertTrue(self.core._heart_lock_verified_in_state(state))
        state['live']['left_signal']['amFreq'] = 1.2
        self.assertFalse(self.core._heart_lock_verified_in_state(state))

    def test_signal_step_caps_actual_volume_and_modulation_complexity(self):
        snapshot = {
            'active_preset': {
                'volume_min': 40, 'volume_max': 100, 'frequency_min_hz': 600, 'frequency_max_hz': 1200,
                'am_rate_min_hz': .2, 'am_rate_max_hz': 5, 'am_depth_min_percent': 4,
                'am_depth_max_percent': 12, 'secondary_am_rate_min_hz': .05,
                'secondary_am_rate_max_hz': 2, 'secondary_am_depth_min_percent': 4,
                'secondary_am_depth_max_percent': 12, 'fm_depth_min_percent': 4,
                'fm_depth_max_percent': 12},
            'live': {
                'left_signal': {'volume': 50, 'freq': 700, 'amFreq': .5, 'amDepth': 5,
                                'secondaryAmFreq': .1, 'secondaryAmDepth': 4, 'fmDepth': 0},
                'right_signal': {'volume': 50, 'freq': 700, 'amFreq': .5, 'amDepth': 5,
                                 'secondaryAmFreq': .1, 'secondaryAmDepth': 4, 'fmDepth': 0}},
            'last_command': None,
        }
        lane = {'intensity': 100, 'texture': 100, 'vibration_rate': 100, 'vibration_depth': 100,
                'secondary_rate': 100, 'secondary_depth': 100, 'modulation': 100}
        bounded = self.ns['bound_signal_lab_step'](
            {'left': dict(lane), 'right': dict(lane), 'transition_seconds': 10, 'reason': 'test'}, snapshot)
        for side in ('left', 'right'):
            resulting_volume = 40 + 60 * bounded[side]['intensity'] / 100
            self.assertLessEqual(resulting_volume, 55.0001)
            self.assertLessEqual(bounded[side]['texture'], 26.667 + 0.001)
            increases = [
                bounded[side]['vibration_depth'] > 12.501,
                bounded[side]['secondary_depth'] > 0.001,
                bounded[side]['modulation'] > 0.001,
            ]
            self.assertLessEqual(sum(increases), 1)

    def test_elapsed_time_does_not_confirm(self):
        self.schedule()
        self.core.history = [{'role': 'assistant', 'content': 'I changed the texture to Smooth.'}]
        self.core._session_brief_conductor_context()
        self.assertEqual('pending', self.core.conductor.milestones[0]['state'])

    def test_signal_parameter_dump_is_removed_before_speech(self):
        self.ns['SESSION_MODE'] = 'signal_lab'
        raw = (
            "Done. Left lane: intensity 38 / texture 64 / vibration_rate 32 / vibration_depth 16. "
            "The signal now feels steadier and more deliberate."
        )
        filtered = self.core._filter_spoken_signal_telemetry(raw)
        self.assertNotIn("vibration_rate", filtered)
        self.assertNotIn("intensity 38", filtered)
        self.assertIn("steadier and more deliberate", filtered)

    def test_explicit_telemetry_can_pass_to_speech(self):
        self.ns['SESSION_MODE'] = 'signal_lab'
        self.core.director_thread.allow_raw_signal_telemetry = True
        raw = "Left lane: intensity 38 / texture 64 / vibration_rate 32."
        self.assertEqual(raw, self.core._filter_spoken_signal_telemetry(raw))

    def test_strict_chat_moves_opening_to_context_and_alternates(self):
        prepared = self.core._prepare_openai_messages([
            {'role': 'system', 'content': 'Director rules'},
            {'role': 'assistant', 'content': 'There you are.'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'user', 'content': 'Testing'},
            {'role': 'assistant', 'content': 'Ready.'},
        ], strict_alternation=True)
        self.assertEqual(['system', 'user', 'assistant'], [item['role'] for item in prepared])
        self.assertIn('PRIOR ASSISTANT OPENING', prepared[0]['content'])
        self.assertIn('Hello\n\nTesting', prepared[1]['content'])

    def test_transition_order_and_no_duplicate_attempt(self):
        self.schedule()
        con = self.core.conductor
        self.assertFalse(con.transition(0, 'confirmed'))
        self.assertTrue(con.transition(0, 'attempted'))
        self.assertFalse(con.transition(0, 'attempted'))
        self.assertTrue(con.transition(0, 'failed', {'error': 'offline'}))
        self.assertEqual([], con.due(1e12))
        self.assertFalse(con.transition(0, 'confirmed'))

    def test_not_due_does_not_run(self):
        self.schedule(9999)
        self.core._execute_tool = lambda *a: self.fail('early execution')
        self.assertFalse(self.core._run_due_brief_control())

    def test_due_runs_once_and_confirms_matching_state(self):
        self.schedule()
        calls = []
        self.core._execute_tool = lambda *a: calls.append(a) or {'ok': True, 'state': {'semantic': {'texture': 'Smooth'}}}
        self.assertTrue(self.core._run_due_brief_control())
        self.assertFalse(self.core._run_due_brief_control())
        self.assertEqual(1, len(calls))
        self.assertEqual('confirmed', self.core.conductor.milestones[0]['state'])
        self.assertEqual(['session_brief_milestone_attempted', 'session_brief_milestone_completed'],
                         [e[0] for e in self.events if e[0].startswith('session_brief')])

    def test_unrelated_action_cannot_confirm(self):
        self.schedule()
        self.core._execute_tool = lambda *a: {'ok': True, 'state': {'semantic': {'texture': 'Rough'}}}
        self.core.execute_tool('vector_set_texture', {'texture': 'Rough'})
        self.assertEqual('pending', self.core.conductor.milestones[0]['state'])

    def test_stale_noop_rejected_or_missing_state_fails(self):
        for result in ({'ok': True}, {'ok': False}, {'ok': True, 'noop': True},
                       {'ok': True, 'state': {'semantic': {'texture': 'Rough'}}},
                       {'ok': True, 'accepted': {'ok': False}, 'state': {'semantic': {'texture': 'Smooth'}}}):
            with self.subTest(result=result):
                self.schedule()
                self.core._execute_tool = lambda *a: result
                self.core.execute_tool('vector_set_texture', {'texture': 'Smooth'})
                self.assertEqual('failed', self.core.conductor.milestones[0]['state'])

    def test_exception_records_failed_attempt(self):
        self.schedule()
        def fail(*args):
            raise RuntimeError('offline')
        self.core._execute_tool = fail
        with self.assertRaises(RuntimeError):
            self.core.execute_tool('vector_set_texture', {'texture': 'Smooth'})
        self.assertEqual('failed', self.core.conductor.milestones[0]['state'])

    def test_hold_disarm_and_recording_preserve_pending(self):
        for attr in ('vector_autonomous_held', 'recording'):
            self.schedule()
            setattr(self.core, attr, True)
            self.assertFalse(self.core._run_due_brief_control())
            setattr(self.core, attr, False)
        self.core.vector_autonomous_active = False
        self.assertFalse(self.core._run_due_brief_control())
        self.assertEqual('pending', self.core.conductor.milestones[0]['state'])

    def test_shutdown_and_cross_mode_authority_preserved(self):
        for name in ('vector_stop', 'vector_resume', 'vector_neutral', 'signal_lab_apply'):
            self.assertFalse(self.core._brief_control_authorized(name))
        self.ns['SESSION_MODE'] = 'signal_lab'
        self.assertFalse(self.core._brief_control_authorized('vector_set_texture'))
        self.assertTrue(self.core._brief_control_authorized('signal_lab_apply'))

    def test_signal_confirmation_requires_exact_validated_command(self):
        args = {'left': {'intensity': 30}, 'right': {'intensity': 40}, 'transition_seconds': 10}
        result = {'ok': True, 'status': 'applied_offline', 'command': {
            'request_id': 'abc', 'left_semantic': {'intensity': 30},
            'right_semantic': {'intensity': 40}, 'transition_seconds': 10,
            'left_signal': {'volume': .3}, 'right_signal': {'volume': .4}}}
        self.assertFalse(self.core._verified_control('signal_lab_apply', args, result))
        result['verification_state'] = {'ok': True, 'last_command': {'request_id': 'abc'},
                                        'live': {'left_signal': {'volume': .3}, 'right_signal': {'volume': .4}}}
        self.assertTrue(self.core._verified_control('signal_lab_apply', args, result))
        result['command']['right_semantic']['intensity'] = 31
        self.assertFalse(self.core._verified_control('signal_lab_apply', args, result))
        self.assertFalse(self.core._verified_control('signal_lab_apply', args, {'ok': True}))

    def test_heart_tempo_confirmation_requires_effective_rate_readback(self):
        args = {'target': 'left', 'mode': 'escalation'}
        result = {'ok': True, 'status': 'heart_tempo_control_requested', 'verification_state': {
            'heart_tempo_control': {'active': True, 'target': 'left', 'mode': 'escalation', 'ratio': 1.0},
            'heart_tempo': {'fresh': True, 'bpm': 48.0},
            'live': {'left_signal': {'amFreq': 0.8}, 'right_signal': {'amFreq': 0.37}},
        }}
        self.assertTrue(self.core._verified_control('signal_lab_heart_tempo', args, result))
        result['verification_state']['live']['left_signal']['amFreq'] = 0.37
        self.assertFalse(self.core._verified_control('signal_lab_heart_tempo', args, result))

    def test_heart_lock_language_is_mechanically_grounded(self):
        self.assertTrue(self.core._contains_unexecuted_action_claim(
            'The left lane is locked to your heartbeat.'
        ))

    def test_generated_motion_checks_plan_not_just_ack(self):
        args = {'pattern': 'sine', 'minimum': .2, 'maximum': .8}
        result = {'ok': True, 'accepted': {'ok': True, 'motion_source': {
            'active': True, 'held': False, 'plan': dict(args)}}}
        self.assertTrue(self.core._verified_control('vector_generated_motion_plan', args, result))
        result['accepted']['motion_source']['plan']['maximum'] = .9
        self.assertFalse(self.core._verified_control('vector_generated_motion_plan', args, result))

    def test_signal_live_readback_confirms_and_is_recorded(self):
        from session_conductor import SessionConductor
        c = self.core
        self.ns['SESSION_MODE'] = 'signal_lab'
        args = {'left': {'intensity': 30}, 'right': {'intensity': 40}, 'transition_seconds': 10}
        command = {'request_id': 'one', 'left_semantic': args['left'], 'right_semantic': args['right'],
                   'transition_seconds': 10, 'left_signal': {'volume': .3}, 'right_signal': {'volume': .4}}
        c.conductor = SessionConductor([{'label': 'Change', 'control': True,
                                        'action': {'name': 'signal_lab_apply', 'args': args}}], 0,
                                       lambda *e: self.events.append(e))
        c._execute_tool = lambda *a: {'ok': True, 'status': 'applied_offline', 'command': command}
        c.signal_lab_get = lambda *a, **kw: {'ok': True, 'last_command': command,
                                           'live': {'left_signal': {'volume': .3}, 'right_signal': {'volume': .4}}}
        c.execute_tool('signal_lab_apply', args)
        self.assertEqual('confirmed', c.conductor.snapshot()[0]['state'])
        self.assertIn('verification_state', c.conductor.snapshot()[0]['evidence'])

    def test_signal_readback_failure_does_not_confirm_post_ack(self):
        from session_conductor import SessionConductor
        c = self.core
        args = {'reason': 'Scheduled neutral'}
        c.conductor = SessionConductor([{'label': 'Neutral', 'control': True,
                                        'action': {'name': 'signal_lab_neutral', 'args': args}}], 0,
                                       lambda *e: self.events.append(e))
        c._execute_tool = lambda *a: {'ok': True, 'status': 'neutral_offline'}
        def offline(*a, **kw):
            raise RuntimeError('readback offline')
        c.signal_lab_get = offline
        c.execute_tool('signal_lab_neutral', args)
        self.assertEqual('failed', c.conductor.snapshot()[0]['state'])

    def test_autonomous_receipt_survives_ledger_reset_and_barge_in(self):
        c = self.core
        c.director_thread.generation = 1
        c._execute_tool = lambda *a: {'ok': True, 'state': {'semantic': {'texture': 'Smooth'}}}
        c.execute_tool('vector_set_texture', {'texture': 'Smooth'})
        c.recording = True
        c.history.append({'role': 'assistant', 'content': 'Narration'})
        self.assertEqual('', c._publish_reply('Narration'))
        self.assertEqual([], c.history)
        c.recording = False
        c.director_thread.generation = None
        c.last_action_ledger = {'executed': [], 'failed': []}
        reply = c._publish_reply('Tell me more.')
        self.assertIn('I’ve changed the texture to Smooth.', reply)
        self.assertTrue(reply.endswith('Tell me more.'))
        self.assertEqual(reply, c.history[-1]['content'])
        self.assertTrue(c.conductor.pending_text())
        c.conductor.acknowledge(reply, c.conductor.receipt(reply))
        self.assertEqual('', c.conductor.pending_text())
        self.assertEqual('Next reply.', c._publish_reply('Next reply.'))

    def test_generation_change_defers_after_recording_ends(self):
        self.core.director_thread.generation = 0
        self.core.conductor.announce('I changed the texture.')
        self.assertEqual('', self.core._publish_reply('Late narration.'))
        self.assertTrue(self.core.conductor.pending_text())

    def test_pending_fact_replaces_stock_fallback(self):
        self.core.conductor.announce('I changed the texture.')
        self.assertEqual('I changed the texture.', self.core._publish_reply(self.core._profile_fallback()))

    def test_ack_only_consumes_included_facts(self):
        c = self.core.conductor
        c.announce('First change.')
        spoken = c.pending_text()
        c.announce('Second change.')
        c.acknowledge(spoken, 1)
        self.assertEqual('Second change.', c.pending_text())

    def test_new_identical_receipt_is_not_consumed_by_old_playback(self):
        c = self.core.conductor
        c.announce('Same change.')
        receipt = c.receipt('Same change.')
        c.announce('Same change.')
        c.acknowledge('Same change.', receipt)
        self.assertEqual('Same change.', c.pending_text())

    def test_new_session_restarts_schedule_but_retains_unspoken_change(self):
        self.schedule()
        c = self.core.conductor
        c.transition(0, 'attempted')
        c.transition(0, 'confirmed', {'ok': True})
        c.announce('An earlier change.')
        c.restart(100)
        self.assertEqual([], c.due(99))
        self.assertEqual('pending', c.due(100)[0]['state'])
        self.assertTrue(c.pending_text())

    def test_conditional_or_volume_prose_does_not_become_a_control(self):
        for text in ('Set texture to Smooth if requested.', 'Set volume to 20 percent.',
                     'While talking, change the texture to Smooth.'):
            self.core.profile = {'session_brief': '1. A control beat\n' + text}
            self.assertIsNone(self.core._build_session_brief_milestones()[0]['action'])

    def test_empty_explicit_fallback_is_respected(self):
        self.assertEqual('', self.core._finalize_reply('', fallback=''))
        self.assertEqual(self.core._profile_fallback(), self.core._finalize_reply(''))

    def test_incomplete_long_trailing_clause_is_not_sent_to_tts(self):
        reply = (
            "The anticipation is part of the architecture; it primes the nervous system for the specific pressure "
            "we are about to apply. I am deepening the right lane's rhythmic depth now, honoring that positive "
            "response by letting the texture build rather than forcing a new direction. It's a subtle shift in "
            "momentum, designed to keep the session flowing with the trust we've established over the"
        )
        finalized = self.core._finalize_reply(reply, fallback='fallback')
        self.assertTrue(finalized.endswith('direction.'))
        self.assertNotIn('established over the', finalized)

    def test_complete_long_reply_is_preserved(self):
        reply = (
            "The anticipation is part of the architecture; it primes the nervous system for what comes next. "
            "I am letting the current texture build naturally, and I will make the next change deliberately."
        )
        self.assertEqual(reply, self.core._finalize_reply(reply, fallback='fallback'))

    def test_repetition_similarity_is_shared_and_detects_duplicate(self):
        reply = (
            "I’m maintaining the present state while we give this part of the session enough room to develop "
            "naturally, without introducing another change before the current pattern has had time to settle."
        )
        self.core.recent_spoken_replies = [reply]
        self.assertEqual(1.0, self.core._near_duplicate_similarity(reply))

    def test_repetition_fallback_rotates_in_vector_mode(self):
        first = self.core._next_repetition_fallback()
        second = self.core._next_repetition_fallback()
        self.assertNotEqual(first, second)
        self.assertIn('Vector', first)
        self.assertIn('Vector', second)

    def test_repetition_fallback_rotates_in_signal_lab_mode(self):
        self.ns['SESSION_MODE'] = 'signal_lab'
        try:
            first = self.core._next_repetition_fallback()
            second = self.core._next_repetition_fallback()
        finally:
            self.ns['SESSION_MODE'] = 'vector'
        self.assertNotEqual(first, second)
        self.assertIn('signal', first.lower())
        self.assertIn('signal', second.lower())

    def test_positive_feedback_uses_grounded_repetition_fallback(self):
        self.assertIn('def _positive_feedback_fallback', SOURCE)
        self.assertIn('self._looks_like_positive_feedback(text)', SOURCE)
        self.assertIn('current signal is working well', SOURCE)

    def test_schedule_parses_numbered_and_timed_sections(self):
        self.core.profile = {'session_brief': '1. Opening\nA conversation.\n\n[02:30] Change\nSet texture to Smooth'}
        result = self.core._build_session_brief_milestones()
        self.assertEqual([0, 150], [m['due_seconds'] for m in result])
        self.assertEqual('vector_set_texture', result[1]['action']['name'])

    def test_flexible_six_phase_brief_uses_window_starts_without_false_controls(self):
        self.core.profile = {'session_brief': (
            '1. Introduction\nTarget window: 0–8 minutes\nEstablish the tone.\n\n'
            '2. Personalisation\nTarget window: 8–16 minutes\nRespond to feedback.\n\n'
            '3. Session Body 1\nTarget window: 16–28 minutes\nExplore volume and texture narratively.\n\n'
            '4. Session Body 2\nTarget window: 28–40 minutes\nDevelop the session.\n\n'
            '5. Session Body 3\nTarget window: 40–52 minutes\nCreate contrast.\n\n'
            '6. Session Closure\nTarget window: 52–60 minutes\nWind down and close.'
        )}
        result = self.core._build_session_brief_milestones()
        self.assertEqual(6, len(result))
        self.assertEqual([0, 480, 960, 1680, 2400, 3120], [m['due_seconds'] for m in result])
        self.assertTrue(all(m['flexible_phase'] for m in result))
        self.assertTrue(all(not m['control'] and m['action'] is None for m in result))

    def test_markdown_blank_lines_keep_window_and_detail_with_phase(self):
        self.core.profile = {'session_brief': (
            '## 1. Introduction\n\nTarget window: 0–8 minutes\n\nEstablish the tone.\n\n'
            '## 2. Personalisation\n\nTarget window: 8–16 minutes\n\nRespond to feedback.'
        )}
        result = self.core._build_session_brief_milestones()
        self.assertEqual(2, len(result))
        self.assertEqual([0, 480], [m['due_seconds'] for m in result])
        self.assertIn('Establish the tone.', result[0]['detail'])

    def test_flexible_phase_requires_verbatim_evidence_and_authoritative_time(self):
        self.assertNotIn('successful playback after phase window opened', SOURCE)
        self.assertIn('matched verbatim journal evidence', SOURCE)
        self.assertIn('_latest_journal_evidence("assistant", text)', SOURCE)
        self.assertIn('AUTHORITATIVE SESSION TIME:', SOURCE)
        self.assertIn('control_action_executed', SOURCE)

    def test_control_journal_has_snapshot_and_verification_status(self):
        self.assertIn('"verification_status": verification_status', SOURCE)
        self.assertIn('"before_state": before_state, "after_state": after_state', SOURCE)
        self.assertIn('verification_status = "simulation"', SOURCE)

    def test_conducted_start_and_close_are_hard_gated(self):
        self.assertIn('def is_conducted_session_start_command', SOURCE)
        self.assertIn('start = is_conducted_session_start_command(raw)', SOURCE)
        self.assertIn('conducted_go_ahead_announced', SOURCE)
        self.assertIn('assistant_announcement_then_explicit_user_consent', SOURCE)
        self.assertIn('session_elapsed >= minutes * 60.0', SOURCE)
        self.assertIn('conducted_final_wave_announced', SOURCE)
        self.assertIn('session_finish_stage == "awaiting_reflection"', SOURCE)
        self.assertIn('session_finish_stage == "awaiting_summary_receipt"', SOURCE)

    def test_latest_due_flexible_phase_becomes_current_direction(self):
        from session_conductor import SessionConductor
        self.core.conductor = SessionConductor([
            {'label': '1. Introduction', 'detail': 'Open', 'due_seconds': 0, 'flexible_phase': True},
            {'label': '2. Personalisation', 'detail': 'Adapt', 'due_seconds': 10, 'flexible_phase': True},
        ], self.ns['time'].monotonic() - 20, lambda *e: None)
        context = self.core._session_brief_conductor_context()
        self.assertIn('CURRENT FLEXIBLE PHASE: 2. Personalisation: Adapt', context)

    def test_configured_schedule_is_sorted_and_validated(self):
        self.core.profile = {'session_brief_schedule': [
            {'label': 'Later', 'due_seconds': 90}, {'label': 'First', 'due_seconds': 10},
            {'due_seconds': -1}, {'due_seconds': float('nan')}, {'due_seconds': True},
            {'due_seconds': 20, 'action': 'invalid'}]}
        self.assertEqual(['First', 'Later'], [m['label'] for m in self.core._build_session_brief_milestones()])


if __name__ == "__main__":
    unittest.main()
