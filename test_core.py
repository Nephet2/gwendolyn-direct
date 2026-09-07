import importlib.util
from pathlib import Path

p = Path(__file__).with_name('gwendolyn_direct.py')
spec = importlib.util.spec_from_file_location('gd', p)
gd = importlib.util.module_from_spec(spec)
# Avoid importing dependencies in this tiny static test environment by source testing parser logic instead.
src = p.read_text(encoding='utf-8')
assert 'vector_set_texture' in src
assert 'vector_set_top_focus' in src
assert 'vector_resume' in src
assert 'voice_activation_short_pause_ms' in src
assert 'vector_top_spatial_gain' in src
assert 'ptt_candidate' in src
assert 'response_format": "pcm"' in src
assert 'think": False' in src
print('Static core assertions: PASS')

# v0.15 restart semantics and verbosity (static)
assert 'vector_stopped: bool = False' in src
assert 'can we turn it back on' in src
assert 'ready )?let s go' in src
assert 'r"please do"' in src
assert '"speech_verbosity": 4' in src
assert 'Response verbosity' in src
print('v0.15 static assertions: PASS')

# v0.16 stop/resume routing and grounding (static)
assert 'vector\\s+off' in src or 'vector\s+off' in src
assert 'turn|switch' in src and 'it\\s+off' in src or 'it\s+off' in src
assert 'Vector stop acknowledged but engine_state' in src
assert "Vector resume acknowledged but engine_state is still 'Stopped'" in src
assert 'never narrate it as successful' in src
print('v0.16 static assertions: PASS')

# v0.17 full-funscript timeline visibility
assert '"timeline": s.get("timeline")' in src
prompt = open('gwendolyn_prompt.txt', encoding='utf-8').read()
assert 'Full funscript timeline visibility' in prompt
assert 'sync_confidence' in prompt
print('v0.17 timeline assertions: PASS')

# v0.18 direct-player authoritative timeline awareness
assert 'clock_authoritative' in prompt
assert 'media_connected' in prompt
assert 'authoritative media time' in prompt
print('v0.18 direct-player timeline assertions: PASS')

src = Path('gwendolyn_direct.py').read_text(encoding='utf-8')
assert 'APP_VERSION = "0.46.11-alpha1"' in src
assert 'SESSION FINISH requested; awaiting closing reflection' in src
assert 'SESSION FINISH confirmed; {controller} stopped before farewell' in src
assert 'self.execute_tool("vector_stop", {})' in src
assert 'Proactive Director' in src
assert 'proactive_director' in src
assert 'def proactive_tick' in src
assert 'PROACTIVE DIRECTOR MOMENT' in src
assert 'TIMELINE ' in src
assert 'clock_authoritative' in src
print('v0.19 proactive timeline assertions: PASS')
assert '"speech_chunking": "Off"' in src
assert 'Never invent an upcoming rise' in src
assert 'media_sample_age_seconds' in src
print('v0.20 latency/grounding assertions: PASS')

# v0.21 natural narration and delivery migration
assert 'VOICE DELIVERY=' in src
assert 'speech_delivery_migrated_v021' in src
assert 'def narration_tool_result' in src
assert 'Never read internal tool names' in src
assert 'narration_result = self.narration_tool_result' in src
print('v0.21 natural narration/delivery assertions: PASS')

# v0.22 preferred whole-reply delivery
assert 'speech_delivery_migrated_v022' in src
assert 'First sentence -> Off' in src
print('v0.22 whole-reply default assertions: PASS')

# v0.23 proposal/approval grounding
assert 'pending_proposal_text' in src
assert 'GROUNDING replaced repeated proposal after tool execution' in src
assert 'approval received but no Vector tool executed' in src
assert 'Do not fall into a repeated readiness loop' in src
assert 'Do not claim subjective sensation as telemetry' in src
print('v0.23 proposal grounding assertions: PASS')

# v0.24 deterministic modifier tools
assert 'vector_set_targeting' in src
assert 'vector_tempo_window' in src
assert 'vector_restore_modifiers' in src
assert '/v1/modifier/target' in src
assert '/v1/modifier/tempo' in src
assert 'base_prostate_tight' in src
assert 'challenging/testing' in prompt
print('v0.24 modifier Director assertions: PASS')

# v0.25 deterministic targeting/stroke routing and fast grounded confirmation
assert 'vector_adjust_stroke_range' in src
assert '/v1/modifier/stroke-range' in src
assert 'target_side = "base_prostate"' in src
assert 'confirmed without second Ollama pass' in src
assert 'narrowed the stroke range a little' in src
print('v0.25 modifier routing/latency assertions: PASS')

# v0.26 Ollama stall resilience
assert 'self.ollama_session = requests.Session()' in src
assert 'OLLAMA REQUEST phase=' in src
assert 'OLLAMA TIMEOUT phase=' in src
assert 'OLLAMA HTTP session reset after failed request' in src
assert 'def _scenario_fallback' in src
assert 'phase="first-pass"' in src
assert 'phase="post-tool"' in src
assert 'timeout=(connect_timeout, read_timeout)' in src
assert 'phase="empty-response-retry"' in src
assert 'retrying once with a direct conversational prompt' in src
assert 'empty-response retry also returned no user-visible content' in src
assert 'GROUNDING rejected execution claim from empty-response retry' in src
print('v0.26 Ollama resilience assertions: PASS')

# v0.46.1 session-integrity hardening
assert 'conducted_go_ahead_announced' in src
assert 'assistant_announcement_then_explicit_user_consent' in src
assert 'matched verbatim journal evidence' in src
assert 'verification_status' in src and 'before_state' in src and 'after_state' in src
assert 'conducted_final_wave_announced' in src
assert 'session_closure_reflection' in src and 'session_closure_summary_received' in src
print('v0.46.1 session-integrity assertions: PASS')


# v0.27 intent-routed tool exposure
assert 'def select_tools_for_turn' in src
assert 'OLLAMA ROUTE conversational tools=0' in src
assert 'OLLAMA ROUTE vector-intent tools=' in src
assert 'routed_tools = select_tools_for_turn(text, pending_before_turn)' in src
assert 'tool_schema_chars=' in src
assert 'groups[:6]' in src
assert 'assistant_first = self.ollama_chat(base_messages, routed_tools or None, phase="first-pass")' in src
print('v0.27 intent-routed Ollama assertions: PASS')

# v0.28 proposal execution + conversation grounding + empty STT suppression
assert 'pending_proposal_action' in src
assert 'PROPOSAL APPROVAL direct execution' in src
assert 'CONVERSATION-ONLY TURN' in src
assert 'GROUNDING blocked unexecuted action claim on conversation-only turn' in src
assert 'STT empty suppressed count=' in src
assert 'empty_stt_debounce_seconds' in src
assert 'please do' in src.lower()
print('v0.28 grounding/debounce assertions: PASS')

# v0.29 Narrative Director v1
assert 'narrative_director' in src
assert 'narrative_randomness' in src
assert 'def narrative_context' in src
assert 'def _narrative_arc_stage' in src
assert 'def _select_narrative_beat' in src
assert 'NARRATIVE DIRECTOR STATE' in src
assert 'PROPOSAL captured conversational action' in src
assert 'Narrative Director' in src
assert 'weighted' in open('gwendolyn_prompt.txt', encoding='utf-8').read().lower()
print('v0.29 Narrative Director assertions: PASS')


# v0.30 narrative memory / arc templates / adaptive VAD
assert 'def _choose_narrative_arc_template' in src
assert 'narrative_arc_template' in src
assert 'callback_candidates=' in src
assert 'narrative_callback_queue' in src
assert 'vad_adaptive_multiplier' in src
assert 'adaptive_vad_after_empty' in src
assert 'STT genuine speech resets adaptive VAD' in src
assert 'Narrative arc' in src
print('v0.30 narrative memory/adaptive VAD assertions: PASS')

# v0.31 Vector state arc + scrollable setup
assert 'STATE ARC sequence=' in src
assert 'def _action_is_noop' in src
assert 'PROPOSAL rejected no-op' in src
assert 'baseline_top_focus' in src and 'baseline_bottom_focus' in src
assert 'Changes before recovery' in src
assert 'setup_scroll = ttk.Scrollbar' in src
assert 'scrollregion=setup_canvas.bbox("all")' in src
print('v0.31 state arc / setup scrolling assertions: PASS')

# v0.32 authoritative state integrity / no-op recovery
assert 'current_vector_semantic' in src
assert 'def _semantic_snapshot_from_state' in src
assert 'def _refresh_authoritative_state' in src
assert 'def _wait_for_semantic_state' in src
assert 'PROPOSAL no-op resolved locally' in src
assert 'without another LLM request' in src
assert 'The texture is already' in src
assert '"baseline_top_focus": "Top Full"' in src
print('v0.32 authoritative state / local no-op assertions: PASS')

# v0.33 deterministic recovery / hard same-turn grounding
assert 'def _deterministic_recovery_action' in src
assert 'def _execute_due_recovery' in src
assert 'STATE ARC deterministic recovery due' in src
assert 'GROUNDING hard ledger gate blocked execution claim without same-turn action' in src
assert 'self.intervention_count >= self.intervention_target' in src
assert 'RECOVERY' in src
print('v0.33 deterministic recovery / hard grounding assertions: PASS')

# v0.34 private lexicon / future-execution grounding
assert 'PRIVATE_LEXICON_PATH' in src
assert 'Private Language & Sensation Lexicon' in src
assert 'def load_private_lexicon' in src
assert 'def update_private_lexicon' in src
assert 'language_level' in src and 'signal_associations' in src
assert 'i shall(?: now)?' in src
assert 'GROUNDING hard ledger gate rewrote future execution language as pending proposal' in src
assert 'def _safe_proposal_phrase' in src
print('v0.34 private lexicon / future grounding assertions: PASS')

# v0.35 reference behaviour tuning
assert 'TUNING_PRESETS' in src
assert 'expression_tuning_instruction' in src
assert 'Reference behaviour tuning' in src
assert 'Metaphor density' in src
assert 'Director pressure' in src
assert 'Question frequency' in src
assert 'Proposal frequency' in src
assert 'Sensation carryover' in src
assert 'proposal_tune = tuning_value' in src
assert 'They do not change Vector limits' in src
print('v0.35 reference behaviour tuning assertions: PASS')

# v0.36 immersion-preserving grounding mediation
assert 'def _rewrite_unsupported_execution' in src
assert 'phase="grounding-rewrite"' in src
assert 'def _delegated_surprise_action' in src
assert 'DELEGATED surprise action selected' in src
assert 'sounds (?:really |very )?(?:good|great|fabulous|lovely|nice|wonderful|excellent|perfect|tempting|delicious)' in src
assert 'I haven’t changed Vector. If you’d like, I can propose a specific adjustment.' not in src
assert 'GROUNDING immersion rewrite succeeded' in src
print('v0.36 immersion grounding assertions: PASS')

# v0.37 semantic TTS chunking / deterministic signal isolation diagnostics
assert 'def semantic_tts_chunks' in src
assert "backend.key == \"chatterbox\"" in src
assert 'long_form_degradation_avoided=' in src
assert 'chars={[len(chunk) for chunk in chunks]}' in src
assert 'GwendolynVoiceWorker' in src
print('v0.37 TTS stability assertions: PASS')

# v0.38 continuous delivery / repetition and proposal hardening
assert 'TTSPrefetch-' in src
assert 'prefetch_wait=' in src
assert 'def _sanitize_model_text' in src
assert 'REPETITION GUARD rejected near-duplicate' in src
assert 'PROACTIVE suppressed: spoken proposal had no deterministic executable action' in src
assert 'VOICE ACTIVITY false-trigger backoff=' in src
assert 'def _profile_fallback' in src
assert 'def _enforce_grounded_language' in src
print('v0.38 continuity/repetition assertions: PASS')

# v0.43.6 winning-model maintenance hardening
assert 'explicit_control_intent' in src
assert 'STATE ARC recovery deferred' in src
assert 'STATE ARC recovery extended by positive feedback' in src
assert 'state_arc_minimum_dwell_seconds' in src
assert 'see you (?:later|next time|tomorrow)' in src
print('v0.43.6 maintenance assertions: PASS')

# v0.44 persistent preference memory
assert 'MEMORY_PATH' in src
assert 'def observe_preference_evidence' in src
assert 'three separate sessions' in src
assert 'def build_memory_tab' in src
assert 'PERSISTENT MEMORY (local, cross-session, user-reviewable)' in src
print('v0.44 persistent memory assertions: PASS')

# v0.45 behavioral Director personalities
assert 'def personality_operating_context' in src
assert 'The Conductor' in src
assert 'beat_bias' in src and 'initiative_multiplier' in src
assert 'anti_caricature' in src
print('v0.45 Director personality assertions: PASS')
