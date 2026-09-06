import json
from pathlib import Path


SOURCE = Path(__file__).with_name("gwendolyn_direct.py").read_text(encoding="utf-8")
CONFIG = json.loads(Path(__file__).with_name("config.example.json").read_text(encoding="utf-8"))


assert CONFIG["llm_backend"] == "ollama"
assert CONFIG["openai_base"] == "http://127.0.0.1:8091/v1"
assert CONFIG["openai_model"] == "gwendolyn-model"
assert CONFIG["ollama_base"] == "http://127.0.0.1:11434"
assert CONFIG["ollama_model"] == "qwen3:14b"

assert 'backend not in {"ollama", "openai"}' in SOURCE
assert '"/chat/completions"' in SOURCE
assert '"/api/chat"' in SOURCE
assert '"max_tokens": max_predict' in SOURCE
assert '"options": {"temperature": temperature, "num_predict": max_predict}' in SOURCE

assert 'choices = data.get("choices") or []' in SOURCE
assert 'return choices[0].get("message") or {}' in SOURCE
assert 'return self._llm_message(data)' in SOURCE
assert '"reasoning_effort": "none"' in SOURCE
assert '"chat_template_kwargs": {"enable_thinking": False}' in SOURCE
assert '"cache_prompt": True' in SOURCE
assert 'os.environ["PATH"] = _torch_dll_text' in SOURCE
assert 'llm_max_reply_tokens' in SOURCE
assert 'PROPOSAL cleared because action is absent from final visible reply' in SOURCE
assert 'VOICE-FIRST SHAPE' in SOURCE
assert 'spoken = re.sub(r"\\*{1,3}", "", spoken)' in SOURCE
assert 'GROUNDING preserved non-mechanical draft sentences after rewrite failure' in SOURCE
assert 'Qwen3.8\'s native Jinja template' in SOURCE
assert 'named_top_focus = semantic_top_focus(text)' in SOURCE
assert 'adaptive_vad_reset_seconds' in SOURCE
assert 'TIMELINE GROUNDING replaced reply contradicting authoritative relaxing band' in SOURCE
assert 'microphone stream reset after repeated empty captures' in SOURCE
assert 'LLM returned no user-visible content' in SOURCE
assert 'cublas64_12.dll' in SOURCE

print("LLM backend adapter assertions: PASS")
