# Model server setup

Gwendolyn supports Ollama's native chat API and an OpenAI-compatible local server. Keep the server on `127.0.0.1` unless you have intentionally secured remote access. A model with reliable tool/function calling is essential because Gwendolyn uses structured Vector actions.

## Option A: Ollama on Windows

1. Install Ollama from the [official Windows instructions](https://docs.ollama.com/windows). Its local API normally listens on `http://localhost:11434`.
2. Open PowerShell and download the default model:

   ```powershell
   ollama pull qwen3:14b
   ```

3. In `config.json`, use:

   ```json
   {
     "llm_backend": "ollama",
     "ollama_base": "http://127.0.0.1:11434",
     "ollama_model": "qwen3:14b"
   }
   ```

4. Run `CHECK_SETUP.bat` and confirm that the configured model is listed.

The 14B model is a practical default, but model speed depends heavily on RAM, VRAM and context size. The [official Ollama Qwen3 library page](https://ollama.com/library/qwen3) lists smaller and larger variants. If you pull a different tag, put that exact tag in `ollama_model`.

## Option B: llama.cpp on Windows

1. Download a current Windows build from the [official llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases), choosing the package for your hardware.
2. Obtain an instruction-tuned GGUF model with tool-calling support. Review and accept that model's license yourself; no model is bundled here.
3. Start the server. With the traditional executable name, a typical NVIDIA example is:

   ```powershell
   .\llama-server.exe -m "C:\Models\your-model.gguf" --host 127.0.0.1 --port 8091 -c 32768 -ngl 99 --jinja --alias gwendolyn-model
   ```

   Current builds may expose the same server as `llama serve`. Run the executable with `--help` if your downloaded build uses the newer command layout. Reduce `-ngl` or the context size when the model does not fit your hardware.

4. In `config.json`, use:

   ```json
   {
     "llm_backend": "openai",
     "openai_base": "http://127.0.0.1:8091/v1",
     "openai_model": "gwendolyn-model",
     "openai_api_key": ""
   }
   ```

5. Run `CHECK_SETUP.bat`. It checks `/v1/models` and does not send a prompt.

llama.cpp documents its [OpenAI-compatible chat endpoint and function calling](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). `--jinja` enables its tool-call parsing path; some models need a matching chat template, so use the model publisher's instructions when tool calls appear as plain text.

## Choosing a model

Start with an instruction-tuned model that explicitly supports tool calling. Smaller models answer faster but are more likely to miss tool schemas or confuse a proposal with an executed action. A 14B-class model is a reasonable baseline; a well-quantized 27B–32B model is preferable when your hardware can run it responsively.

Gwendolyn sends a substantial system prompt and live Vector state. Allocate enough context for the model and watch the server console for context truncation. Do not expose an unprotected model server to the LAN or internet.
