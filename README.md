# Frontier Model API Client

A lightweight Python client and CLI to **call text and vision models through Kaggle Model Proxy**, keep separate conversation sessions, and record each request as JSONL. This is **not a benchmark framework**: no datasets, tasks, scoring, leaderboards, hosted executions, code execution, or direct OpenAI/Gemini/Anthropic API backends.

Original repository: https://github.com/HoangVo-Prog/frontier-model-benchmarking

## Applying this refactor to the existing Git repository

The downloaded bundle contains `apply_to_repo.py` because repository writes may not be available from every ChatGPT integration. Extract the ZIP **outside** your existing repository, then run:

```bash
# Run from anywhere; use paths matching your own machine.
python /path/to/frontier-model-api-client/apply_to_repo.py /path/to/frontier-model-benchmarking
cd /path/to/frontier-model-benchmarking
git diff --cached --stat
git diff --cached
python -m pip install -e ".[dev]"
python -m pytest -q
git commit -m "Refactor to Kaggle Model Proxy API client"
```

The migration **refuses a dirty working tree**, creates `refactor/frontier-model-api-client` and an `archive/pre-frontier-api-refactor-...` backup branch, and stages deletions of tracked legacy benchmark modules and research artifacts. The original files remain recoverable from the backup branch and Git history. It does **not** push, merge, commit, delete untracked work, or replace your current Conda environment. Review the staged diff before committing. If you need old experiment reports in the new branch, restore them selectively from the backup branch.

## Install in an existing environment

Use Python **3.11+**. Do not recreate an existing Conda environment:

```bash
conda activate linux-python-3-11
python -m pip install -e ".[dev]"
frontier doctor
```

Or from a new/existing Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate           # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
frontier doctor
```

The package includes Kaggle's official `kaggle` CLI and the `kaggle-benchmarks` inference SDK (required for Model Proxy). It does not install KaggleHub, dataset loaders, or evaluation packages. `frontier doctor` displays interpreter and dependency status without printing credentials.

## Authentication (two separate credentials)

1. Generate a Kaggle API token in [Kaggle Settings](https://www.kaggle.com/settings/api) and export it to your shell, or place it in a private `.env` file:

   ```bash
   export KAGGLE_API_TOKEN="YOUR_KAGGLE_API_TOKEN"
   ```

2. Obtain the **temporary Model Proxy credential**, separate from the account token:

   ```bash
   kaggle benchmarks auth -y --env-file .env
   frontier auth check
   ```

The Kaggle CLI writes `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`, and `MODEL_PROXY_EXPIRY_TIME` into `.env`. `frontier` loads an optional `.env` from the working directory or its parents without overriding existing shell environment variables. If you refresh the credential, restart the Python process if it has already loaded the old values (or remove stale shell overrides). `KAGGLE_API_TOKEN` by itself **cannot call Model Proxy**.

`.env`, token files and `logs/` are gitignored. Never commit credentials. Login via browser is **not** initiated by this client. `frontier auth check` reports presence and an expiry check when the CLI supplies a parseable timestamp; it does not guarantee that the server will accept a token.

## Model discovery and quota

```bash
frontier models list
frontier quota
```

These commands delegate to Kaggle's official `kaggle benchmarks tasks models` and `kaggle benchmarks quota`. The catalog can include models you **cannot** call from a local Model Proxy token; the local `LLMS_AVAILABLE` set and Kaggle account policy may restrict access. Kaggle's hosted-task model catalog does not imply unrestricted local inference. `quota` reports **Model Proxy inference spending**, not Kaggle GPU compute hours. Review Kaggle's current terms and restrictions before use.

## Python API

```python
from frontier_client import FrontierClient

client = FrontierClient(model="MODEL_ID")
response = client.prompt("Explain high-resolution visual search.")
print(response.text)
print(response.usage)  # fields are None when the SDK supplies no usage
print(response.request_id, response.latency_seconds)
```

For images, select a **vision-capable** model:

```python
response = client.prompt("Describe the objects.", image="images/example.png")
print(response.text)
```

Local `.png`, `.jpg` and `.jpeg` files (maximum **20 MiB**) and public **HTTPS** URLs ending with a supported image extension are accepted. URL contents are fetched by the Kaggle SDK during inference; remote URLs cannot be prevalidated for size, so only use trusted public endpoints. The SDK handles image payload creation. This package never logs raw image bytes or base64 payloads. No PDF, audio, video, ZIP, or dataset inputs.

For multi-turn chat, each session keeps the Kaggle SDK conversation history and can be reset/exported:

```python
with client.chat(system_prompt="You are a helpful assistant.") as chat:
    chat.send("Explain transformers.")
    response = chat.send("Explain that more simply.")
    print(response.text)
    chat.reset()                         # starts a fresh conversation
    chat.send("New topic")
    chat.export("conversation.json")   # explicit export of prompts and answers
```

Sessions are isolated, including when calls are interleaved; using the session as a context manager ensures cleanup. A failed request is rolled back from the SDK chat history. `client.prompt` is stateless regardless of any existing session.

Optional generation parameters:

```python
client = FrontierClient(model="MODEL_ID", reasoning="low")
# temperature requires an SDK model/backend that supports it.
client = FrontierClient(model="MODEL_ID", api="genai", temperature=0.2)
```

`api="openai"` (default) and `api="genai"` select **two Kaggle Model Proxy SDK adapters**, not provider-direct APIs. Depending on the installed SDK, OpenAI-style proxy adapter models may not support temperature. Unsupported options fail with explicit errors rather than being silently ignored. Reasoning support depends on the selected model. Optional `schema=SomeDataclass` on `client.prompt` or `chat.send` passes a structured-output schema to the SDK; `response.structured` preserves the result, and `response.text` is a JSON serialization of the structure.

## CLI

```bash
frontier ask --model MODEL_ID --prompt "Explain transformers."
frontier ask --model MODEL_ID --prompt "Describe the image." --image images/example.png
frontier ask --model MODEL_ID --prompt "Describe the image." --image-url https://example.org/photo.jpg
frontier chat --model MODEL_ID
```

Interactive chat commands: `/reset`, `/quit`, `/model MODEL_ID`, `/export FILE.json`, and `/image path/to/pic.png Describe this`. Quote image paths with spaces. `--log-dir DIR`, `--no-log`, `--log-prompts`, `--reasoning`, `--temperature` and `--api` can be passed to `ask` and `chat`.

## Response structure and logging

Each `FrontierResponse` exposes `.text`, `.model`, `.usage`, `.latency_seconds`, `.request_id`, `.timestamp`, `.structured`, and `.metadata`. Token/cost metadata comes from the **individual assistant message** (not accumulated conversation usage). `total_cost` is derived from SDK nanodollar reporting only when provided. Missing values remain `null`.

Unless `logging=False` / `--no-log` is set, every successful or failed request appends a single record to `logs/requests.jsonl` (configurable using `log_dir` or `--log-dir`). A cross-process advisory lock and fsync protect against simultaneous writes and truncation. By default, prompt text is omitted for privacy; set `log_prompts=True` / `--log-prompts` when you explicitly need full prompt archives. **Full model responses are always logged when logging is enabled**; disable persistent logging for sensitive outputs. Image references may reveal paths or URLs. Never store credentials or private query tokens in image URLs.

Example schema (illustrative, *not* an actual result):

```json
{
  "request_id": "unique-id",
  "timestamp": "2026-11-19T00:00:00+00:00",
  "session_id": null,
  "model": "MODEL_ID",
  "input": {"type": "text", "prompt": null, "images": []},
  "output": {"text": "Example response", "structured": null},
  "usage": {
    "input_tokens": null,
    "output_tokens": null,
    "input_tokens_cost_nanodollars": null,
    "output_tokens_cost_nanodollars": null,
    "total_cost_nanodollars": null,
    "total_backend_latency_ms": null,
    "total_cost": null
  },
  "latency_seconds": 0.42,
  "status": "success",
  "error": null
}
```

When inference fails, `status` is `failed`, `error` contains a **sanitized error category and message**, and `output.text` is `null`. Logging does not serialize SDK objects with unfiltered `repr()` or arbitrary `str()` calls. Each `.jsonl` line is independently parseable JSON.

## Troubleshooting and tests

| Problem | Action |
| --- | --- |
| Missing/expired Proxy token | `kaggle benchmarks auth -y --env-file .env`; rerun `frontier auth check` |
| Kaggle CLI command not found | Install `kaggle>=2.2.0`; check `which kaggle` / `where kaggle` |
| 403 / unknown model | Check local token restrictions, `LLMS_AVAILABLE`, and model support |
| Unsupported image | Use a vision-capable model and PNG/JPEG input |
| Temperature unsupported | Omit `temperature` or select a supported SDK backend |
| Quota exhausted | Run `frontier quota` |
| Wrong environment | Run `frontier doctor` to inspect the interpreter |

Run the isolated mocked test suite without Kaggle credentials or API usage:

```bash
python -m pytest -q
```

If you have valid credentials and quota, manually try one tiny text request and one image request with your permitted model; these live calls are intentionally **not** automatic tests.

For the official API and model restrictions, see [Kaggle Benchmarks Python SDK](https://github.com/Kaggle/kaggle-benchmarks), [Kaggle CLI Benchmarks commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md), and [Kaggle documentation](https://www.kaggle.com/docs/benchmarks).
