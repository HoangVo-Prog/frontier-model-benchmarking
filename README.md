# Frontier Chat — Kaggle Model Proxy

A minimal **local** Python API and `frontier` CLI for text prompts, images, and isolated multi-turn conversations through the official **Kaggle Model Proxy SDK** (`kaggle-benchmarks`). No benchmark evaluation, hosted runs, leaderboards, or direct provider APIs.

```bash
frontier models list
frontier ask --model MODEL_ID --prompt "Hello!"
frontier ask --model MODEL_ID --prompt "Describe this" --image photo.png
frontier chat --model MODEL_ID
```

## Install

Python 3.11+; existing environment is fine:

```bash
conda activate linux-python-3-11
python -m pip install -e ".[dev]"
frontier doctor
```

## Authentication and **local** model access

1. Obtain a Kaggle account token from [Kaggle settings](https://www.kaggle.com/settings/api). Set `KAGGLE_API_TOKEN` in the shell or private `.env`.
2. In a **temporary directory** (so SDK examples and documentation don't clutter this repo), initialize Kaggle's *local* Model Proxy environment:

   ```bash
   mkdir -p /tmp/frontier-kaggle-init
   cd /tmp/frontier-kaggle-init
   kaggle benchmarks init -y
   # Copy the generated .env to this project's root, or set FRONTIER_ENV_FILE.
   ```

   The generated `.env` includes temporary `MODEL_PROXY_*` credentials and the **comma-separated `LLMS_AVAILABLE` local allowlist**. Make sure `LLM_DEFAULT` is one of those models; the client also selects an allowed fallback if the default is missing. To refresh the credentials from your project directory: `frontier auth refresh` (equivalent to `kaggle benchmarks auth -y --env-file .env`). Then run `frontier auth check`.

   If you already used `kaggle benchmarks auth` without `init`, you still need `LLMS_AVAILABLE` from `init`. You can copy the list into `.env` yourself. `frontier models list` reports **only this local configuration**, not the full Kaggle hosted model catalog. Local credentials cannot call arbitrary hosted models. Access can also be denied by server-side policy even for listed IDs. To update the model allowlist, rerun `init` in a temporary directory and copy the refreshed configuration.

`frontier quota` checks **Model Proxy inference spend**, not GPU hours. `.env` and `logs/` are ignored by Git. Never commit credentials.

## Python API

```python
from frontier_client import FrontierClient

client = FrontierClient(model="MODEL_ID")  # Choose from frontier models list
print(client.prompt("Explain transformers.").text)
print(client.prompt("Describe this.", image="photo.png").text)

with client.chat(system_prompt="Keep answers brief.") as conversation:
    conversation.send("Explain transformers.")
    print(conversation.send("Explain more simply.").text)
    conversation.reset()
    conversation.export("chat.json")
```

Images: PNG/JPEG local files (up to 20 MiB) or public HTTPS URLs ending in `.png`, `.jpg`, or `.jpeg`. Choose a vision-capable model. Structured output is available with `schema=SomeDataclass` where supported by the SDK. Responses expose `.text`, `.structured`, `.usage`, `.latency_seconds`, `.request_id`, and `.timestamp`. Unreported usage remains `None`.

## Interactive chat

`frontier chat --model MODEL_ID` accepts `/reset`, `/model MODEL_ID` (starts a fresh session), `/image PATH PROMPT` (quote paths with spaces), `/export FILE.json`, and `/quit`. Exports explicitly contain prompts and responses, unlike default request logs.

## Logging

Successful and failed requests append to `logs/requests.jsonl` by default. Each entry includes model, request ID, timestamp, prompt type, sanitized image reference, response, available usage/cost, latency, error category and chat session ID. Prompt text is omitted by default; full **responses are stored**, so use `--no-log` or `FrontierClient(..., logging=False)` for sensitive outputs. Choose another directory via `--log-dir DIR` or `log_dir=...`. The Python API also accepts `log_prompts=True`. Neither image bytes nor credentials are logged.

## Troubleshooting and tests

- Missing or expired credentials: `frontier auth refresh`; inspect with `frontier auth check`. Restart already-running processes after refresh.
- No local models: populate `LLMS_AVAILABLE` using the Kaggle CLI `init` command in a temporary directory.
- Model rejected: use an ID from `frontier models list` and verify account/model capabilities. The local allowlist does not guarantee server authorization.
- CLI or dependency issue: run `frontier doctor` within the active Python environment.
- Inference quota: `frontier quota`. Failed images: use a vision-capable model and valid PNG/JPEG files.

Credential-free unit tests: `python -m pytest -q`. A real inference test is optional and uses your personal quota.

Official docs: [Kaggle Benchmarks SDK](https://github.com/Kaggle/kaggle-benchmarks) · [Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md)
