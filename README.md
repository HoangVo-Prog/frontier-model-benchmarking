# Frontier local chat client

A small Python package and `frontier` CLI for text, images and multi-turn chat through the **official Kaggle Model Proxy SDK**. No hosted benchmark jobs, direct provider APIs, or dataset pipelines.

## Install

```bash
conda activate linux-python-3-11
python -m pip install -e ".[dev]"
frontier doctor
```

## Configure Kaggle accounts in `.env` (no CLI login required)

Copy `.env.example` to `.env` **once** and edit it yourself. The client only reads it. It never modifies `.env`, writes Proxy keys, or saves an expiry timestamp.

One account:

```dotenv
KAGGLE_API_TOKEN=KGAT_your_real_account_token
LLMS_AVAILABLE=MODEL_ID_1,MODEL_ID_2
```

Multiple accounts in priority order (named):

```dotenv
KAGGLE_ACCOUNTS=primary,backup
KAGGLE_API_TOKEN_PRIMARY=KGAT_your_first_account_token
KAGGLE_API_TOKEN_BACKUP=KGAT_your_second_account_token
LLMS_AVAILABLE=MODEL_ID_1,MODEL_ID_2
FRONTIER_MIN_REMAINING_USD=0.05
FRONTIER_QUOTA_CACHE_SECONDS=30
```

Or paste multiple tokens on one line and receive automatic aliases `account_1`, `account_2`, etc.:

```dotenv
KAGGLE_API_TOKENS=KGAT_first_account_token,KGAT_second_account_token
LLMS_AVAILABLE=MODEL_ID_1,MODEL_ID_2
```

Precedence is `KAGGLE_ACCOUNTS`, then `KAGGLE_API_TOKENS`, then `KAGGLE_API_TOKEN`. `KAGGLE_API_TOKEN_*` are **Kaggle account API tokens**, not `MODEL_PROXY_API_KEY` values. Generate them in [Kaggle account settings](https://www.kaggle.com/settings/api). Keep `.env` private; it is ignored by Git. API keys from **different permitted Kaggle accounts** are supported. Multiple tokens from the same account may share the same account quota and do not provide independent limits.

The client uses the same **first-party Kaggle API endpoints** as `kaggle benchmarks auth` and `kaggle benchmarks quota`. For each account it requests a short-lived Model Proxy token **in memory** and refreshes it automatically before expiry (or once on a Proxy authentication error). Expiry exists at Kaggle even though you do not configure or persist it. `frontier auth refresh` explicitly refreshes tokens in its **current process only**; subsequent CLI runs mint fresh tokens automatically. Browser login/`kaggle auth login` is optional and not used by this workflow.

## Commands

```bash
frontier accounts list
frontier auth check              # inspect .env only
frontier auth check --verify     # test Kaggle API/quota
frontier auth refresh            # re-mint Proxy credentials in memory
frontier quota                   # daily and monthly quota per account
frontier models list             # only LLMS_AVAILABLE, not hosted catalog
frontier ask --model MODEL_ID --prompt "Hello!"
frontier ask --model MODEL_ID --prompt "Describe this." --image photo.png
frontier chat --model MODEL_ID
```

Interactive chat: `/reset`, `/model MODEL_ID`, `/image PATH PROMPT`, `/export FILE.json`, `/quit`. Quote file paths containing spaces.

### Automatic quota rotation

Before an inference request, `frontier` checks quota via Kaggle's API (cached for 30 seconds by default). An account is usable only if **all** daily/monthly periods have more than `FRONTIER_MIN_REMAINING_USD` left. Accounts are selected in the `.env` order; the client sticks to one until its available quota drops below the threshold or a **quota-specific** inference error occurs. It then switches to the next usable account. Ordinary HTTP 429 rate limiting, invalid model requests, image failures and unrelated errors **do not** trigger account rotation. An authentication failure refreshes that account's temporary token once, instead of silently switching accounts. If all accounts are unavailable, inference fails with an actionable error rather than guessing quota balances.

During a chat, quota failover retains the same SDK conversation thread/history. Each CLI `ask` process begins anew and independently checks quota; account state is kept only in memory. Quota checks can be slightly stale within the configured cache duration; set `FRONTIER_QUOTA_CACHE_SECONDS=0` for a preflight before every request. This client does not bypass Kaggle's account limits or guarantee that a quota check reserves budget.

## Python API

```python
from frontier_client import FrontierClient

client = FrontierClient(model="MODEL_ID")
response = client.prompt("Hello!")
print(response.text, response.usage, response.metadata.get("account"))
response = client.prompt("Describe this.", image="photo.png")

with client.chat(system_prompt="Be concise.") as chat:
    chat.send("Explain attention.")
    print(chat.send("Simplify that.").text)
```

`FrontierClient` also accepts `logging=False`, `log_dir=...`, `log_prompts=True`, `temperature=...`, `reasoning=...`, `api="genai"`, and `schema=YourDataclass` when supported by the specific model. All model calls use Kaggle Model Proxy SDK adapters, not direct provider APIs.

## Logging and privacy

By default requests append to `logs/requests.jsonl` with request ID, account **alias**, attempted accounts, model, response, real SDK usage/cost when supplied, latency, status, and sanitized errors. The prompt text is omitted unless `log_prompts=True`, but model responses are logged. No Kaggle account tokens, Proxy tokens, authorization headers, Base64 image data or raw exception bodies are logged. `--no-log` disables writing requests. Chat export explicitly writes prompts and responses to the named JSON file.

Only locally configured `LLMS_AVAILABLE` models are offered. Kaggle's hosted catalog may contain models unavailable to local Proxy credentials. If a model has no local permission, Kaggle will reject it. `LLMS_AVAILABLE` is a local allowlist, not a claim of account authorization.

## Troubleshooting

- Missing account: set `KAGGLE_API_TOKEN` or `KAGGLE_ACCOUNTS` and named tokens in `.env`.
- `frontier quota` unavailable: ensure the Kaggle account has Model Proxy access and completed any necessary identity verification.
- Missing `LLMS_AVAILABLE`: supply your local model allowlist (not Kaggle's complete hosted catalog).
- No eligible account: inspect `frontier quota`, verify your threshold and wait for the normal Kaggle refill. Multiple tokens on the same account may share quota.
- Proxy authentication error: the client automatically re-mints the temporary Proxy token once.
- Model unavailable: choose a locally permitted ID and check vision support for image prompts.
- Account login through Kaggle CLI remains optional: `kaggle auth login`. It does **not** replace the tokens explicitly listed in `.env`.

Run `python -m pytest -q` for mocked tests; they never use real credentials or paid inference. The first-party model/token endpoints require network access and cannot be fully verified without real Kaggle credentials.

## Test environments with a local `.env`

Unit tests explicitly isolate the missing-local-model case from any developer
`.env` file. Running `frontier models list` normally **should** discover models
from your `.env`; clearing `LLMS_AVAILABLE` in a test is not enough by itself.
To run tests on WSL/Windows-mounted directories that disallow pytest cache writes:

```bash
python -m pytest -q -p no:cacheprovider
```

If an earlier version of the bundle stopped after staging files due to this
test, see `APPLY_TO_BRANCH.md` for the safe `--resume` command.
