# Frontier Model Benchmarking

A portable Python CLI for evaluating models through Kaggle Benchmarks and for interactive Kaggle-backed chat. The original research scripts and bundled audio samples remain under [`frontier-bench-smoke/`](frontier-bench-smoke/).

## Requirements and setup

Python 3.11+ is recommended; local-only package features target Python 3.10+, while the current official Kaggle CLI requires Python 3.11+. Kaggle-backed features additionally need the official Kaggle CLI, Kaggle Benchmarks SDK, and Kaggle credentials. Dataset and result tooling can be used locally without Kaggle credentials when the selected model client is mocked or replaced.

Install in an existing Conda environment:

```bash
conda activate YOUR_ENV
python -m pip install -e ".[dev,kaggle]"
```

Or create a new environment with Conda:

```bash
conda create -n frontier-bench python=3.11 pip
conda activate frontier-bench
python -m pip install -e ".[dev,kaggle]"
```

For an existing `.venv`, activate it and run the same pip command. To create one, use `python -m venv .venv`, then activate `.venv` (Windows: `.venv\\Scripts\\activate`; Linux/WSL: `source .venv/bin/activate`). The CLI uses the interpreter that launches it; `frontier doctor --python PATH` inspects another executable, while `--conda-env NAME` and `--conda-prefix PATH` inspect Conda environments without assuming Conda exists everywhere. The project never creates or replaces environments automatically. The Kaggle extra installs the official [Kaggle CLI source](https://github.com/Kaggle/kaggle-cli) at a pinned commit because the PyPI release available during validation did not yet include the documented `benchmarks` command group.

The local package supports Python 3.10+; Kaggle extras require Python 3.11+. Optional Parquet support uses PyArrow: install it with `python -m pip install -e ".[data]"`, or `python -m pip install -e ".[data,kaggle]"` for the full stack. On Python 3.10/3.11, the data extra caps PyArrow below 12 to remain compatible with the existing research environment; Python 3.12+ uses a newer PyArrow. Kaggle SDK and CLI dependencies are optional so local tooling remains usable without Kaggle.

## Kaggle authentication and model availability

Authenticate Kaggle account access using the supported CLI:

```bash
kaggle auth login
kaggle benchmarks auth -y --env-file .env
frontier auth check
frontier quota
frontier models list
```

Kaggle account credentials control dataset and hosted-task access. The temporary Model Proxy credentials issued by `kaggle benchmarks auth` are separate and may expire. Do not commit `.env`, Kaggle credential files, or private data. Copy `frontier-bench-smoke/.env.example` only as a historical reference; the new framework recognizes a root `.env` for process configuration and never prints its values.

See the [Kaggle Benchmarks SDK](https://github.com/Kaggle/kaggle-benchmarks), [Kaggle CLI benchmark guide](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md), and [Kaggle Benchmarks documentation](https://www.kaggle.com/docs/benchmarks) for the current service API and usage rules.

`frontier models list` asks the installed Kaggle CLI for its current model catalog. No fixed list in this package is authoritative. The official CLI guide says the local-development token minted by `kaggle benchmarks init` is restricted to the `LLMS_AVAILABLE` subset; hosted task runs can use the full catalog. Model access, modalities, generation options, quota, and pricing depend on the account and current Kaggle service policy.

## Local benchmark

Map your dataset columns in YAML. JSON/JSONL, CSV, Parquet and directories of media files with optional metadata are supported.

```yaml
# configs/examples/local.yaml
dataset:
  source: local
  path: ../../data/questions.jsonl
  columns:
    id: id
    question: prompt
    reference_answer: answer
models: [MODEL_ID]
prompt_template: "Answer the question accurately.\n\n{question}"
max_samples: 10
metrics: [exact_match]
output_dir: ../../runs
generation:
  temperature: 0
```

```bash
frontier benchmark run --config configs/examples/local.yaml
```

The dataset path is resolved relative to the YAML file. CLI options override environment variables, which override YAML values, which override built-in defaults. The YAML file itself is selected by `--config`; `FRONTIER_*` variables can override model, dataset path and result directory. Each run writes sample records to JSONL and a summary/configuration snapshot to JSON. Failed requests remain in the record and are included in failure counts. Metrics are computed only when a reference answer exists.

Built-in metrics are exact match, normalized exact match, and word error rate (`wer`). Custom scorers can be supplied to the Python runner API. Image/audio columns are carried through the common sample representation; actual modality encoding depends on the selected SDK/model capabilities and is not silently guessed.

## Kaggle datasets and hosted runs

For local evaluation using an authorized public or private Kaggle dataset, set `source: kaggle`, `id: owner/dataset-name`, and optionally `version` when the installed KaggleHub API supports it. KaggleHub 1.0.3 (the version checked here) does not expose a version argument, so the loader reports a clear error if `dataset.version` is requested. The loader uses `kagglehub.dataset_download`; authorization is still enforced by Kaggle. Access denied errors explain that the account needs permission.

```yaml
dataset:
  source: kaggle
  id: owner/public-dataset
  data_file: questions.jsonl
```

The loader downloads/reads data on the local machine. For hosted evaluation, first publish or upload the data to Kaggle explicitly, then use the Kaggle CLI task workflow and attach a dataset with `-d owner/dataset-name`. For a Kaggle Benchmarks task script (such as the retained ASR task), the workflow is:

```bash
kaggle benchmarks tasks push TASK_SLUG -f path/to/task.py -d owner/dataset-name
kaggle benchmarks tasks run TASK_SLUG -m MODEL_ID
kaggle benchmarks tasks status TASK_SLUG
```

Local files are not visible to hosted jobs automatically, and this package does not upload them implicitly. Check `kaggle benchmarks tasks --help` and the current Kaggle Benchmarks guide before submitting, because hosted-task options can change.

The preserved `frontier-bench-smoke/execution/benchmarks/` scripts contain existing text probes and ASR implementations, including ViMedCSS and Meddies samples. Their historical SDK calls may require adaptation to the current SDK before use.

## Chat

```bash
frontier chat --model MODEL_ID
```

Enter `/reset` to clear the current conversation, `/model MODEL_ID` to start an isolated session on another model, `/export FILE.json` or `/export FILE.md` to save the current session, and `/quit` to exit. This uses the SDK's supported chat context and system-instruction API. It is a local-development convenience, not a claim that Kaggle authorizes unrestricted general-purpose chatbot use. Local proxy credentials can be restricted to the models in `LLMS_AVAILABLE`, and all requests remain subject to Kaggle's terms and quota. The current SDK has text, image and audio prompt inputs; actual model modality support varies. For broader model runs, use an authorized hosted task.

## Configuration and troubleshooting

Supported source values are `local` and `kaggle`. Common keys: `dataset.path` or `dataset.id`, `dataset.version`, `dataset.columns` (`id`, `question`, `image`, `audio`, `reference_answer`), `models`, `prompt_template`, `max_samples`, `metrics`, `output_dir`, `system_prompt`, `generation` (`temperature`, `seed`, `reasoning_effort`, or provider-specific `extra_api_params`), and `retry` (`retries`, `backoff_seconds`). These map to Kaggle Benchmarks SDK prompt options; unsupported provider parameters remain model-dependent. See [`configs/examples/`](configs/examples/).

- `frontier doctor` checks the running interpreter and optional Kaggle CLI/SDK availability.
- `frontier auth check` checks Kaggle CLI authentication and looks for Model Proxy credential variables without showing secret values.
- If proxy credentials expire, run `kaggle benchmarks auth -y --env-file .env` again.
- `kaggle benchmarks quota` reports service quota; inference can fail even with valid account credentials.
- A private dataset permission error means the authenticated Kaggle account lacks access; request access from its owner.
- Unsupported model/parameter errors are returned by the SDK and included per sample; remove unsupported options or select a supported model.

## Security

The repository ignores `.env`, `.venv`, Kaggle credentials, caches, generated runs, and conventional root `data/` and `datasets/` directories. Add other private-data locations to `.gitignore` before placing data there. Never place credentials in YAML, source control, or prompt data. The CLI masks credential values and records errors without printing environment contents.

## Development

```bash
python -m pip install -e ".[dev,kaggle]"
pytest
```

Automated tests use local fixtures and mocks; they do not need credentials or make inference requests.
