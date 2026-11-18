from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from frontier_bench.datasets import Sample, load_dataset
from .metrics import METRICS


def run_benchmark(config: dict[str, Any], *, client_factory: Callable[..., Any] | None = None,
                  scorers: dict[str, Callable[[str, str], float]] | None = None) -> dict[str, Any]:
    """Evaluate configured models; persist one JSONL record for every attempted sample."""
    if client_factory is None:
        from frontier_bench.models import KaggleModelClient
        client_factory = KaggleModelClient
    dataset = config["dataset"]
    output = Path(config["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    custom_metrics = dict(METRICS)
    custom_metrics.update(scorers or {})
    metrics = config.get("metrics", [])
    unknown = set(metrics) - custom_metrics.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(sorted(unknown))}")
    all_records: list[dict[str, Any]] = []
    for model_id in config.get("models", []):
        retry = config.get("retry", {})
        client_error = None
        try:
            client = client_factory(model_id, system_prompt=config.get("system_prompt"),
                                    generation=config.get("generation", {}),
                                    retries=retry.get("retries", 2),
                                    backoff_seconds=retry.get("backoff_seconds", 1.0))
        except Exception as exc:
            client, client_error = None, exc
        for sample in load_dataset(config):
            prompt = config.get("prompt_template", "{question}").format(question=sample.question,
                                                                           sample_id=sample.sample_id,
                                                                           **sample.metadata)
            record: dict[str, Any] = {
                "run_id": run_id, "model_id": model_id,
                "dataset_source": dataset.get("source", "local"),
                "dataset_identifier": dataset.get("id", dataset.get("path")),
                "sample_id": sample.sample_id, "prompt": prompt,
                "image_input": str(sample.image) if sample.image is not None else None,
                "audio_input": str(sample.audio) if sample.audio is not None else None,
                "sample_metadata": sample.metadata,
                "reference_answer": sample.reference_answer, "prediction": None,
                "metrics": {}, "latency_seconds": None, "usage": {}, "error": None,
            }
            start = time.perf_counter()
            try:
                if client_error is not None:
                    raise client_error
                response = client.prompt(prompt, image=sample.image, audio=sample.audio)
                record["prediction"] = response.text
                record["latency_seconds"] = response.latency_seconds
                record["usage"] = response.usage
                if sample.reference_answer is not None:
                    for name in metrics:
                        record["metrics"][name] = custom_metrics[name](sample.reference_answer, response.text)
            except Exception as exc:
                record["latency_seconds"] = time.perf_counter() - start
                from frontier_bench.models.client import _redact
                record["error"] = {"type": type(exc).__name__, "message": _redact(str(exc))}
            all_records.append(record)
    jsonl = output / f"{run_id}.jsonl"
    with jsonl.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    summary = {
        "run_id": run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": config, "records_file": jsonl.name,
        "attempted": len(all_records), "failed": sum(r["error"] is not None for r in all_records),
        "metric_means": {},
    }
    for metric in metrics:
        values = [r["metrics"][metric] for r in all_records if metric in r["metrics"]]
        summary["metric_means"][metric] = sum(values) / len(values) if values else None
    report = output / f"{run_id}.summary.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary["summary_file"] = str(report)
    summary["records_file"] = str(jsonl)
    return summary
