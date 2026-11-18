"""Adapters for local tabular/media datasets and authorized Kaggle datasets."""
from __future__ import annotations

import csv
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Sample:
    sample_id: str
    question: str = ""
    image: Any = None
    audio: Any = None
    reference_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_kaggle_id(identifier: str) -> tuple[str, str]:
    parts = identifier.strip().strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Kaggle dataset ID must be owner/dataset-name")
    return parts[0], parts[1]


def _records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", [data])
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of records or an object with 'data'")
        yield from data
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)
    elif suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet loading requires the 'data' extra: pip install -e '.[data]'") from exc
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1000):
            yield from batch.to_pylist()
    else:
        raise ValueError(f"Unsupported dataset file format: {path.suffix}")


def _sample(row: dict[str, Any], index: int, columns: dict[str, str], root: Path) -> Sample:
    def get(field_name: str, default: Any = None) -> Any:
        return row.get(columns.get(field_name, field_name), default)
    def media(value: Any) -> Any:
        if value is None or (isinstance(value, str) and value == ""):
            return None
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve() if candidate.exists() else value
    reserved = {columns.get(k, k) for k in ("id", "question", "image", "audio", "reference_answer")}
    return Sample(
        sample_id=str(get("id", index)), question=str(get("question", "") or ""),
        image=media(get("image")), audio=media(get("audio")),
        reference_answer=None if get("reference_answer") is None else str(get("reference_answer")),
        metadata={k: v for k, v in row.items() if k not in reserved},
    )


def load_local_dataset(path: str | Path, *, columns: dict[str, str] | None = None,
                       max_samples: int | None = None) -> Iterator[Sample]:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    mapping = columns or {}
    if root.is_dir():
        metadata_file = next((p for p in (root / "metadata.jsonl", root / "metadata.csv", root / "metadata.json") if p.is_file()), None)
        if metadata_file:
            for i, row in enumerate(_records(metadata_file)):
                row = dict(row)
                if not any(k in row for k in (mapping.get("image", "image"), mapping.get("audio", "audio"))):
                    media_files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".wav", ".mp3", ".flac"})
                    for media in media_files:
                        if media.stem == str(row.get(mapping.get("id", "id"), i)):
                            key = "image" if media.suffix.lower() in {".jpg", ".jpeg", ".png"} else "audio"
                            row[mapping.get(key, key)] = str(media.relative_to(root))
                            break
                yield _sample(row, i, mapping, root)
                if max_samples is not None and i + 1 >= max_samples:
                    break
            return
        media_files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".wav", ".mp3", ".flac"})
        for i, media in enumerate(media_files):
            key = "image" if media.suffix.lower() in {".jpg", ".jpeg", ".png"} else "audio"
            yield Sample(sample_id=media.stem, image=media if key == "image" else None,
                         audio=media if key == "audio" else None, metadata={"path": str(media)})
            if max_samples is not None and i + 1 >= max_samples:
                break
        return
    for i, row in enumerate(_records(root)):
        yield _sample(row, i, mapping, root.parent)
        if max_samples is not None and i + 1 >= max_samples:
            break


def load_kaggle_dataset(identifier: str, *, version: int | None = None, data_file: str | None = None,
                        columns: dict[str, str] | None = None,
                        max_samples: int | None = None) -> Iterator[Sample]:
    parse_kaggle_id(identifier)
    try:
        import kagglehub
    except Exception as exc:
        raise RuntimeError("Kaggle dataset loading requires KaggleHub: install with `python -m pip install -e '.[kaggle]'`") from exc
    if version is not None and "version" not in inspect.signature(kagglehub.dataset_download).parameters:
        raise ValueError("The installed KaggleHub dataset_download API does not support version selection; omit `dataset.version` to use the current authorized dataset version.")
    try:
        kwargs = {"handle": identifier}
        if version is not None:
            kwargs["version"] = version
        location = Path(kagglehub.dataset_download(**kwargs))
    except Exception as exc:
        text = str(exc).lower()
        if any(term in text for term in ("403", "404", "permission", "forbidden", "unauthorized", "not found", "access denied")):
            raise PermissionError(f"Kaggle dataset {identifier!r} is unavailable to this account. Check the ID and request dataset access from its owner.") from exc
        raise RuntimeError(f"Could not download Kaggle dataset {identifier!r}: {exc}") from exc
    selected = (location / data_file).resolve() if data_file else location
    if data_file and not selected.exists():
        raise FileNotFoundError(f"Requested file {data_file!r} not found in downloaded dataset {identifier}")
    yield from load_local_dataset(selected, columns=columns, max_samples=max_samples)


def load_dataset(config: dict[str, Any], *, max_samples: int | None = None) -> Iterator[Sample]:
    limit = max_samples if max_samples is not None else config.get("max_samples")
    dataset = config.get("dataset", config)
    source = dataset.get("source", config.get("source", "local"))
    if source == "local":
        yield from load_local_dataset(dataset["path"], columns=dataset.get("columns"), max_samples=limit)
    elif source == "kaggle":
        yield from load_kaggle_dataset(dataset["id"], version=dataset.get("version"),
                                       data_file=dataset.get("data_file"), columns=dataset.get("columns"),
                                       max_samples=limit)
    else:
        raise ValueError(f"Unsupported dataset source: {source}")
