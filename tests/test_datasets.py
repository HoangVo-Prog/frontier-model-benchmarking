import json
import importlib.util
from pathlib import Path
import sys
import types

import pytest

from frontier_bench.datasets import load_dataset, load_kaggle_dataset, load_local_dataset, parse_kaggle_id


def test_local_jsonl_mapping_and_limit(tmp_path):
    file = tmp_path / "rows.jsonl"
    file.write_text('{"uid": 1, "prompt": "Q", "target": "A", "extra": 4}\n{"uid": 2, "prompt": "Q2", "target": "B"}\n', encoding="utf-8")
    rows = list(load_local_dataset(file, columns={"id": "uid", "question": "prompt", "reference_answer": "target"}, max_samples=1))
    assert len(rows) == 1
    assert (rows[0].sample_id, rows[0].question, rows[0].reference_answer) == ("1", "Q", "A")
    assert rows[0].metadata == {"extra": 4}


def test_csv_loading(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("question,answer\nq,a\n", encoding="utf-8")
    rows = list(load_local_dataset(path, columns={"reference_answer": "answer"}))
    assert rows[0].question == "q"
    assert rows[0].reference_answer == "a"


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="optional Parquet dependency")
def test_parquet_loading(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = tmp_path / "data.parquet"
    pq.write_table(pa.Table.from_pylist([{"prompt": "q", "target": "a"}]), path)
    rows = list(load_local_dataset(path, columns={"question": "prompt", "reference_answer": "target"}))
    assert (rows[0].question, rows[0].reference_answer) == ("q", "a")


def test_kaggle_id_validation():
    assert parse_kaggle_id(" owner/dataset ") == ("owner", "dataset")
    with pytest.raises(ValueError):
        parse_kaggle_id("dataset")


def test_private_kaggle_permission_message(monkeypatch):
    module = types.SimpleNamespace(dataset_download=lambda **kw: (_ for _ in ()).throw(Exception("403 Forbidden")))
    monkeypatch.setitem(sys.modules, "kagglehub", module)
    with pytest.raises(PermissionError, match="request dataset access"):
        list(load_dataset({"dataset": {"source": "kaggle", "id": "owner/private"}}))


def test_version_selection_is_checked_against_installed_kagglehub(monkeypatch):
    module = types.SimpleNamespace(dataset_download=lambda handle: handle)
    monkeypatch.setitem(sys.modules, "kagglehub", module)
    with pytest.raises(ValueError, match="does not support version selection"):
        list(load_kaggle_dataset("owner/dataset", version=2))
