import json
from pathlib import Path

from typer.testing import CliRunner

from frontier_bench.cli import app
from frontier_bench.evaluation import run_benchmark
from frontier_bench.evaluation.metrics import exact_match, word_error_rate
from frontier_bench.models import ModelResponse


class FakeClient:
    def __init__(self, model_id, **kwargs): self.model_id = model_id
    def prompt(self, prompt, **kwargs):
        if prompt == "bad": raise RuntimeError("offline failure")
        return ModelResponse("yes", .02, {"input_tokens": 1})


def test_metrics():
    assert exact_match("Yes", "yes") == 1
    assert word_error_rate("a b", "a c") == .5


def test_failed_samples_are_serialized(tmp_path):
    data = tmp_path / "data.json"
    data.write_text(json.dumps([{"question": "ok", "answer": "yes"}, {"question": "bad", "answer": "no"}]), encoding="utf-8")
    summary = run_benchmark({"dataset": {"source": "local", "path": str(data), "columns": {"reference_answer": "answer"}},
                             "models": ["m"], "prompt_template": "{question}", "metrics": ["exact_match"],
                             "output_dir": str(tmp_path / "out")}, client_factory=FakeClient)
    assert summary["attempted"] == 2 and summary["failed"] == 1
    records = [json.loads(line) for line in Path(summary["records_file"]).read_text(encoding="utf-8").splitlines()]
    assert records[1]["error"]["type"] == "RuntimeError"
    assert records[0]["metrics"]["exact_match"] == 1


def test_model_setup_failure_is_counted_for_samples(tmp_path):
    data = tmp_path / "one.jsonl"
    data.write_text('{"question":"q"}\n', encoding="utf-8")
    def unavailable(*args, **kwargs): raise RuntimeError("model is unavailable")
    summary = run_benchmark({"dataset": {"source": "local", "path": str(data)},
                             "models": ["missing"], "output_dir": str(tmp_path / "out")},
                            client_factory=unavailable)
    assert summary["attempted"] == summary["failed"] == 1
    record = json.loads(Path(summary["records_file"]).read_text(encoding="utf-8"))
    assert record["error"]["message"] == "model is unavailable"


def test_cli_help_and_doctor():
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["benchmark", "run", "--help"]).exit_code == 0
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    selected = runner.invoke(app, ["doctor", "--python", __import__("sys").executable])
    assert selected.exit_code == 0
    assert Path(__import__("sys").executable).name in selected.stdout


def test_cli_models_and_quota_dispatch(monkeypatch):
    import frontier_bench.cli as cli
    calls = []
    monkeypatch.setattr(cli, "_run_cli", lambda args: calls.append(args) or 0)
    runner = CliRunner()
    assert runner.invoke(app, ["models", "list"]).exit_code == 0
    assert runner.invoke(app, ["quota"]).exit_code == 0
    assert calls == [["benchmarks", "tasks", "models"], ["benchmarks", "quota"]]


def test_cli_auth_check_reports_both_layers(monkeypatch):
    import frontier_bench.cli as cli
    monkeypatch.setattr(cli, "auth_status", lambda: {
        "kaggle_cli": True, "account_auth": True, "proxy_credential_configured": False,
        "proxy_url_configured": False, "account_credential_files": [], "proxy_hint": "proxy separate"})
    result = CliRunner().invoke(app, ["auth", "check"])
    assert result.exit_code == 0
    assert "Model Proxy credentials are missing or incomplete" in result.stdout


def test_cli_benchmark_run_dispatch(tmp_path, monkeypatch):
    import frontier_bench.evaluation as evaluation
    config = tmp_path / "config.yaml"
    config.write_text("models: [configured]\ndataset:\n  source: local\n  path: data.jsonl\n", encoding="utf-8")
    observed = {}
    def fake_run(cfg):
        observed.update(cfg)
        return {"attempted": 0, "failed": 0, "metric_means": {}}
    monkeypatch.setattr(evaluation, "run_benchmark", fake_run)
    result = CliRunner().invoke(app, ["benchmark", "run", "--config", str(config), "--model", "cli-model"])
    assert result.exit_code == 0
    assert observed["models"] == ["cli-model"]


def test_cli_chat_opens_isolated_session(monkeypatch):
    from contextlib import nullcontext
    import frontier_bench.models as models
    class DummyClient:
        def __init__(self, *args, **kwargs): pass
        def conversation(self): return nullcontext(self)
    monkeypatch.setattr(models, "KaggleModelClient", DummyClient)
    result = CliRunner().invoke(app, ["chat", "--model", "test-model"], input="/quit\n")
    assert result.exit_code == 0
