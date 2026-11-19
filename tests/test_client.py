import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier_client import FrontierClient
from frontier_client.errors import AuthenticationError, ImageInputError, ModelRequestError, UnsupportedParameterError


def _logs(tmp_path):
    return [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text().splitlines()]


def test_stateless_text_prompt_and_usage(client, sdk, tmp_path):
    first = client.prompt("hello")
    second = client.prompt("again")
    assert first.text == "reply:hello"
    assert first.model == "fake/model" and first.request_id != second.request_id
    assert first.usage["input_tokens"] == 2
    assert first.usage["total_cost"] == 300 / 1e9
    assert sdk.model.calls[0]["thread"] is not sdk.model.calls[1]["thread"]
    records = _logs(tmp_path)
    assert len(records) == 2
    assert records[0]["input"]["prompt"] == "hello"
    assert records[0]["status"] == "success"
    assert records[0]["latency_seconds"] >= 0


def test_image_prompt_factory(client, sdk, png, tmp_path):
    response = client.prompt("describe", image=png)
    assert response.text == "reply:describe"
    assert sdk.model.calls[0]["kwargs"]["image"] == {"local": str(png)}
    log = _logs(tmp_path)[0]
    assert log["input"]["type"] == "text_image"
    assert log["input"]["images"] == [str(png)]
    assert "base64" not in json.dumps(log)


def test_image_url(client, sdk):
    client.prompt("describe", image="https://example.org/pic.png")
    assert sdk.model.calls[-1]["kwargs"]["image"] == {"url": "https://example.org/pic.png"}


@pytest.mark.parametrize("image", ["missing.png", "invalid.gif", "ftp://foo/bar.png", "https://127.0.0.1/pic.png", "http://example.org/pic.png"])
def test_invalid_images_are_logged(client, tmp_path, image):
    with pytest.raises(ImageInputError):
        client.prompt("describe", image=image)
    record = _logs(tmp_path)[-1]
    assert record["status"] == "failed"
    assert record["error"]["type"] == "ImageInputError"


def test_bad_file_signature_and_size(client, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not a png")
    with pytest.raises(ImageInputError, match="valid PNG"):
        client.prompt("hi", image=bad)
    large = tmp_path / "too_big.png"
    with large.open("wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.truncate(21 * 1024 * 1024)
    with pytest.raises(ImageInputError, match="20 MB"):
        client.prompt("hi", image=large)


def test_response_structured(client, tmp_path):
    response = client.prompt("make JSON", schema=dict)
    assert response.structured == {"reply": "reply:make JSON"}
    assert response.text == '{"reply": "reply:make JSON"}'
    assert _logs(tmp_path)[0]["output"]["structured"] == response.structured


def test_failed_request_redacted(client, sdk, tmp_path):
    sdk.model.failure = RuntimeError("401 unauthorized: secret=top-secret")
    with pytest.raises(AuthenticationError, match="auth"):
        client.prompt("hi")
    data = (tmp_path / "requests.jsonl").read_text()
    assert "top-secret" not in data
    assert '"status":"failed"' in data
    assert '"error"' in data


def test_opt_out_logging(tmp_path, sdk):
    client = FrontierClient("fake/model", sdk=sdk, logging=False)
    client.prompt("hi")
    assert not (tmp_path / "requests.jsonl").exists()


def test_no_prompt_logging_default(tmp_path, sdk):
    client = FrontierClient("fake/model", sdk=sdk, log_dir=tmp_path)
    client.prompt("private prompt")
    assert "private prompt" not in _logs(tmp_path)[0]["input"].values()
    assert _logs(tmp_path)[0]["input"]["prompt"] is None


def test_parameter_validation(sdk, tmp_path):
    with pytest.raises(UnsupportedParameterError):
        FrontierClient("fake/model", sdk=sdk, temperature=3)
    with pytest.raises(UnsupportedParameterError):
        FrontierClient("fake/model", sdk=sdk, reasoning="extreme")
    sdk.model.support_temperature = False
    with pytest.raises(UnsupportedParameterError, match="temperature"):
        FrontierClient("fake/model", sdk=sdk, temperature=0.2)


def test_model_init_failure(sdk):
    with pytest.raises(ModelRequestError):
        FrontierClient("missing", sdk=sdk, logging=False)
