import json
import pytest


def test_chat_context_isolation_interleaved_and_reset(client, sdk, png, tmp_path):
    with client.chat(system_prompt="Short answers.") as a, client.chat() as b:
        r1 = a.send("one")
        r2 = b.send("two", image=png)
        a.send("three")
        assert a.session_id != b.session_id
        assert len(a.history) == 4 and len(b.history) == 2
        assert sdk.model.calls[0]["thread"] is sdk.model.calls[2]["thread"]
        assert sdk.model.calls[0]["thread"] is not sdk.model.calls[1]["thread"]
        assert sdk.model.calls[0]["thread"].messages[0].content == "Short answers."
        previous_thread = sdk.model.calls[0]["thread"]
        a.reset()
        assert a.history == []
        assert a.send("four").text == "reply:four"
        assert sdk.model.calls[-1]["thread"] is not previous_thread
        exported = b.export(tmp_path / "session.json")
        data = json.loads(exported.read_text())
        assert data["model"] == "fake/model"
        assert data["messages"][0]["image"] == str(png)
    with pytest.raises(RuntimeError, match="with client.chat"):
        a.send("closed")


def test_chat_failure_does_not_poison_context(client, sdk):
    with client.chat() as chat:
        sdk.model.failure = RuntimeError("503 backend unavailable")
        with pytest.raises(Exception):
            chat.send("bad")
        sdk.model.failure = None
        chat.send("good")
        assert [m.content for m in chat._thread.messages] == ["good", "reply:good"]


def test_chat_export_only_json(client):
    with client.chat() as chat:
        with pytest.raises(ValueError):
            chat.export("chat.md")
