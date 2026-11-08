"""Everything here runs on the mock backend: no weights, no network."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from cev import Cev, CevError, info

TICKET = {"subject": "App crashes on login", "body": "The app shows an error and crashes when I tap Log in."}
QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "Payments, invoices, refunds", "tech": "Bugs, crashes, error messages"},
        "task": "router.team",
    },
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "severity": {"type": "score", "instructions": "How severe is the issue?", "criteria": ["Cosmetic", "Degraded", "Blocking"]},
}


@pytest.fixture
def cev():
    with Cev.mock() as c:
        yield c


def code(excinfo) -> str:
    return excinfo.value.code


def test_info_describes_the_build():
    i = info()
    assert i["version"].count(".") == 2
    assert isinstance(i["metal"], bool)
    assert i["cores"] >= 1


def test_decide_answers_every_question_with_its_type(cev):
    assert cev.model["id"] == "cev-mock"
    assert cev.model["threads"] >= 1

    res = cev.decide(TICKET, QUESTIONS)
    assert list(res["answers"]) == ["team", "refund", "severity"]
    assert res["request_id"].startswith("req_")

    team = res["answers"]["team"]
    assert team["choice"] == "tech"
    assert sum(team["probabilities"].values()) == pytest.approx(1, abs=1e-5)
    assert 0 < team["confidence"] <= 1
    assert team["x_cev"]["decision_id"].startswith("dec_")
    assert team["x_cev"]["task"] == "router.team"

    assert 0 <= res["answers"]["refund"]["noul"] <= 1

    severity = res["answers"]["severity"]
    assert 0 <= severity["score"] <= 2
    assert severity["legend"] == {"0": "Cosmetic", "1": "Degraded", "2": "Blocking"}
    assert sum(severity["probabilities"].values()) == pytest.approx(1, abs=1e-5)


def test_the_model_says_its_family_and_what_it_takes(cev):
    assert cev.model["family"] == "cev"
    assert cev.model["modalities"] == ["text"]
    with Cev.mock(media=True) as m:
        assert m.model["modalities"] == ["text", "image", "audio"]


PNG = b"\x89PNG\r\n\x1a\n\x01\x02\x03"
SEEN = {"seen": {"type": "noul", "instructions": "Is something shown?"}}


def test_images_and_audio_go_as_paths_bytes_or_base64(cev, tmp_path):
    with pytest.raises(CevError) as e:
        cev.decide("x", SEEN, images=[PNG])
    assert code(e) == "bad_request"
    with pytest.raises(CevError) as e:
        cev.decide("x", SEEN, audio=b"RIFF....WAVEfmt ")
    assert code(e) == "bad_request"

    path = tmp_path / "shot.png"
    path.write_bytes(PNG)
    with Cev.mock(media=True) as m:
        plain = m.decide("x", SEEN)["usage"]["input_tokens"]
        raw = m.decide("x", SEEN, images=[PNG])["usage"]["input_tokens"]
        assert raw > plain, "the image reached the model"
        b64 = "data:image/png;base64," + __import__("base64").b64encode(PNG).decode()
        for image in (path, bytearray(PNG), memoryview(PNG), b64):
            assert m.decide("x", SEEN, images=[image])["usage"]["input_tokens"] == raw
        m.decide("x", SEEN, audio=b"RIFF....WAVEfmt ")
        with pytest.raises(TypeError):
            m.decide("x", SEEN, images=path)  # one image is still a list
        with pytest.raises(CevError) as e:
            m.decide("x", SEEN, images=["not base64!!"])
        assert code(e) == "bad_request"
        [ex] = m.examples([{"state": "x", "questions": SEEN, "labels": {"seen": True}, "images": [path]}])
        assert ex["response"]["usage"]["input_tokens"] == raw


def test_namespaces_keep_feedback_apart(cev):
    team = {"team": QUESTIONS["team"]}
    acme = cev.decide(TICKET, team, namespace="acme")["answers"]["team"]["x_cev"]
    plain = cev.decide(TICKET, team)["answers"]["team"]["x_cev"]
    assert acme["namespace"] == "acme"
    assert "namespace" not in plain

    assert cev.feedback(decision_id=acme["decision_id"], label="billing")["namespace"] == "acme"
    cev.feedback(decision_id=plain["decision_id"], label="tech")

    assert len(cev.tasks()) == 2, "every namespace"
    [mine] = cev.tasks("acme")
    assert mine["namespace"] == "acme"
    assert len(cev.tasks("")) == 1, "the default namespace"

    [d] = cev.decisions(namespace="acme")
    assert d["namespace"] == "acme"
    assert [r["namespace"] for r in cev.export(namespace="acme")] == ["acme"]

    assert cev.reset_task("router.team") is True, "the default namespace"
    assert cev.reset_task("router.team", "acme") is True
    assert cev.reset_task("router.team", "acme") is False


def test_state_can_be_text_and_criteria_a_list_of_names(cev):
    questions = {"kind": {"type": "choice", "instructions": "What is this about?", "criteria": ["refund", "bug", "other"]}}
    assert cev.decide("Please refund my last invoice.", questions)["answers"]["kind"]["choice"] == "refund"


def test_feedback_is_stored_learned_from_and_exported(cev):
    res = cev.decide(TICKET, QUESTIONS)
    team_id = res["answers"]["team"]["x_cev"]["decision_id"]

    by_id = cev.feedback(decision_id=team_id, label="tech", comment="crash", metadata={"by": "test"})
    assert by_id["task"] == "router.team"
    assert by_id["learned"] is True
    assert by_id["task_examples"] == 1
    assert isinstance(by_id["served_loss"], float)

    by_question = cev.feedback(request_id=res["request_id"], question_id="severity", label=2)
    assert by_question["decision_id"] == res["answers"]["severity"]["x_cev"]["decision_id"]

    # A noul label of False is a label, not a missing one.
    assert cev.feedback(request_id=res["request_id"], question_id="refund", label=False)["learned"] is True

    assert next(t for t in cev.tasks() if t["task"] == "router.team")["examples"] == 1

    detail = cev.decision(team_id)
    assert detail["state"] == TICKET
    assert "Which team should handle this ticket?" in detail["prompt"]
    assert detail["feedback"][0]["comment"] == "crash"
    assert cev.decision("dec_missing") is None

    assert len(cev.decisions()) == 3
    assert len(cev.decisions(task="router.team")) == 1

    rows = cev.export()
    assert len(rows) == 3
    row = next(r for r in rows if r["task"] == "router.team")
    assert row["target"] == [0, 1]
    assert row["options"] == ["billing", "tech"]
    assert len(cev.export(labeled=False)) == 3
    ndjson = cev.export_ndjson(task="router.team")
    assert len(ndjson.splitlines()) == 1
    assert json.loads(ndjson)["decision_id"] == row["decision_id"]

    assert cev.stats()["store"] == {"requests": 1, "decisions": 3, "feedback": 3, "labeled_decisions": 3}

    assert cev.reset_task("router.team") is True
    assert cev.reset_task("router.team") is False
    assert len(cev.export()) == 3, "labels outlive the adapter"


def test_examples_decide_and_then_learn():
    states = ["The app crashes on launch", "I was charged twice", "Error 500 when saving", "Refund my invoice please"]
    with Cev.mock(min_examples=2) as cev:
        results = cev.examples(
            [{"state": s, "questions": {"team": QUESTIONS["team"]}, "labels": {"team": "billing" if i % 2 else "tech"}} for i, s in enumerate(states)]
        )
        assert len(results) == 4
        assert results[3]["feedback"][0]["task_examples"] == 4
        (task,) = cev.tasks()
        assert task["examples"] == 4
        assert task["buffered"] == 4


def test_no_store_answers_cannot_take_feedback(cev):
    res = cev.decide(TICKET, QUESTIONS, no_store=True, debias="none")
    assert res["answers"]["team"]["x_cev"]["decision_id"] == ""
    assert cev.stats()["store"]["decisions"] == 0
    with pytest.raises(CevError) as e:
        cev.feedback(request_id=res["request_id"], question_id="team", label="tech")
    assert code(e) == "not_found"


def test_errors_carry_a_code():
    cev = Cev.mock()
    with pytest.raises(CevError) as e:
        cev.decide("x", {})
    assert code(e) == "bad_request"
    with pytest.raises(CevError) as e:
        cev.decide("x", {"q": {"type": "choice", "instructions": "Pick", "criteria": ["only"]}})
    assert code(e) == "bad_request"
    with pytest.raises(CevError) as e:
        cev.feedback(decision_id="dec_missing", label=True)
    assert code(e) == "not_found"
    res = cev.decide(TICKET, QUESTIONS)
    with pytest.raises(CevError) as e:
        cev.feedback(decision_id=res["answers"]["team"]["x_cev"]["decision_id"], label="sales")
    assert code(e) == "bad_request"
    cev.close()
    with pytest.raises(CevError) as e:
        cev.stats()
    assert code(e) == "closed"
    cev.close()


def test_threads_can_call_in_parallel(cev):
    with ThreadPoolExecutor(8) as pool:
        done = list(pool.map(lambda i: cev.decide({**TICKET, "i": i}, QUESTIONS), range(50)))
    assert len({r["request_id"] for r in done}) == 50
    assert cev.stats()["store"]["requests"] == 50


def test_the_log_persists_in_a_database_file(tmp_path):
    db = str(tmp_path / "cev.db")
    with Cev.mock(db=db) as first:
        res = first.decide(TICKET, QUESTIONS)
        first.feedback(decision_id=res["answers"]["team"]["x_cev"]["decision_id"], label="tech")
    with Cev.mock(db=db) as second:
        assert second.stats()["store"] == {"requests": 1, "decisions": 3, "feedback": 1, "labeled_decisions": 1}
        assert second.tasks()[0]["examples"] == 1, "adapters are restored"


# CEV_TEST_MODEL=Qwen/Qwen3-0.6B pytest   (needs the weights; downloads them if missing)
@pytest.mark.skipif(not os.environ.get("CEV_TEST_MODEL"), reason="set CEV_TEST_MODEL to a model")
def test_a_real_model_routes_a_ticket():
    with Cev.load(os.environ["CEV_TEST_MODEL"]) as cev:
        res = cev.decide(TICKET, QUESTIONS, no_store=True)
        assert res["answers"]["team"]["choice"] == "tech"
        assert res["answers"]["refund"]["noul"] < 0.5
        assert res["usage"]["input_tokens"] > 100
