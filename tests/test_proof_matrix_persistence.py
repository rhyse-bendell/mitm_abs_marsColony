"""Proof-matrix persistence and the provision-traceability fields of communication_exchange."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import proof_matrix as pm  # noqa: E402

from modules.simulation import SimulationState  # noqa: E402


def _job(out, name="A_neutral", seed=0, existing="skip", mode="compact_panel", duration=3):
    return (name, seed, duration, str(out), None, mode, existing)


def test_24_raw_session_and_summary_are_persisted_under_the_requested_layout(tmp_path):
    name, seed, summary_path, status = pm._run(_job(tmp_path))
    assert status == "ran"
    run_root, expected_summary = pm.run_paths(tmp_path, "A_neutral", 0)
    assert Path(summary_path) == expected_summary and expected_summary.exists()
    assert run_root == tmp_path / "runs" / "A_neutral" / "seed_0"
    summary = json.loads(expected_summary.read_text(encoding="utf-8"))
    session = Path(summary["session_folder"])
    assert session.exists() and run_root in session.parents  # the real, persistent session path
    assert (session / "session_manifest.json").exists() and (session / "logs").is_dir()


def test_25_nothing_is_cleaned_up_or_placed_in_a_temp_dir(tmp_path):
    pm._run(_job(tmp_path))
    summary = json.loads(pm.run_paths(tmp_path, "A_neutral", 0)[1].read_text(encoding="utf-8"))
    assert Path(summary["session_folder"]).exists()  # still there after the run returned
    assert not str(summary["session_folder"]).startswith(tempfile.gettempdir()) or str(tmp_path).startswith(tempfile.gettempdir())
    assert summary["run_root"] == str(pm.run_paths(tmp_path, "A_neutral", 0)[0])


def test_26_each_condition_seed_has_its_own_unique_root(tmp_path):
    roots = {pm.run_paths(tmp_path, n, s)[0] for n in pm.CONDITIONS for s in (0, 1, 2)}
    assert len(roots) == len(pm.CONDITIONS) * 3
    pm._run(_job(tmp_path, seed=0))
    pm._run(_job(tmp_path, seed=1))
    sessions = {json.loads(pm.run_paths(tmp_path, "A_neutral", s)[1].read_text(encoding="utf-8"))["session_folder"] for s in (0, 1)}
    assert len(sessions) == 2


def test_27_existing_output_policy_skip(tmp_path):
    pm._run(_job(tmp_path))
    path = pm.run_paths(tmp_path, "A_neutral", 0)[1]
    before = path.read_text(encoding="utf-8")
    assert pm._run(_job(tmp_path, existing="skip"))[3] == "skipped"
    assert path.read_text(encoding="utf-8") == before


def test_28_existing_output_policy_error_and_overwrite(tmp_path):
    pm._run(_job(tmp_path))
    with pytest.raises(FileExistsError):
        pm._run(_job(tmp_path, existing="error"))
    assert pm._run(_job(tmp_path, existing="overwrite"))[3] == "ran"
    run_root = pm.run_paths(tmp_path, "A_neutral", 0)[0]
    assert len(list((run_root / "Outputs").iterdir())) == 1  # the old session was replaced, not accumulated


def test_29_summary_records_the_explicit_profile_mode(tmp_path):
    pm._run(_job(tmp_path))
    summary = json.loads(pm.run_paths(tmp_path, "A_neutral", 0)[1].read_text(encoding="utf-8"))
    assert summary["profile_mode"] == "compact_panel" and summary["manifest_condition"]["profile_mode"] == "compact_panel"
    assert pm._settings("A_neutral", None)["profile_mode"] == "compact_panel"


def test_30_communication_exchange_names_exact_provided_elements_and_solicitation():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="comm", seed=1)
        exchanges = []
        state.logger.register_event_listener(lambda e: exchanges.append(e["payload_data"]) if e["event_type"] == "communication_exchange" else None)
        try:
            for _ in range(300):
                state.update(1.0)
            assert exchanges, "expected at least one exchange in 300 s"
            for ex in exchanges:
                assert isinstance(ex["provided_element_ids"], list) and isinstance(ex["solicited"], bool)
                assert all(i.split(":", 1)[0] in {"data", "information", "rule"} for i in ex["provided_element_ids"])
                if ex["solicited"]:
                    assert ex["provided_element_ids"]
                if any(t in {"TDP", "TIP", "TKP"} for t in ex["message_types"]):
                    assert ex["provided_element_ids"]
            # direct: a provision answering an explicit prior request is solicited; an unsolicited push is not
            giver = next(a for a in state.agents if a.mental_model["information"])
            peer = next(a for a in state.agents if a is not giver)
            eid = f"information:{next(iter(giver.mental_model['information'])).id}"
            giver.theory_of_mind.setdefault(peer.name, {"goals": [], "knowledge_ids": set(), "last_seen": None})["requested_dik_ids"] = {eid}
            giver.generate_message(recipient_name=peer.name, include_requests=False)
            assert giver._provision_solicited is True and giver._provided_element_ids == [eid]
            giver.theory_of_mind[peer.name]["requested_dik_ids"] = set()
            giver.source_memory_state.setdefault(f"{giver.role}_Info", {})["pending_role_share_ids"] = [eid]
            giver.generate_message(recipient_name=peer.name, include_requests=False)
            assert giver._provision_solicited is False
        finally:
            state.stop()