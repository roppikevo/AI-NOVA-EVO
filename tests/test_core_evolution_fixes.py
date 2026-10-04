"""
Regression tests for core-evolution infrastructure fixes (2026-09-29).

GEN5-CORE-002 was rejected with KeyError: 'config' inside
TrainingRunner.run_screening() — the candidate was never trained,
yet it was recorded as REJECT and copied into passed/.
"""

import json

import pytest

from evo.engine.core_evolution_engine import (
    CoreEvolutionError,
    is_infrastructure_failure,
    resolve_training_metadata,
)

PARENT_CONFIG = {
    "vocab_size": 16384,
    "d_model": 384,
    "d_state": 384,
    "num_layers": 6,
    "conv_kernel": 5,
    "forget_bias": 1.125,
    "learnable_initial_state": False,
}


def _new_candidate():
    # Shape produced by CoreEvolutionEngine.create_candidate()
    return {
        "candidate_id": "GEN5-CORE-002",
        "candidate": "GEN5-CORE-002",
        "type": "core",
        "parent": "GEN4-CORE-002",
        "status": "GENERATED",
    }


def test_config_resolved_from_parent_official_result(tmp_path):
    (tmp_path / "GEN4-CORE-002.official.json").write_text(
        json.dumps({"candidate": {"config": PARENT_CONFIG}})
    )
    c = resolve_training_metadata(_new_candidate(), {}, tmp_path)
    assert c["config"] == PARENT_CONFIG
    assert c["generation"] == 5
    assert c["candidate"] == "GEN5-CORE-002"


def test_config_falls_back_to_project_state(tmp_path):
    state = {"primary_parent_config": PARENT_CONFIG, "current_generation": 5}
    c = resolve_training_metadata(_new_candidate(), state, tmp_path)
    assert c["config"] == PARENT_CONFIG


def test_config_is_copied_not_shared(tmp_path):
    state = {"primary_parent_config": PARENT_CONFIG}
    c = resolve_training_metadata(_new_candidate(), state, tmp_path)
    c["config"]["d_model"] = 1
    assert PARENT_CONFIG["d_model"] == 384


def test_missing_config_raises_clear_error(tmp_path):
    with pytest.raises(CoreEvolutionError, match="config"):
        resolve_training_metadata(_new_candidate(), {}, tmp_path)


def test_incomplete_config_raises(tmp_path):
    state = {"primary_parent_config": {"d_model": 384}}
    with pytest.raises(CoreEvolutionError, match="config keys"):
        resolve_training_metadata(_new_candidate(), state, tmp_path)


GEN5_STDERR = '''Traceback (most recent call last):
  File "/opt/ai/work/nova-evo/evo/engine/sandbox/GEN5-CORE-002/train_candidate.py", line 200, in <module>
    result = runner.run_robust(**kwargs)
  File "/opt/ai/work/nova-evo/evo/engine/training_runner.py", line 324, in run_robust
    result = self.run_screening(
  File "/opt/ai/work/nova-evo/evo/engine/training_runner.py", line 96, in run_screening
    config = candidate["config"]
KeyError: 'config'
'''

CANDIDATE_BUG_STDERR = '''Traceback (most recent call last):
  File "/opt/ai/work/nova-evo/evo/engine/training_runner.py", line 160, in run_screening
    loss = model(x)
  File "/opt/ai/work/nova-evo/evo/engine/sandbox/GEN5-CORE-002/nova/blocks_scan.py", line 40, in forward
    y = x @ self.w
RuntimeError: mat1 and mat2 shapes cannot be multiplied
'''


def test_gen5_failure_is_infrastructure():
    assert is_infrastructure_failure(GEN5_STDERR)


def test_error_in_generated_core_is_candidate_fault():
    assert not is_infrastructure_failure(CANDIDATE_BUG_STDERR)


def test_empty_stderr_is_not_infrastructure():
    assert not is_infrastructure_failure("")


DICT_CONFIG_TRACE = '''Traceback (most recent call last):
  File "/opt/ai/work/nova-evo/evo/engine/training_runner.py", line 161, in run_screening
    model = build_model(config)
  File "/opt/ai/work/nova-evo/evo/engine/sandbox/GEN5-CORE-002/train_candidate.py", line 120, in build_candidate
    return CandidateModel(config)
  File "/opt/ai/work/nova-evo/evo/engine/sandbox/GEN5-CORE-002/train_candidate.py", line 60, in __init__
    config.vocab_size,
AttributeError: 'dict' object has no attribute 'vocab_size'
'''


def test_harness_config_error_is_infrastructure():
    assert is_infrastructure_failure(DICT_CONFIG_TRACE)


def test_harness_renders_valid_python(tmp_path, monkeypatch):
    """The generated train_candidate.py must be valid Python."""
    import ast
    from evo.engine import core_evolution_engine as ce

    written = {}
    monkeypatch.setattr(ce, "write_file", lambda cid, name, text: written.setdefault(name, text))
    src = tmp_path / "blocks_scan.py"
    src.write_text("# core\n")
    (tmp_path / "GEN4-CORE-002.official.json").write_text(
        json.dumps({"candidate": {"config": PARENT_CONFIG}})
    )
    monkeypatch.setattr(ce, "RESULT_DIR", tmp_path)

    engine = ce.CoreEvolutionEngine.__new__(ce.CoreEvolutionEngine)
    engine.load_project_state = lambda: {}
    cand = dict(_new_candidate(), local_source=str(src))
    engine.write_training_harness(cand, 100, 1000)

    harness = written["train_candidate.py"]
    ast.parse(harness)
    assert "isinstance(config, dict)" in harness
    assert json.loads(written["candidate.json"])["config"] == PARENT_CONFIG
