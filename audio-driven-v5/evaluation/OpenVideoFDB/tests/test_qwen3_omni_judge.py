"""The local Qwen3-Omni judge refuses to start until its checkpoint directory is set."""

from __future__ import annotations

import pytest

from ovfdb.score.judge import JudgeUnavailable
from ovfdb.score.qwen3_omni import ask


def test_missing_checkpoint_is_reported_before_any_worker(monkeypatch):
    monkeypatch.delenv("QWEN3_OMNI_PATH", raising=False)
    with pytest.raises(JudgeUnavailable, match="QWEN3_OMNI_PATH"):
        ask("system", "user")
