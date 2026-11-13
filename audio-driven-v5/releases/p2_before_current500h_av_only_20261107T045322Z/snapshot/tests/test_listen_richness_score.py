"""Scoring is a sidecar: evidence stays verbatim and failures cannot become score zero."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from listen_richness_score import compile_response, output_schema


def test_zero_is_a_valid_observed_rating_with_original_evidence():
    caption="She remains still. She does not speak."
    value=compile_response(dict(score=0,reason="保持静止，没有明确回应变化",evidence_ids=["s001","s002"],uncertainty=""),caption)
    assert value["score"] == 0
    assert value["evidence"] == [dict(id="s001",text="She remains still."),dict(id="s002",text="She does not speak.")]


@pytest.mark.parametrize("bad",[True, -1, 6, None, "3"])
def test_invalid_score_is_rejected_not_coerced_to_zero(bad):
    with pytest.raises(ValueError):
        compile_response(dict(score=bad,reason="说明",evidence_ids=["s001"],uncertainty=""),"She smiles.")


def test_missing_evidence_and_unknown_ids_fail():
    for refs in [[],["s002"],["s001","s001"]]:
        with pytest.raises(ValueError):
            compile_response(dict(score=1,reason="微笑",evidence_ids=refs,uncertainty=""),"She smiles.")
    assert output_schema("She smiles.")["properties"]["evidence_ids"]["items"]["enum"] == ["s001"]
