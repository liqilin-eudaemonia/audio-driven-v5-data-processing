import csv
import numpy as np
import pytest
from p2_glare_unfiltered.rules import write_fer, policy
from p2_glare_rgb.detector import policy as strict_policy
from p2_glare_eval.detector import EMOTIONS
from p2_glare_eval.upstream import reaction_module


def face(value):
    return dict(box=[0, 0, 10, 10], emotions={k: value for k in EMOTIONS})


def test_missing_and_multiple_faces_keep_full_timeline(tmp_path):
    raw = [dict(frame_index=0, faces=[]),
           dict(frame_index=1, faces=[face(.7), face(.1)])]
    path = tmp_path / 'fer.csv'
    stats = write_fer(raw, path, 2)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2 and rows[0]['happy'] == ''
    assert float(rows[1]['happy']) == .7
    assert stats['fer_no_face_frames'] == stats['fer_multiple_face_frames'] == 1
    # Exercise the actual pinned reaction reader: no invented filling rule.
    assert reaction_module('smiling').load_fer_happy_scores(path) == {0: 0., 1: .7}


def test_reference_pose_gap_rule_is_unchanged():
    rule = reaction_module('nodding')
    values = np.array([1., 0., 3.], dtype=np.float32)
    detected = np.array([1., 0., 1.], dtype=np.float32)
    np.testing.assert_array_equal(rule.fill_pose_gaps(values, detected), [1., 2., 3.])
    np.testing.assert_array_equal(rule.fill_pose_gaps(values, np.zeros(3)), [0., 0., 0.])


def test_only_quality_admission_changes():
    strict, new = strict_policy(), policy()
    for key in ['reaction_parameters', 'fps', 'fer_input', 'nodding_variant', 'combination', 'score_threshold']:
        assert strict[key] == new[key]
    assert new['required_frame_coverage'] is None and new['min_face_iou'] is None
    assert new['reject_multiple_faces'] is False


@pytest.mark.parametrize('raw,count', [
    ([dict(frame_index=1, faces=[])], 1),
    ([dict(frame_index=0, faces=[])], 2),
    ([dict(frame_index=0, faces=[face(float('nan'))])], 1),
])
def test_corruption_is_still_an_error(tmp_path, raw, count):
    with pytest.raises(ValueError):
        write_fer(raw, tmp_path / 'bad.csv', count)
