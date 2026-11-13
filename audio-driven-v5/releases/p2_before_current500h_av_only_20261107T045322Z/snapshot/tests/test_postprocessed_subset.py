from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from listen_benchmark.postprocessed_subset import event_statistics


def event(i,category,start,end,source=None):
    return dict(event_id=i,type=category,start=start,end=end,source=dict(event_id=source or i))


def test_support_pieces_and_multiclass_descriptions_do_not_inflate_event_count():
    label=dict(events=[event("a-part1","listener_vocalization",1,1.2,"a"),
        event("a-part2","listener_vocalization",1.5,1.8,"a"),
        event("b","head_nod",2,3,"b"),event("b2","smile",2,3,"b")])
    result=event_statistics(label,5)
    assert result["event_count"]==2
    assert result["listener_vocalization_count"]==1
    assert result["head_nod_count"]==1 and result["smile_count"]==1
    assert result["canonical_interval_count"]==4
    assert result["non_smile_event_count"]==2
    assert abs(result["audio_coverage_s"]-.5)<1e-9


def test_actual_repeated_responses_remain_distinct_and_deleted_audio_scores_zero():
    before=dict(events=[event("a","listener_vocalization",1,1.3),event("b","listener_vocalization",1.3,1.6),event("c","smile",0,4)])
    assert event_statistics(before,5)["listener_vocalization_count"]==2
    after=dict(events=[before["events"][-1]])
    result=event_statistics(after,5)
    assert result["audio_event_count"]==0
    assert result["non_smile_event_count"]==0

