from pathlib import Path
import pytest
from p2_listen.contracts import atomic_json
from p2_j_sft_event_queue import predecessor_finished,verify_initial


def prepare(tmp_path):
    atomic_json(tmp_path/'run.json',dict(experiment=dict(training=dict(steps=8927))))
    atomic_json(tmp_path/'status.json',dict(complete=True,step=8927))
    atomic_json(tmp_path/'controller_state.json',dict(state='complete',step=8927))
    atomic_json(tmp_path/'logs/completed.json',dict(returncodes=[0,0,0,0]))


def test_requires_all_completion_signals(tmp_path):
    assert not predecessor_finished(tmp_path)
    prepare(tmp_path);assert predecessor_finished(tmp_path)
    for file,row in [
        ('status.json',dict(complete=False,step=8927)),
        ('status.json',dict(complete=True,step=8926)),
        ('controller_state.json',dict(state='training',step=8927)),
        ('logs/completed.json',dict(returncodes=[0,0,1,0])),
    ]:
        prepare(tmp_path);atomic_json(tmp_path/file,row);assert not predecessor_finished(tmp_path)


def rows():
    return [dict(step=i,av=1.,av_video=.6,av_audio=.4,event=.2,weighted_total=1.01,
                 feature_video=0.,feature_audio=0.,seconds=4.,peak_reserved_gib=80.,
                 grad_norm=dict(connector=.1,events=.2,text_lora=.3)) for i in range(1,4)]


def test_real_objective_and_gradient_gate():
    assert verify_initial(rows())['objective_verified']
    for field,value in [('feature_video',1.),('weighted_total',2.),('av',float('nan'))]:
        r=rows();r[0][field]=value
        with pytest.raises(ValueError):verify_initial(r)
    r=rows()
    for x in r:x['grad_norm']['events']=0.
    with pytest.raises(ValueError):verify_initial(r)
