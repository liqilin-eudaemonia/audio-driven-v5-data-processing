from concurrent.futures import Future
import time
from listen_benchmark.caption_guard import due_prepares,shared_identity
from run_glare_reaction_omni_pilot import cue_gaps

def test_ready_future_never_times_out():
    f=Future();f.set_result(1)
    assert due_prepares({f:{'submitted_at':time.monotonic()-1000}},time.monotonic(),180)==[]

def test_silent_language_scope():
    assert not cue_gaps([],[],[],"She doesn't speak.",silent=True).get('silent_vocal')
    for text in ['She hums softly.','She softly chuckles.','She is not smiling and says "yeah".']:
        assert cue_gaps([],[],[],text,silent=True).get('silent_vocal')

def test_identity_is_frozen_per_process():
    assert shared_identity() is shared_identity()
