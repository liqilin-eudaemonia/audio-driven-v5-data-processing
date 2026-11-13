"""Review-only audio composition must retain source timing and relative levels."""
import importlib.util
from pathlib import Path
import json
import subprocess
import numpy as np
import soundfile as sf
import pytest
spec=importlib.util.spec_from_file_location('review_builder',Path(__file__).resolve().parents[1]/'scripts/build_p2_av_review.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def test_mixed_audio_keeps_time_and_never_boosts_quiet_listener(tmp_path):
 listener=np.zeros((4800,2),np.float32);speaker=np.zeros(2400,np.float32);listener[960]=.00003;speaker[1200]=.5
 a=tmp_path/'listener.wav';b=tmp_path/'speaker.wav';sf.write(a,listener,48000,subtype='FLOAT');sf.write(b,speaker,24000,subtype='FLOAT')
 before=[review.sha(a),review.sha(b)];mixed,m=review.aligned_mix(a,b,.1)
 assert mixed.shape==(4800,2);assert m['listener_gain']==m['speaker_gain']==1
 assert np.argmax(mixed[:,0])==2400
 assert mixed[960,0]==pytest.approx(.00003,abs=1e-8)
 assert np.array_equal(mixed[:,0],mixed[:,1]);assert before==[review.sha(a),review.sha(b)]


def test_peak_safety_applies_identical_gain_to_both_sources(tmp_path):
 a=tmp_path/'a.wav';b=tmp_path/'b.wav';sf.write(a,np.ones(480,dtype=np.float32)*.8,48000,subtype='FLOAT');sf.write(b,np.ones(480,dtype=np.float32)*.6,48000,subtype='FLOAT')
 mixed,m=review.aligned_mix(a,b,.01)
 assert np.max(abs(mixed))==pytest.approx(.95,abs=1e-6)
 assert m['listener_gain']==m['speaker_gain'];assert m['listener_gain']<1


def test_gt_remux_has_audio_and_preserves_video_frames(tmp_path,monkeypatch):
 media=tmp_path/'media';media.mkdir();monkeypatch.setattr(review,'MEDIA',media)
 video=tmp_path/'original.mp4';audio=tmp_path/'target.wav'
 subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=32x32:r=25:d=0.16','-c:v','libx264','-threads','1',str(video)],check=True)
 t=np.arange(7680)/48000;sf.write(audio,np.sin(2*np.pi*440*t).astype(np.float32)*.1,48000,subtype='FLOAT');before=[review.sha(video),review.sha(audio)]
 path=media/Path(review.mux(video,audio,.16)).name
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_type','-of','json',str(path)],text=True))
 assert {x['codec_type'] for x in probe['streams']}=={'audio','video'}
 def frames(p):return subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'])
 assert frames(video)==frames(path);assert before==[review.sha(video),review.sha(audio)]
