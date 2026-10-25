from pathlib import Path
import hashlib,json
import soundfile as sf
R=Path(__file__).resolve().parents[1]
def verify():
 evidence=json.loads((R/'archived_source_evidence.json').read_text());wave,sr=sf.read(R/'source_audio.wav',dtype='float32');digest=hashlib.sha256(wave.tobytes()).hexdigest();assert sr==evidence['audio_sample_rate'] and len(wave)==evidence['audio_samples'];assert digest==evidence['audio_float32_mono_sha256'],'Audio differs from approved source; do not apply its frozen words/annotations to another source.'
 (R/'replay_audio_validation.json').write_text(json.dumps({'waveform_samples_identical_to_archived_source':True,'float32_sha256':digest,'samples':len(wave),'sample_rate':sr},indent=2));print('AUDIO_PAYLOAD_MATCH')
if __name__=='__main__':verify()
