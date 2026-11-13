"""Native DECA for L2L and EMOCA-v2 for DIM, kept in separate spaces."""
from pathlib import Path
import sys

import numpy as np
import torch

from .data import ROOT, read, save, sha
from .features import backend, ids_for, receipt


def export(method, work, ids, device):
    backend(device)
    sys.path.insert(0, str(ROOT / 'compare/deployment'))
    from skimage.transform import estimate_transform, warp
    from PIL import Image
    if method == 'l2l':
        vendor = ROOT / 'compare/deployment/vendor/DECA'
        sys.path.insert(0, str(vendor))
        from decalib.models import resnet
        resnet.load_ResNet50Model = lambda: resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3])
        from decalib.models.encoders import ResnetEncoder
        model = ResnetEncoder(236)
        checkpoint = ROOT / 'compare/deployment/assets/deca_model.tar'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['E_flame'], strict=True)
        model.to(device).eval().requires_grad_(False)
        info = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), space='DECA exp50+pose6')
    else:
        from vico_literature_emoca import Encoder, Detector
        model, detector = Encoder(), Detector()
        info = dict(model.info, space='EMOCA-v2 pose6+exp50')
    for sid in ids_for(work, ids):
        dest = Path(work) / 'features' / method / (sid + '.pt')
        if dest.with_suffix('.json').exists():
            from .engine import check_features
            check_features(work, method, sid)
            continue
        media = Path(work) / 'media' / sid
        values = {}
        if method == 'l2l':
            import librosa
            wave, sr = librosa.load(media / 'speaker.wav', sr=16000)
            mel = librosa.feature.melspectrogram(y=wave, sr=sr, hop_length=160, win_length=400, n_fft=512, n_mels=128)
            frames = read(media / 'receipt.json')['frames30']
            values['audio'] = torch.from_numpy(np.array(Image.fromarray(mel).resize(
                (frames * 4, 128), Image.Resampling.LANCZOS)).T.copy()).float()
            bfm = torch.load(Path(work) / 'features/bfm' / (sid + '.pt'), map_location='cpu', weights_only=True)
        for role in ['speaker', 'listener']:
            frames = np.load(media / (role + '30.npy'))
            crops = []
            if method == 'l2l':
                assert bfm[role + '_detected'].all(), 'Refusing unsupported DECA training frames'
                for frame, lm in zip(frames, bfm[role + '_landmarks'].numpy()):
                    left, top = lm.min(0)
                    right, bottom = lm.max(0)
                    size = int((right - left + bottom - top) / 2 * 1.1 * 1.25)
                    center = np.array([(right + left) / 2, (bottom + top) / 2])
                    src = np.array([[center[0]-size/2, center[1]-size/2], [center[0]-size/2, center[1]+size/2],
                                    [center[0]+size/2, center[1]-size/2]])
                    t = estimate_transform('similarity', src, np.array([[0, 0], [0, 223], [223, 0]]))
                    crops.append(warp(frame / 255., t.inverse, output_shape=(224, 224)).transpose(2, 0, 1).astype(np.float32))
            else:
                for frame in frames:
                    crop, count = detector.crop(frame)
                    assert crop is not None and count == 1, 'Unsupported EMOCA training frame'
                    crops.append(crop)
            output = []
            with torch.inference_mode():
                for start in range(0, len(crops), 32):
                    x = torch.from_numpy(np.stack(crops[start:start+32])).to(device)
                    if method == 'l2l':
                        full = model(x)
                        motion = torch.cat([full[:, 150:200], full[:, 200:206]], dim=1)
                    else:
                        motion = model.encode(x)
                    output.append(motion.cpu())
            values[role] = torch.cat(output)
            assert values[role].shape == (len(frames), 56) and torch.isfinite(values[role]).all()
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(values, dest)
        receipt(dest, sid, work, method, dict(extractor=info, fps=30, synthetic_features=False))
        print(method, sid, len(frames), flush=True)
    if method == 'dim':
        export_hubert(work, ids, device)


def export_hubert(work, ids, device):
    import soundfile as sf
    from transformers import HubertModel
    checkpoint = ROOT / 'compare/deployment/assets/hubert-base-ls960'
    model = HubertModel.from_pretrained(checkpoint, local_files_only=True).to(device).eval().requires_grad_(False)
    for sid in ids_for(work, ids):
        path = Path(work) / 'features/dim' / (sid + '.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        if 'audio' in value:
            continue
        wave, sr = sf.read(Path(work) / 'media' / sid / 'speaker.wav', dtype='float32')
        assert sr == 16000
        # S3PRL's hubert-base uses the raw waveform and final hidden layer.
        # Match the native DIM linear resampling (align_corners=True).
        with torch.inference_mode():
            hidden = model(torch.from_numpy(wave)[None].to(device)).last_hidden_state
            hidden = torch.nn.functional.interpolate(hidden.transpose(1, 2), size=len(value['speaker']),
                                                      mode='linear', align_corners=True).transpose(1, 2)
        value['audio'] = hidden[0].cpu()
        assert value['audio'].shape == (len(value['speaker']), 768)
        torch.save(value, path)
        rec = read(path.with_suffix('.json'))
        rec['feature_sha256'] = sha(path)
        rec['audio'] = dict(model=str(checkpoint), sha256=sha(checkpoint / 'pytorch_model.bin'),
                            layer='last', waveform='raw 16k; no W2V replacement', interpolation='linear align_corners=True')
        save(path.with_suffix('.json'), rec)
        print('hubert', sid, flush=True)
