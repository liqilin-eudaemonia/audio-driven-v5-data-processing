import os
import torch
import torchaudio
import torch.nn.functional as F
from audio.UniSpeech.WavLM.WavLM import WavLM, WavLMConfig
from tqdm import tqdm
def load_wav(wav_path, target_sr=16000):
    speech, sample_rate = torchaudio.load(wav_path, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, f'Wav sample rate {sample_rate} must be greater than {target_sr}'
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def extract_feature(model, wav_tensor):
    with torch.no_grad():
        rep = model.extract_features(wav_tensor)[0][0]  # shape: (T, 1024)
    return rep  # return frame-level features

def cosine_similarity_global(rep1, rep2):
    mean1 = rep1.mean(dim=0, keepdim=True)  # (1, 1024)
    mean2 = rep2.mean(dim=0, keepdim=True)  # (1, 1024)
    return F.cosine_similarity(mean1, mean2).item()

def cosine_similarity_framewise(rep1, rep2):
    T = min(rep1.size(0), rep2.size(0))
    rep1, rep2 = rep1[:T], rep2[:T]  # align to same length
    sim = F.cosine_similarity(rep1, rep2, dim=1)  # (T,)
    return sim.mean().item()

def evaluate_folder(wav_folder, model):
    results = []
    files = os.listdir(wav_folder)
    candidates = [f for f in files if f.endswith('.wav') and not f.endswith('_gt.wav')]

    for fname in tqdm(candidates):
        base = fname[:-4]
        gt_name = f"{base}_gt.wav"
        path_pred = os.path.join(wav_folder, fname)
        path_gt = os.path.join(wav_folder, gt_name)

        if not os.path.exists(path_gt):
            print(f"[!] GT file missing for {fname}, skipped.")
            continue

        try:
            wav_pred = load_wav(path_pred)
            wav_gt = load_wav(path_gt)
            rep_pred = extract_feature(model, wav_pred)
            rep_gt = extract_feature(model, wav_gt)

            sim_o = cosine_similarity_global(rep_pred, rep_gt)
            sim_f = cosine_similarity_framewise(rep_pred, rep_gt)

            results.append((fname, sim_o, sim_f))
        except Exception as e:
            print(f"[!] Error processing {fname}: {e}")

    return results

if __name__ == "__main__":
    wav_folder = ""
    checkpoint_path = "SpeakerVid-5M/weights/WavLM-Large.pt"
    checkpoint = torch.load(checkpoint_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    sim_results = evaluate_folder(wav_folder, model)

    print("\nEvaluation Results:")
    for fname, sim_o, sim_f in sim_results:
        print(f"{fname:<60} SIM-O: {sim_o:.4f}    SIM-F: {sim_f:.4f}")

    if sim_results:
        avg_o = sum(sim_o for _, sim_o, _ in sim_results) / len(sim_results)
        avg_f = sum(sim_f for _, _, sim_f in sim_results) / len(sim_results)
        print(f"\n[Summary] Average SIM-O: {avg_o:.4f}, Average SIM-F: {avg_f:.4f}")
    else:
        print("\nNo valid audio pairs were evaluated.")