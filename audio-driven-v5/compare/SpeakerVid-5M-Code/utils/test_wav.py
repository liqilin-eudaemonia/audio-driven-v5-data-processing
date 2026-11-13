from video_utils import ffmpeg_to_full_audio_binary,ffmpeg_save_audio
import torchaudio
from scipy.io import wavfile
import numpy as np
audio_path = "/data/pipline/human_centric_vw_model/1o343YfhaYU_S14_18.6_21.64.wav"
audio_binary_io, full_audio_binary = ffmpeg_to_full_audio_binary("/data/pipline/human_centric_vw_model/1o343YfhaYU_S14_18.6_21.64.mp4")
ffmpeg_save_audio("/data/pipline/human_centric_vw_model/1o343YfhaYU_S14_18.6_21.64.mp4","/data/pipline/human_centric_vw_model/1o343YfhaYU_S14_18.6_21.64.wav")
wav_data, fs = torchaudio.load(audio_binary_io)
wav, fs = torchaudio.load("/data/pipline/human_centric_vw_model/1o343YfhaYU_S14_18.6_21.64.wav")
sample_rate, audio = wavfile.read(audio_path)
audio_np = wav_data.numpy().squeeze().T  # 转置
audio_np = np.round(audio_np * 32767).astype(np.int16)


from pydub import AudioSegment
import torchaudio
import torch
def from_torchaudio_to_AudioSegment(wav_data, fs):
    # 转换为pydub对象
    if wav_data.dtype == torch.float32:  # 处理归一化数据
        # wav_data = (wav_data * 32767).clamp(-32768, 32767).short()
        wav_data = torch.round(wav_data * 32767).clamp(-32768, 32767).short()
    audio = AudioSegment(
        wav_data.numpy().tobytes(),
        frame_rate=fs,
        sample_width=2,  # 16-bit
        channels=wav_data.shape[0]
    )
    return audio
def from_AudioSegment_to_torchaudio(audio):
    # 转换为pydub对象
    samples = audio.get_array_of_samples()  # 获取numpy数组
    wav_data = torch.FloatTensor(samples).view(-1, audio.channels).T  # 转换为(channel, samples)
    wav_data = wav_data / 32767
    fs = audio.frame_rate
    return wav_data, fs

audio_path = "/data/pipline/human_centric_vw_model/1o343YfhaYU_S14_18.6_21.64.wav"
wav, fs = torchaudio.load(audio_path)
audio_from_torchaudio = from_torchaudio_to_AudioSegment(wav, fs)
audio = AudioSegment.from_file(audio_path)
wav_from_AudioSegment,fs_from_AudioSegment = from_AudioSegment_to_torchaudio(audio)


breakpoint()