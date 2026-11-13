
import os
import io
import json
import time
import numpy as np
import subprocess
import soundfile as sf
import ffmpeg
from syncnet.sync_score import SyncNetInstance
from syncnet.preprocess import PreProcessor

class SyncNet_detector:
    def __init__(self, sync_model_file=None, s3fd_model_file=None, device=None):
        ### initialize one object of SyncNetInstance class
        self.syncnet_obj = SyncNetInstance(model_path=sync_model_file, device=device, av_offset=0)
        self.preprocessor = PreProcessor(s3fd_model_file, device)

    def inference(self, video_ndarry, audio_binary, sample_rate):
        
        aligned_img_list = self.preprocessor.align(video_ndarry)

        if aligned_img_list is None:
            print('no face detected')
            return None, None, None
        
        avoff, lsec, lsed = self.syncnet_obj.evaluate_single(aligned_img_list, audio_binary, sample_rate)
        
        return avoff, lsec, lsed


def ffmpeg_to_ndarray_only(input_file, height=816, width=496):
    # "ffmpeg -v quiet -y -i {} -ss {} -t {} -q:v 2 -r 25 -filter:v fps=25 {}"
    command = [
        'ffmpeg',
        '-i', input_file,
        '-q:v', '2',
        '-r', '25',
        '-filter:v', 'fps=25',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', '-'
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = height * width * 3

    frames = []
    while True:
        raw_image = process.stdout.read(frame_size)
        if len(raw_image) != frame_size:
            break
        frame = np.frombuffer(raw_image, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        frames.append(frame)

    process.stdout.close()
    process.wait()

    return np.array(frames)

import subprocess
import io

def process_wav_to_audio_binary(input_wav_file):
    audio_command = [
        'ffmpeg',
        '-i', input_wav_file,
        '-ac', '1',
        '-ar', '16000',
        '-f', 'wav',
        '-acodec', 'pcm_s16le',
        '-vn',
        '-'
    ]
    audio_process = subprocess.Popen(audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_binary = audio_process.stdout.read()
    audio_binary = io.BytesIO(audio_binary)
    audio_process.stdout.close()
    audio_process.wait()
    return audio_binary


def ffmpeg_to_audio_binary(input_file):
    # "ffmpeg -v quiet -y -i {} -q:a 0 -ac 1 -ar 16000 -acodec pcm_s16le -async 1 -vn {}/audio.wav"
    audio_command = [
        'ffmpeg',
        '-i', input_file,
        '-q:a', '0',
        '-ac', '1',
        '-ar', '16000',
        '-f', 'wav',
        '-acodec', 'pcm_s16le',
        '-async', '1',
        '-vn',
        '-'
    ]
    audio_process = subprocess.Popen(audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_binary = audio_process.stdout.read()
    audio_binary = io.BytesIO(audio_binary)
    audio_process.stdout.close()
    audio_process.wait()
    return audio_binary


if __name__ == '__main__':
    
    s3fd_model_path = "/weights/sfd_face.pth"
    syncnet_model_path = "/weights/syncnet_v2.model"
    use_audio_in_video = False
    use_gt_audio = True
    # video_paths = "/data/pipline/human_centric_vw_model/benchmark/out_8_0_32_VA_eval_5s_pretrain_step_84000"
    video_paths = ''

    wav_paths = video_paths
    conf_list = []
    for video_path in os.listdir(video_paths):
        if not video_path.endswith(".mp4"):
            continue
        video_path = os.path.join(video_paths, video_path)
        probe_res = ffmpeg.probe(video_path)
        video_info = next(s for s in probe_res['streams'] if s['codec_type'] == "video")
        height = int(video_info['height'])
        width = int(video_info['width'])
        print(video_path, height, width)
        frame_list = ffmpeg_to_ndarray_only(video_path, height=height, width=width)
        if use_audio_in_video:
            audio_binary = ffmpeg_to_audio_binary(video_path)
        else:
            if use_gt_audio:
                audio_path = video_path.replace(".mp4", "_gt.wav")
            else:
                audio_path = video_path.replace(".mp4", ".wav")
            audio_binary = process_wav_to_audio_binary(audio_path)
        data, sr = sf.read(audio_binary)
        syncnet_detector = SyncNet_detector(syncnet_model_path, s3fd_model_path, 'cuda')
        time1 = time.time()
        a,conf,c = syncnet_detector.inference(frame_list, data, sr)
        print(a,conf,c)
        conf_list.append(conf)
    print(conf_list)
    print(sum(conf_list)/len(conf_list))
    print(time.time() - time1)