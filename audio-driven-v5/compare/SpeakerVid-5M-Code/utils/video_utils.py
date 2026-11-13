import os
import io
import av
import json
import subprocess
import numpy as np

def ffmpeg_to_ndarray(input_file, height, width, start, duration):
    # "ffmpeg -v quiet -y -i {} -ss {} -t {} -q:v 2 -r 25 -filter:v fps=25 {}"
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start),
        '-t', str(duration),
        '-q:v', '2',
        '-r', '25',
        '-filter:v', 'fps=25',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', 
        # '-threads', '0',
        '-'
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

    # "ffmpeg -v quiet -y -i {} -q:a 0 -ac 1 -ar 16000 -acodec pcm_s16le -async 1 -vn {}/audio.wav"
    audio_command = [
        'ffmpeg',
        '-i', input_file,
        '-q:a', '0',
        '-ac', '1',
        '-ar', '16000',
        '-ss', str(start),
        '-t', str(duration),
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

    return np.array(frames), audio_binary

def ffmpeg_to_ndarray_only(input_file, height, width, start, duration):
    # "ffmpeg -v quiet -y -i {} -ss {} -t {} -q:v 2 -r 25 -filter:v fps=25 {}"
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ss', str(start),
        '-t', str(duration),
        '-q:v', '2',
        '-r', '25',
        '-filter:v', 'fps=25',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', 
        # '-threads', '0',
        '-'
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

def ffmpeg_to_audio_binary(input_file, start, duration):
    
    # "ffmpeg -v quiet -y -i {} -q:a 0 -ac 1 -ar 16000 -acodec pcm_s16le -async 1 -vn {}/audio.wav"
    audio_command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
        '-i', input_file,
        '-q:a', '0',
        '-ac', '1',
        '-ar', '16000',
        '-ss', str(start),
        '-t', str(duration),
        '-f', 'wav',
        '-acodec', 'pcm_s16le',
        '-async', '1',
        '-vn',
        '-'
    ]
    audio_process = subprocess.Popen(audio_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_binary = audio_process.stdout.read()
    audio_binary_io = io.BytesIO(audio_binary)
    audio_process.stdout.close()
    audio_process.wait()

    return audio_binary_io, audio_binary

def ffmpeg_to_full_audio_binary(input_file):
    
    # "ffmpeg -v quiet -y -i {} -q:a 0 -ac 1 -ar 16000 -acodec pcm_s16le -async 1 -vn {}/audio.wav"
    audio_command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
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
    audio_binary_io = io.BytesIO(audio_binary)
    audio_process.stdout.close()
    audio_process.wait()

    return audio_binary_io, audio_binary

def ffmpeg_save_audio(input_file,audio_path):
    
    # "ffmpeg -v quiet -y -i {} -q:a 0 -ac 1 -ar 16000 -acodec pcm_s16le -async 1 -vn {}/audio.wav"
    audio_command = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
        '-i', input_file,
        '-q:a', '0',
        '-ac', '1',
        '-ar', '16000',
        '-f', 'wav',
        '-acodec', 'pcm_s16le',
        '-async', '1',
        '-vn',
        audio_path
    ]
    subprocess.run(audio_command, check=True)

def save_ndarray_to_video(frames, output_file, fps=25):

    height, width, _ = frames[0].shape
    
    command = [
        'ffmpeg',
        '-y',  # 覆盖已存在的文件
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-an',  # 无音频
        '-vcodec', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_file
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()

def ffmpeg_probe(video_b):
    command = [
        'ffprobe',
        '-i', '-',
        '-hide_banner',
        '-show_format',
        '-show_streams',
        '-print_format', 'json'
    ]

    # 执行 FFmpeg 命令
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False
    )
    stdout, stderr = process.communicate(input=video_b)
    metadata = json.loads(stdout.strip().decode('utf-8'))
    video_info = next(s for s in metadata['streams'] if s['codec_type'] == "video")
    raw_fps = eval(video_info['avg_frame_rate'])
    height = int(video_info['height'])
    width = int(video_info['width'])
    bitrate = int(video_info['bit_rate'])
    duration = float(video_info['duration'])

    print(raw_fps, height, width, bitrate, duration, bitrate / np.sqrt(height*width))
    # print(metadata)

    return raw_fps, height, width, bitrate, duration, bitrate / np.sqrt(height*width)

def ffmpeg_binary_to_ndarray(video_b, height, width, start, duration):
    command = [
        'ffmpeg',
        # '-hwaccel', 'cuvid',
        '-i', 'pipe:',
        '-ss', str(start),
        '-t', str(duration),
        '-q:v', '2',
        '-r', '25',
        '-filter:v', 'fps=25',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo', 
        '-threads', '0',
        '-'
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = height * width * 3

    # chunk_size = 10 * 1024  #
    # print(len(video_b))
    # for i in range(0, len(video_b), chunk_size):
    #     chunk = video_b[i:i + chunk_size]
    #     process.stdin.write(chunk)
    process.stdin.write(video_b)
    process.stdin.close()
    # stdout, stderr = process.communicate(input=video_b)

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

def ffmpeg_binary_to_audio_binary(video_b, start, duration):
    audio_command = [
        'ffmpeg',
        '-i', 'pipe:',
        '-q:a', '0',
        '-ac', '1',
        '-ar', '16000',
        '-ss', str(start),
        '-t', str(duration),
        '-f', 'wav',
        '-acodec', 'pcm_s16le',
        '-async', '1',
        '-vn',
        '-'
    ]
    audio_process = subprocess.Popen(audio_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_process.stdin.write(video_b)
    audio_process.stdin.close()

    audio_binary = audio_process.stdout.read()
    audio_binary = io.BytesIO(audio_binary)
    audio_process.stdout.close()
    audio_process.wait()

    return audio_binary

def pyav_ndarray_to_binary(video_ndarray, res_size):
    output_buffer = io.BytesIO()
    out_container = av.open(output_buffer, format='mp4', mode='w')
    video_stream = out_container.add_stream('h264', rate=25)
    video_stream.width = res_size[1]
    video_stream.height = res_size[0]
    video_stream.pix_fmt = 'yuv420p'
    for frame in video_ndarray:
        av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        for packet in video_stream.encode(av_frame):
            out_container.mux(packet)

    for packet in video_stream.encode():
        out_container.mux(packet)
    out_container.close()

    output_buffer.seek(0)
    b = output_buffer.read()

    return b

