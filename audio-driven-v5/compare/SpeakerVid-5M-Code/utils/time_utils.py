
import os
from scenedetect import FrameTimecode

def time_format_diff(time1, time2):
    time_len = [3600, 60, 1]

    time1_list = [float(i) for i in time1.split(':')]
    time2_list = [float(i) for i in time2.split(':')]

    diff1 = 0
    diff2 = 0
    for i in range(len(time1_list)):
        d_iff1 = float(time1_list[i])
        d_iff2 = float(time2_list[i])
        diff1 += time_len[i] * d_iff1
        diff2 += time_len[i] * d_iff2

    diff = abs(diff2 - diff1)

    dur_str = ""
    for i in range(len(time_len)):
        if i == 0:
            dur_str += "{:02d}:".format(int(diff / time_len[i]))
        elif i == 1:
            dur_str += "{:02d}:".format(int((diff % time_len[0]) / time_len[i]))
        else:
            num1 = int((diff % time_len[0]) % time_len[1])
            num2 = int(((diff % time_len[0]) % time_len[1] - num1) * 1000)
            dur_str += "{:02d}.{:03d}".format(num1, num2)

    return diff, dur_str


def recompute_start(start, secs):
    time_len = [3600, 60, 1]

    start_list = [float(i) for i in start.split(':')]
    sec = 0
    for i in range(len(start_list)):
        seci = float(start_list[i])
        sec += time_len[i] * seci
    extend_start_secs = sec + secs

    dur_str = ""
    for i in range(len(time_len)):
        if i == 0:
            dur_str += "{:02d}:".format(int(extend_start_secs / time_len[i]))
        elif i == 1:
            dur_str += "{:02d}:".format(int((extend_start_secs % time_len[0]) / time_len[i]))
        else:
            num1 = int((extend_start_secs % time_len[0]) % time_len[1])
            num2 = int(((extend_start_secs % time_len[0]) % time_len[1] - num1) * 1000)
            dur_str += "{:02d}.{:03d}".format(num1, num2)

    return dur_str

def recompute_duration(duration, secs):
    time_len = [3600, 60, 1]

    start_list = [float(i) for i in duration.split(':')]
    sec = 0
    for i in range(len(start_list)):
        seci = float(start_list[i])
        sec += time_len[i] * seci
    extend_start_secs = sec - secs

    dur_str = ""
    for i in range(len(time_len)):
        if i == 0:
            dur_str += "{:02d}:".format(int(extend_start_secs / time_len[i]))
        elif i == 1:
            dur_str += "{:02d}:".format(int((extend_start_secs % time_len[0]) / time_len[i]))
        else:
            num1 = int((extend_start_secs % time_len[0]) % time_len[1])
            num2 = int(((extend_start_secs % time_len[0]) % time_len[1] - num1) * 1000)
            dur_str += "{:02d}.{:03d}".format(num1, num2)

    return dur_str

def time_to_float(time):
    time_len = [3600, 60, 1]

    time_list = [float(i) for i in time.split(':')]

    time_count = 0
    for i in range(len(time_list)):
        tt = float(time_list[i])
        time_count += time_len[i] * tt

    return time_count

def float_to_time(time):
    time_len = [3600, 60, 1]

    time_str = ""
    for i in range(len(time_len)):
        if i == 0:
            time_str += "{:02d}:".format(int(time / time_len[i]))
        elif i == 1:
            time_str += "{:02d}:".format(int((time % time_len[0]) / time_len[i]))
        else:
            num1 = int((time % time_len[0]) % time_len[1])
            num2 = int(((time % time_len[0]) % time_len[1] - num1) * 1000)
            time_str += "{:02d}.{:03d}".format(num1, num2)

    return time_str

def scene_list_process(scene_list, dur=10):
    res_scene_list = []

    for scene in scene_list:
        start = scene[0].get_timecode()
        end = scene[1].get_timecode()

        fps = float(scene[0].get_framerate())

        start = time_to_float(start)
        end = time_to_float(end)

        diff = abs(end - start)
        if diff < 3:
            continue
        
        if diff / dur > 1.4:

            for i in range(int(round(diff / dur + 0.5))):
                si = dur * i + start
                ei = min(dur * (i + 1) + start, end)

                si = float_to_time(si)
                ei = float_to_time(ei)

                res_scene_list.append(
                    (
                        FrameTimecode(si, fps=fps),
                        FrameTimecode(ei, fps=fps)
                    )
                )
        else:
            res_scene_list.append(scene)

    return res_scene_list

    