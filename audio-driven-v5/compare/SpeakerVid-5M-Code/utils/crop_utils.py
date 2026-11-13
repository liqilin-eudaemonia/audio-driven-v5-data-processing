import os
import torch
import numpy as np
from PIL import Image

def compute_union(boxes_list):
    
    ##### step1: compute union for each pid
    union_boxes = [] ### for each person
    boxes_list_raw = []
    start_end_list = []
    size_list = []

    # each item in boxes_list is a list of all frames' box for one person 
    for boxes in boxes_list:
        start = 0
        end = 100000
        total_len = len(boxes)
        for i in range(total_len):
            if len(boxes[i]) > 0:
                start = i
                break
        for i in range(total_len-1, -1, -1):
            if len(boxes[i]) > 0:
                end = i + 1
                break

        start_end_list.append([start, end])

        temp_boxes = []
        for i in range(start, end):
            if len(boxes[i]) > 0:
                temp_boxes.append(boxes[i])

        temp_boxes = np.stack(temp_boxes)
        xmin = np.min(temp_boxes[:, 0])
        ymin = np.min(temp_boxes[:, 1])
        xmax = np.max(temp_boxes[:, 2])
        ymax = np.max(temp_boxes[:, 3])

        all_len_size = np.sqrt((temp_boxes[:, 2] - temp_boxes[:, 0]) * (temp_boxes[:, 3] - temp_boxes[:, 1]))

        size_list.append([np.mean(all_len_size), all_len_size[0], all_len_size[-1]]) # avg_size, min frame size, max frame size

        union_boxes.append([xmin, ymin, xmax, ymax])

        boxes_raw = []
        for box in boxes:
            if len(box) == 0:
                boxes_raw.append(box)
            else:
                boxes_raw.append([box[0]-xmin, box[1]-ymin, box[2]-xmin, box[3]-ymin])
        boxes_list_raw.append(boxes_raw)

    return union_boxes, boxes_list_raw, start_end_list, size_list

def near_equal(a, b, thres=0.9):
    if a > b:
        if b / a > thres:
            return True
        else:
            return False
    else:
        if a / b > thres:
            return True
        else:
            return False

def extend_size(boxes_list, boxes_list_raw, start_end_list, size_list, height, width, extend_rate=1.2, equal_thres=0.9, drop_thres=0.5):

    res_boxes_list = []
    res_startend_list = []
    res_boxes_list_raw = []

    for box, boxes_raw, startend, size in zip(boxes_list, boxes_list_raw, start_end_list, size_list):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
        box_size = np.sqrt((xmax - xmin) * (ymax - ymin))
        avg_size, size_minf, size_maxf = size[0], size[1], size[2]

        if near_equal(box_size, avg_size, equal_thres):
            center = [(xmin+xmax)/2, (ymin+ymax)/2]
            h = (ymax-ymin) * extend_rate; w = (xmax-xmin) * extend_rate

            xmin_ext = center[0] - w/2
            xmax_ext = center[0] + w/2
            ymin_ext = center[1] - h/2
            ymax_ext = center[1] + h/2

            xmin = max(0, xmin_ext)
            ymin = max(0, ymin_ext)
            xmax = min(width-1, xmax_ext)
            ymax = min(height-1, ymax_ext)

            shift_x = box[0] - xmin
            shift_y = box[1] - ymin

            res_boxes_list.append([xmin, ymin, xmax, ymax])
            res_startend_list.append(startend)

            res_boxes_raw = []
            for box_raw in boxes_raw:
                if len(box_raw) == 0:
                    res_boxes_raw.append(box_raw)
                else:
                    res_boxes_raw.append([box_raw[0]+shift_x, box_raw[1]+shift_y, box_raw[0]+shift_x, box_raw[1]+shift_y])

            res_boxes_list_raw.append(res_boxes_raw)

        else:
            if size_minf / box_size < drop_thres or size_maxf / box_size < drop_thres:
                continue
            else:
                res_boxes_list.append(box)
                res_startend_list.append(startend)
                res_boxes_list_raw.append(boxes_raw)

    return res_boxes_list, boxes_list_raw, res_startend_list



def compute_intersect(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    return not (xmax2 < xmin1 or xmax1 < xmin2 or ymax2 < ymin1 or ymax1 < ymin2)

def merge_boxes(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    return (
        min(xmin1, xmin2),
        min(ymin1, ymin2),
        max(xmax1, xmax2),
        max(ymax1, ymax2)
    )

def merge_all_boxes(boxes, start_end_list):
    merged = True
    while merged:
        merged = False
        new_boxes = []
        new_startend = []
        while boxes:
            current_box = boxes.pop(0)
            current_start_end = start_end_list.pop(0)
            found_intersection = False
            for i, data_item in enumerate(zip(boxes, start_end_list)):
                box, start_end = data_item[0], data_item[1]
                if compute_intersect(current_box, box):
                    current_box = merge_boxes(current_box, box)
                    current_start_end = [min(current_start_end[0], start_end[0]), max(current_start_end[1], start_end[1])]
                    boxes.pop(i)
                    start_end_list.pop(i)
                    merged = True
                    found_intersection = True
                    break
            if not found_intersection:
                new_boxes.append(current_box)
                new_startend.append(current_start_end)
        if merged:
            new_boxes.append(current_box)
            new_startend.append(current_start_end)
        boxes = new_boxes
        start_end_list = new_startend
    return boxes, start_end_list

def compute_crop_bounding_box(union_boxes, start_end_list):
    res_list = []
    res_startend_list = []

    ### each box itself
    res_list.extend(union_boxes)
    res_startend_list.extend(start_end_list)

    ### merge box
    res_boxes, res_startend = merge_all_boxes(union_boxes, start_end_list)
    # res_list.extend(res_boxes)
    # res_startend_list.extend(res_startend)

    for box, startend in zip(res_boxes, res_startend):
        if not box in res_list:
            res_list.append(box)
            res_startend_list.append(startend)

    return res_list, res_startend_list

def video_crop(images, box, startend, denomitor=16):
    ##### images: shape of [f, h, w, c]

    start = startend[0]; end = startend[1]
    xmin = int(box[0]); ymin = int(box[1]); xmax = int(box[2]); ymax = int(box[3])

    height = ymax - ymin + 1
    width = xmax - xmin + 1

    height_new = int(round(height / denomitor + 0.5)) * denomitor
    width_new = int(round(width / denomitor + 0.5)) * denomitor

    res_frames = images[start:end, ymin:ymax, xmin:xmax]
    res_frames_format = []
    for frame in res_frames:
        frame = np.array(Image.fromarray(frame).resize((width_new, height_new), Image.Resampling.LANCZOS)).astype(np.uint8)
        res_frames_format.append(frame)

    return res_frames_format, [height_new, width_new, 3]

def video_crop_all_frame(images, box, denomitor=16):
    ##### images: shape of [f, h, w, c]

    xmin = int(box[0]); ymin = int(box[1]); xmax = int(box[2]); ymax = int(box[3])

    height = ymax - ymin + 1
    width = xmax - xmin + 1

    height_new = int(round(height / denomitor + 0.5)) * denomitor
    width_new = int(round(width / denomitor + 0.5)) * denomitor

    res_frames = images[:, ymin:ymax, xmin:xmax]
    res_frames_format = []
    for frame in res_frames:
        frame = np.array(Image.fromarray(frame).resize((width_new, height_new), Image.Resampling.LANCZOS)).astype(np.uint8)
        res_frames_format.append(frame)

    return res_frames, res_frames_format, [height_new, width_new, 3]