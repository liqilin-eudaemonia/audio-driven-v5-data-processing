import os
import torch
import numpy as np
from torchvision.ops import box_convert
import time

def same_box(box1, box2, size_diff_thres=0.3, center_pos_diff_thres=0.2):

    center_pos1 = [(box1[2] + box1[0]) / 2, (box1[1] + box1[3]) / 2]
    center_pos2 = [(box2[2] + box2[0]) / 2, (box2[1] + box2[3]) / 2]
    box_size1 = np.sqrt((box1[2] - box1[0]) * (box1[3] - box1[1]))
    box_size2 = np.sqrt((box2[2] - box2[0]) * (box2[3] - box2[1]))

    center_diff = np.sqrt((center_pos1[0] - center_pos2[0]) ** 2 + (center_pos1[1] - center_pos2[1]) ** 2)
    flag = (abs((box_size1 - box_size2) / box_size1) > size_diff_thres or abs(center_diff / box_size1) > center_pos_diff_thres)

    return (not flag)

def win_complete(box_list, win_size=7):
    res_list = []

    for i in range(0, win_size//2):
        res_list.append(box_list[i])
    for i in range(win_size//2, len(box_list)-win_size//2):
        if len(box_list[i]) == 0:
            box_temp = []
            flag = False
            flag1 = False
            flag2 = False
            for j in range(-win_size//2, win_size//2+1):
                if j < 0:
                    if len(box_list[i+j]) > 0 or len(res_list[i+j]) > 0:
                        flag1 = True
                        box_temp.append(np.array(box_list[i+j]) if len(box_list[i+j]) > 0 else np.array(res_list[i+j]))
                else:
                    if len(box_list[i+j]) > 0:
                        flag2 = True
                        box_temp.append(np.array(box_list[i+j]))
            flag = flag1 and flag2
            if len(box_temp) == 0 or not flag:
                res_list.append(box_list[i])
            else:
                new_box = np.stack(box_temp).mean(0)
                res_list.append(new_box)
        else:
            res_list.append(box_list[i])
    for i in range(len(box_list)-win_size//2, len(box_list)):
        res_list.append(box_list[i])

    return res_list

def count_none_blank(box_list):
    count = 0
    for box in box_list:
        if len(box) > 0:
            count += 1
    return count

def compute_avg_size(box_list):
    count = 0
    temp_list = []
    for box in box_list:
        if len(box) > 0:
            count += 1
            temp_list.append(np.array(box))
    
    if count > 0:
        temp_list = np.stack(temp_list)
        avg_size = np.sqrt((temp_list[:, 2] - temp_list[:, 0]) * (temp_list[:, 3] - temp_list[:, 1])).mean(0)
    else:
        avg_size = 0

    return count, avg_size

def filter_boxes(res_list, box_size_list, max_boxes_num):
    max_boxes_num = min(max_boxes_num, len(res_list))
    keep_ind = np.argsort(box_size_list)[-max_boxes_num:]
    keep_list = []

    for ind in keep_ind:
        keep_list.append(res_list[ind])

    return keep_list

def box_cxcywh_to_xyxy(boxes):
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (Tensor[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (Tensor(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = np.stack((x1, y1, x2, y2), dim=-1)

    return boxes

def box_postporcess(boxes_list, width, height, size_thres=0.2, max_boxes_num=3, frame_min_len=68, frame_margin=14, win_size=17, size_diff_thres=0.3, center_pos_diff_thres=0.2):
    ### boxes_list: n frames, k boxes for each boxes, where k is a variation one

    res_list = []
    box_size_list = []
    res_dict = {}

    img_size = np.sqrt(width * height)

    frame_len = len(boxes_list)
    # boxes_list = torch.tensor(boxes_list)
    boxes_list = [torch.tensor(boxes) for boxes in boxes_list]

    ##### step1: compute fake track through the entire video
    for idx, boxes in enumerate(boxes_list):
        # print(len(boxes))

        boxes = boxes * torch.Tensor([width, height, width, height])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        if idx == 0:
            for bi in range(len(boxes)):
                res_dict['{:02d}'.format(bi)] = [['{:05d}'.format(idx), boxes[bi]]]

        else:
            for bi in range(len(boxes)):
                cur_box = boxes[bi]

                flag = False
                for key in res_dict:
                    temp_box = res_dict[key][-1][1]
                    temp_box_frame_idx = int(res_dict[key][-1][0])
                    if same_box(temp_box, cur_box, size_diff_thres, center_pos_diff_thres) and abs(temp_box_frame_idx - idx) < frame_margin and abs(temp_box_frame_idx - idx) > 0:
                        res_dict[key].append(['{:05d}'.format(idx), cur_box])
                        flag = True
                        break

                if not flag: # new id
                    key_id = len(res_dict) + 1
                    res_dict['{:02d}'.format(key_id)] = [['{:05d}'.format(idx), cur_box]]
    ##### step2: complete sequence for each pid through the entire video
    for key in res_dict:
        
        cur_list = res_dict[key]

        temp_box_list = []
        for idx in range(frame_len):
            if len(cur_list) == 0:
                temp_box_list.append(np.array([-1, -1, -1, -1]))
            else:
                if int(cur_list[0][0]) > idx:
                    temp_box_list.append(np.array([-1, -1, -1, -1]))
                elif int(cur_list[0][0]) == idx:
                    cur_box = cur_list.pop(0)
                    temp_box_list.append(cur_box[1])
                else:
                    raise("error when process box time sequence refine")
        if len(temp_box_list) <= win_size//2:
            continue
        ### complete the false negative due to the ability of grounding-dino 1.5
        temp_box_list = win_complete(temp_box_list, win_size=win_size)

        ### compute appear time and average size for each person
        valid_count, avg_size = compute_avg_size(temp_box_list)
        if valid_count < frame_min_len or avg_size < img_size * size_thres:
            continue
        
        res_list.append(temp_box_list)
        box_size_list.append(avg_size)

    ### filter and keep the three boxes with top average box size
    res_list = filter_boxes(res_list, box_size_list, max_boxes_num=max_boxes_num)

    return res_list

def complete_anyway(boxes):
    len_boxes = len(boxes)
    ii = 0
    while True:
        if len(boxes[ii]) > 0:
            boxes[ii] = np.array(boxes[ii])
            ii += 1
            if ii >= len_boxes:
                break
            continue
        
        # if ii >= len_boxes:
        #     break

        jj = ii + 1
        while True:
            if len(boxes[jj]) > 0:
                break
            jj += 1 

        blank_len = jj - ii
        for idx in range(ii, jj):
            # ii is the first blank, jj is the first value after blank

            boxes[idx] = ((1 - (idx - ii + 1) / (blank_len + 1)) * np.array(boxes[ii-1]) + ((idx - ii + 1) / (blank_len + 1)) * np.array(boxes[jj]))
        
        ii = jj

    return boxes

def compute_center_point(boxes):
    points = []
    labels = []
    for box in boxes:
        points.append([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
        labels.append([1])
    
    return np.stack(points), np.array(labels)