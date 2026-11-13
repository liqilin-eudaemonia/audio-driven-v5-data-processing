import os
import cv2
import math
import torch
import torchvision.transforms as transforms
import numpy as np
from lib.rt_gene.src.rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from lib.rt_gene.src.rt_gene.tracker_face_encoding import FaceEncodingTracker
# from src.rt_gene.tracker_sequential import SequentialTracker
from lib.rt_gene.src.rt_gene.SFD.sfd_detector import SFDDetector
import lib.rt_gene.src.rt_gene.ThreeDDFA.mobilenet_v1 as mobilenet_v1
from lib.rt_gene.src.rt_gene.estimate_gaze_pytorch import GazeEstimator
from lib.rt_gene.src.rt_bene.estimate_blink_pytorch import BlinkEstimatorPytorch

from lib.rt_gene.src.rt_gene.ThreeDDFA.ddfa import ToTensorGjz, NormalizeGjz
from lib.rt_gene.src.rt_gene.ThreeDDFA.inference import crop_img, predict_68pts, parse_roi_box_from_bbox, parse_roi_box_from_landmark
from lib.rt_gene.src.rt_gene.tracker_generic import TrackedSubject
import lib.rt_gene.src.rt_gene.gaze_tools as gaze_tools
import lib.rt_gene.src.rt_gene.ros_tools as ros_tools


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.

    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

class GazeBlink(object):
    def __init__(
        self, 
        fd_model_path = "/mnt/wangduomin/data/models/gaze/rt_gene/models/SFD/s3fd_facedetector.pth",
        ldmk_model_path = "/mnt/wangduomin/data/models/gaze/rt_gene/models/phase1_wpdc_vdc.pth.tar",
        gaze_model_path = ["/mnt/wangduomin/data/models/gaze/rt_gene/models/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model"],
        blink_model_path = ["/mnt/wangduomin/data/models/gaze/rt_gene/models/blink_model_pytorch_vgg16_allsubjects1.model"],
        model_points_file = "/mnt/wangduomin/data/models/gaze/rt_gene/models/face_model_68.txt",
        dlib_path = "/mnt/wangduomin/data/models/gaze/rt_gene/models/dlib_face_recognition_resnet_model_v1.dat"
    ):

        self.model_size_rescale = 16.0
        self.head_pitch = 0.0
        self.interpupillary_distance = 0.058
        self.eye_image_size = (60, 36)

        #### face detector
        self.face_net = SFDDetector(device='cuda:0', path_to_detector=fd_model_path)

        #### landmark detector
        self.ldmk_net = mobilenet_v1.mobilenet_1()
        checkpoint = torch.load(ldmk_model_path, map_location=lambda storage, loc: storage)['state_dict']
        model_dict = self.ldmk_net.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.ldmk_net.load_state_dict(model_dict)

        #### blink estimator
        self.blink_estimator = BlinkEstimatorPytorch('cuda:0', blink_model_path, 'vgg16', threshold=0.5)

        #### gaze estimator
        self.gaze_estimator = GazeEstimator('cuda:0', gaze_model_path)

        #### eval and cuda
        # self.face_net.cuda()
        self.ldmk_net.cuda()
        # self.blink_estimator.cuda()
        # self.gaze_estimator.cuda()
        
        # self.face_net.eval()
        self.ldmk_net.eval()
        # self.blink_estimator.eval()
        # self.gaze_estimator.eval()
        
        
        raw_value = []
        with open(model_points_file) as f:
            for line in f:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # index the expansion of the model based.
        self.model_points = model_points * (self.interpupillary_distance * self.model_size_rescale)

        # ldmk_transform
        self.facial_landmark_transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

        # face tracker
        self.subject_tracker = FaceEncodingTracker(dlib_path)

    def get_face_bb(self, image):
        faceboxes = []
        fraction = 4.0
        image = cv2.resize(image, (0, 0), fx=1.0 / fraction, fy=1.0 / fraction)
        detections = self.face_net.detect_from_image(image)

        for result in detections:
            # scale back up to image size
            box = result[:4]
            confidence = result[4]

            if gaze_tools.box_in_image(box, image) and confidence > 0.6:
                box = [x * fraction for x in box]  # scale back up
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = gaze_tools.move_box(box, [0, offset_y])

                # Make box square.
                facebox = gaze_tools.get_square_box(box_moved)
                faceboxes.append(facebox)

        return faceboxes

    def ddfa_forward_pass(self, color_img, roi_box_list):
        img_step = [crop_img(color_img, roi_box) for roi_box in roi_box_list]
        img_step = [cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR) for img in img_step]
        _input = torch.cat([self.facial_landmark_transform(img).unsqueeze(0) for img in img_step], 0)
        with torch.no_grad():
            _input = _input.cuda()
            param = self.ldmk_net(_input).cpu().numpy().astype(np.float32)

        return [predict_68pts(p.flatten(), roi_box) for p, roi_box in zip(param, roi_box_list)]

    def get_subjects_from_faceboxes(self, color_img, faceboxes):
        face_images = [gaze_tools.crop_face_from_image(color_img, b) for b in faceboxes]
        subjects = []
        roi_box_list = [parse_roi_box_from_bbox(facebox) for facebox in faceboxes]
        initial_pts68_list = self.ddfa_forward_pass(color_img, roi_box_list)
        roi_box_refined_list = [parse_roi_box_from_landmark(initial_pts68) for initial_pts68 in initial_pts68_list]
        pts68_list = self.ddfa_forward_pass(color_img, roi_box_refined_list)

        for pts68, face_image, facebox in zip(pts68_list, face_images, faceboxes):
            np_landmarks = np.array((pts68[0], pts68[1])).T
            subjects.append(TrackedSubject(np.array(facebox), face_image, np_landmarks))
        return subjects


    def update_subject_tracker(self, color_img):
        faceboxes = self.get_face_bb(color_img)
        if len(faceboxes) == 0:
            self.subject_tracker.clear_elements()
            return

        tracked_subjects = self.get_subjects_from_faceboxes(color_img, faceboxes)

        # track the new faceboxes according to the previous ones
        self.subject_tracker.track(tracked_subjects)


    def get_head_pose(self, img_size, landmarks):

        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        success, rodrigues_rotation, translation_vector, _ = cv2.solvePnPRansac(self.model_points,
                                                                                landmarks.reshape(len(self.model_points), 1, 2),
                                                                                cameraMatrix=camera_matrix,
                                                                                distCoeffs=dist_coeffs, flags=cv2.SOLVEPNP_DLS)

        # if self._pnp_iterate_after:
        #     success, rodrigues_rotation, translation_vector = cv2.solvePnP(self.model_points,
        #                                                                    landmarks.reshape(len(self.model_points), 1, 2),
        #                                                                    rvec=rodrigues_rotation,
        #                                                                    tvec=translation_vector,
        #                                                                    useExtrinsicGuess=True,
        #                                                                    cameraMatrix=camera_matrix,
        #                                                                    distCoeffs=dist_coeffs,
        #                                                                    flags=cv2.SOLVEPNP_ITERATIVE)


        _rotation_matrix, _ = cv2.Rodrigues(rodrigues_rotation)
        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        _m = np.zeros((4, 4))
        _m[:3, :3] = _rotation_matrix
        _m[3, 3] = 1
        pitch, yaw, roll = np.array(euler_from_matrix(_m))

        rotation = quaternion_from_euler(pitch, yaw, roll)

        return pitch, yaw, roll, rotation


    def get_gaze_blink(self, image):

        h, w, c = image.shape

        self.update_subject_tracker(image)
        self.subject_tracker.update_eye_images(self.eye_image_size)

        input_blink_list = [[], []]
        input_gaze_list = [[], []]
        input_head_list = []
        
        gaze_est = [None, None]
        probs = [None]
        gaze_feat = [None]

        try:
            for subject_id, subject in self.subject_tracker.get_tracked_elements().items():
                pitch, yaw, roll, rotation = self.get_head_pose([h, w], subject.landmarks)

                img = image.copy()
                for ldmk in subject.landmarks:
                    x = int(ldmk[0])
                    y = int(ldmk[1])
                    img[y, x] = [255, 255, 255]

                img_l = subject.left_eye_color.copy()
                img_r = subject.right_eye_color.copy()
                ## process blink img
                b_left, b_right = self.blink_estimator.inputs_from_images(img_r, img_l)
                ## process gaze img
                g_left = self.gaze_estimator.input_from_image(img_r)
                g_right = self.gaze_estimator.input_from_image(img_l)

                input_blink_list[0].append(b_left)
                input_blink_list[1].append(b_right)
                input_gaze_list[0].append(g_left)
                input_gaze_list[1].append(g_right)

                _m = quaternion_matrix([rotation[0], rotation[1], rotation[2], rotation[3]])
                euler_angles_head = list(euler_from_matrix(np.dot(ros_tools.camera_to_ros, _m)))
                euler_angles_head = gaze_tools.limit_yaw(euler_angles_head)

                phi_head, theta_head = gaze_tools.get_phi_theta_from_euler(euler_angles_head)
                # phi_head, theta_head = gaze_tools.get_phi_theta_from_euler([float(pitch), float(yaw), float(roll)])
                input_head_list.append([theta_head, phi_head])

            gaze_est, gaze_feat = self.gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_gaze_list[0],
                                                                inference_input_right_list=input_gaze_list[1],
                                                                inference_headpose_list=input_head_list)
            probs = self.blink_estimator.predict(input_blink_list[0], input_blink_list[1])

            # print(gaze_est * 180 / np.pi, probs) #, pitch, yaw, roll, phi_head, theta_head)
            gaze_est = gaze_est[0] * 180 / np.pi
            probs = probs[0]

            self.subject_tracker.clear_elements()
        except Exception as e:
            print(e)
            return None, None, None, None

        return gaze_est[0], gaze_est[1], probs[0], gaze_feat[0]


if __name__ == "__main__":
    detector = GazeBlink()

    # gaze_img = cv2.imread("/mnt/wangduomin/data/zhongzhiren/aligned_images/lip_sync_data_test2_cut_9s_30s/aligned_images_224/000118.png") # (-3.923969 -11.200309), 0.00515187
    # gaze_img = cv2.imread("/mnt/wangduomin/data/zhongzhiren/aligned_images/lip_sync_data_test2_cut_9s_30s/aligned_images_224/000278.png") # (-5.2766967  3.5040658), 0.00598543
    gaze_img = cv2.imread("/mnt/wangduomin/data/zhongzhiren/aligned_images/lip_sync_data_test2_cut_9s_30s/aligned_images_224/000305.png") # (11.496355 13.072686), 0.00914004
    blink_img = cv2.imread("/mnt/wangduomin/data/zhongzhiren/aligned_images/lip_sync_data_test2_cut_9s_30s/aligned_images_224/000195.png") # (3.743553 4.174017), 0.12036792

    img, img_l, img_r = detector.get_gaze_blink(gaze_img)
    cv2.imwrite("gaze_img_ldmk.jpg", img)
    cv2.imwrite("gaze_img_l.jpg", img_l)
    cv2.imwrite("gaze_img_r.jpg", img_r)
    img, img_l, img_r = detector.get_gaze_blink(blink_img)
    cv2.imwrite("blink_img_ldmk.jpg", img)
    cv2.imwrite("blink_img_l.jpg", img_l)
    cv2.imwrite("blink_img_r.jpg", img_r)