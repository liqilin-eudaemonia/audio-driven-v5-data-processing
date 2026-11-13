"""Lazy, offline pretrained backends. Missing weights never mean random weights."""
import os
from pathlib import Path
import cv2
import numpy as np


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Required local pretrained weight: {path}')
    return path


def batches(frames, size):
    for start in range(0, len(frames), size):
        yield frames[start:start+size]


class Networks:
    def __init__(self, assets, device='cpu', batch_size=16):
        import torch
        self.torch, self.assets = torch, assets
        self.device, self.batch_size = device, batch_size
        self.models = {}

    def tensor(self, frames):
        return self.torch.from_numpy(np.asarray(frames).copy()).permute(0,3,1,2).contiguous().to(self.device).float()/255

    def lpips(self, gt, generated):
        if 'lpips' not in self.models:
            import lpips
            model = lpips.LPIPS(net='alex', version='0.1', pnet_rand=True, verbose=False)
            state = self.torch.load(require_file(self.assets['alexnet']), map_location='cpu', weights_only=True)
            # LPIPS splits torchvision AlexNet.features into five named slices.
            # Load ALL feature tensors strictly before the model can be used.
            mapped = {key: state['features.'+key.split('.', 1)[1]] for key in model.net.state_dict()}
            model.net.load_state_dict(mapped, strict=True)
            self.models['lpips'] = model.eval().to(self.device)
        values = []
        with self.torch.inference_mode():
            for g, p in zip(batches(gt, self.batch_size), batches(generated, self.batch_size)):
                values.extend(self.models['lpips'](self.tensor(g)*2-1, self.tensor(p)*2-1).flatten().cpu().tolist())
        return values

    def fid(self, frames):
        if 'fid' not in self.models:
            from pytorch_fid import inception
            path = require_file(self.assets['fid_inception'])
            original = inception.load_state_dict_from_url
            def local_weights(*args, **kwargs):
                return self.torch.load(path, map_location='cpu', weights_only=True)
            # Upstream constructor only offers URL loading. Restore immediately;
            # this runner is single-threaded at model initialization.
            inception.load_state_dict_from_url = local_weights
            try:
                model = inception.InceptionV3([3], resize_input=True, normalize_input=True,
                                             requires_grad=False, use_fid_inception=True)
            finally:
                inception.load_state_dict_from_url = original
            self.models['fid'] = model.eval().to(self.device)
        result = []
        with self.torch.inference_mode():
            for batch in batches(frames, self.batch_size):
                result.append(self.models['fid'](self.tensor(batch))[0].flatten(1).cpu().numpy())
        return np.concatenate(result)

    def fvd(self, frames):
        if 'fvd' not in self.models:
            self.models['fvd'] = self.torch.jit.load(str(require_file(self.assets['i3d'])),
                                                   map_location=self.device).eval()
        from ..fvd.frechet_video_distance import batch_to_bcthw_uint8
        tensor = batch_to_bcthw_uint8([self.torch.from_numpy(np.asarray(frames).copy())]).to(self.device)
        with self.torch.inference_mode():
            return self.models['fvd'](tensor, rescale=True, resize=False,
                                      return_features=True).flatten(1).cpu().numpy()


class Faces:
    def __init__(self, assets, emotion=True, threads=2, padding=0):
        if type(padding) is not int or padding < 0:
            raise ValueError('Face padding must be a nonnegative integer')
        self.padding=padding
        import onnxruntime as ort
        from insightface.model_zoo.model_zoo import ModelRouter
        root = Path(assets['insightface_root'])
        for filename in ('det_10g.onnx', '2d106det.onnx'):
            require_file(root/'models'/'buffalo_l'/filename)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        # InsightFace 0.7's top-level get_model discards sess_options. Route
        # explicitly so --threads really controls ORT (no hundreds of threads).
        model_dir = root/'models'/'buffalo_l'
        self.detector = ModelRouter(str(model_dir/'det_10g.onnx')).get_model(
            providers=['CPUExecutionProvider'], sess_options=options)
        self.landmarker = ModelRouter(str(model_dir/'2d106det.onnx')).get_model(
            providers=['CPUExecutionProvider'], sess_options=options)
        self.detector.prepare(ctx_id=-1, det_thresh=.2, input_size=(640,640))
        self.landmarker.prepare(ctx_id=-1)
        self.emotion = None
        if emotion:
            weight = require_file(assets['emotion'])
            # DeepFace's official constructor loads from DEEPFACE_HOME/.deepface.
            home = weight.parent/'deepface_runtime'
            destination = home/'.deepface'/'weights'/weight.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.symlink_to(weight.resolve())
            if destination.resolve() != weight.resolve():
                raise ValueError('DeepFace cache points to a different emotion checkpoint')
            os.environ['DEEPFACE_HOME'] = str(home)
            os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
            import tensorflow as tf
            tf.config.set_visible_devices([], 'GPU')
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
            from deepface.models.demography.Emotion import EmotionClient
            self.emotion = EmotionClient()

    def extract(self, frames):
        landmarks = np.full((len(frames),106,2), np.nan)
        boxes = np.full((len(frames),4), np.nan)
        labels = np.full(len(frames), -1, dtype=int)
        face_counts = []
        crops, indices = [], []
        for i, rgb in enumerate(frames):
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pad=self.padding
            detection_image=cv2.copyMakeBorder(bgr,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=(0,0,0)) if pad else bgr
            boxes_found, keypoints = self.detector.detect(detection_image, max_num=0, metric='default')
            face_counts.append(len(boxes_found))
            # Fixed single-identity listener crop. Never silently switch identities.
            if len(boxes_found) != 1:
                continue
            from insightface.app.common import Face
            face = Face(bbox=boxes_found[0,:4], det_score=boxes_found[0,4],
                        kps=keypoints[0] if keypoints is not None else None)
            self.landmarker.get(detection_image,face)
            # Return to original image coordinates; emotion still reads original pixels.
            box = np.asarray(face.bbox, float)-pad
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x1, y1 = max(x1,0), max(y1,0)
            x2, y2 = min(x2,bgr.shape[1]), min(y2,bgr.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue
            landmarks[i] = face.landmark_2d_106-pad
            boxes[i] = box
            if self.emotion is not None:
                from deepface.modules.preprocessing import resize_image
                # DeepFace.extract_faces normalizes BEFORE float interpolation.
                # Resizing uint8 first introduces avoidable quantization differences.
                crops.append(resize_image(bgr[y1:y2,x1:x2].astype(np.float64)/255,
                                          target_size=(224,224))[0])
                indices.append(i)
        if crops:
            for start in range(0,len(crops),32):
                scores = np.atleast_2d(self.emotion.predict(crops[start:start+32]))
                if scores.shape[1] != 7 or not np.isfinite(scores).all():
                    raise ValueError('Invalid DeepFace emotion prediction')
                labels[indices[start:start+32]] = scores.argmax(axis=1)
        return landmarks, boxes, labels, face_counts


def normalize_landmarks(gt, generated, gt_boxes, generated_boxes):
    """Shared fixed square size, EMA bbox center, 224 coordinates.

    Adapted from AnyTalker. Do not replace motion outliers with stationary points
    or bridge a missed detection. Both videos share the same spatial scale.
    """
    both = np.concatenate([gt_boxes, generated_boxes])
    valid = np.isfinite(both).all(1)
    if not valid.any():
        return gt.copy(), generated.copy(), None
    side = max(224., float(np.max(both[valid,2:]-both[valid,:2])))
    result = []
    for points, boxes in ((gt,gt_boxes), (generated,generated_boxes)):
        normalized = np.full_like(points, np.nan)
        previous = None
        for i, box in enumerate(boxes):
            if not np.isfinite(box).all():
                previous = None
                continue
            center = (box[:2]+box[2:])/2
            center = center if previous is None else .7*center + .3*previous
            previous = center
            normalized[i] = (points[i]-center+side/2)*224/side
        result.append(normalized)
    return *result, side
