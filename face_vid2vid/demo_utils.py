import os
import sys
import cv2
import yaml
import imageio
import numpy as np
import torch
import torch.nn.functional as F


sys.path.append("./face-vid2vid")
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from animate import normalize_kp
from batch_face import RetinaFace


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareSPADEGenerator(**config["model_params"]["generator_params"], **config["model_params"]["common_params"])
    # convert to half precision to speed up
    generator.cuda().half()

    kp_detector = KPDetector(**config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"])
    # the result will be wrong if converted to half precision, not sure why
    kp_detector.cuda()  # .half()

    he_estimator = HEEstimator(**config["model_params"]["he_estimator_params"], **config["model_params"]["common_params"])
    # the result will be wrong if converted to half precision, not sure why
    he_estimator.cuda()  # .half()

    print("Loading checkpoints")
    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint["generator"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    he_estimator.load_state_dict(checkpoint["he_estimator"])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)
    he_estimator = DataParallelWithCallback(he_estimator)

    generator.eval()
    kp_detector.eval()
    he_estimator.eval()
    print("Model successfully loaded!")

    return generator, kp_detector, he_estimator


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred, dim=1)
    degree = torch.sum(pred * idx_tensor, axis=1) * 3 - 99

    return degree


def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat(
        [
            torch.ones_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.zeros_like(pitch),
            torch.cos(pitch),
            -torch.sin(pitch),
            torch.zeros_like(pitch),
            torch.sin(pitch),
            torch.cos(pitch),
        ],
        dim=1,
    )
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat(
        [
            torch.cos(yaw),
            torch.zeros_like(yaw),
            torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.ones_like(yaw),
            torch.zeros_like(yaw),
            -torch.sin(yaw),
            torch.zeros_like(yaw),
            torch.cos(yaw),
        ],
        dim=1,
    )
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat(
        [
            torch.cos(roll),
            -torch.sin(roll),
            torch.zeros_like(roll),
            torch.sin(roll),
            torch.cos(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.zeros_like(roll),
            torch.ones_like(roll),
        ],
        dim=1,
    )
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum("bij,bjk,bkm->bim", pitch_mat, yaw_mat, roll_mat)

    return rot_mat


def keypoint_transformation(kp_canonical, he, estimate_jacobian=False, free_view=False, yaw=0, pitch=0, roll=0, output_coord=False):
    kp = kp_canonical["value"]
    if not free_view:
        yaw, pitch, roll = he["yaw"], he["pitch"], he["roll"]
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)
    else:
        if yaw is not None:
            yaw = torch.tensor([yaw]).cuda()
        else:
            yaw = he["yaw"]
            yaw = headpose_pred_to_degree(yaw)
        if pitch is not None:
            pitch = torch.tensor([pitch]).cuda()
        else:
            pitch = he["pitch"]
            pitch = headpose_pred_to_degree(pitch)
        if roll is not None:
            roll = torch.tensor([roll]).cuda()
        else:
            roll = he["roll"]
            roll = headpose_pred_to_degree(roll)

    t, exp = he["t"], he["exp"]

    rot_mat = get_rotation_matrix(yaw, pitch, roll)

    # keypoint rotation
    kp_rotated = torch.einsum("bmp,bkp->bkm", rot_mat, kp)

    # keypoint translation
    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical["jacobian"]
        jacobian_transformed = torch.einsum("bmp,bkps->bkms", rot_mat, jacobian)
    else:
        jacobian_transformed = None

    if output_coord:
        return {"value": kp_transformed, "jacobian": jacobian_transformed}, {
            "yaw": float(yaw.cpu().numpy()),
            "pitch": float(pitch.cpu().numpy()),
            "roll": float(roll.cpu().numpy()),
        }

    return {"value": kp_transformed, "jacobian": jacobian_transformed}


def get_square_face(coords, image):
    x1, y1, x2, y2 = coords
    # expand the face region by 1.5 times
    length = max(x2 - x1, y2 - y1) // 2
    x1 = x1 - length * 0.5
    x2 = x2 + length * 0.5
    y1 = y1 - length * 0.5
    y2 = y2 + length * 0.5

    # get square image
    center = (x1 + x2) // 2, (y1 + y2) // 2
    length = max(x2 - x1, y2 - y1) // 2
    x1 = max(int(round(center[0] - length)), 0)
    x2 = min(int(round(center[0] + length)), image.shape[1])
    y1 = max(int(round(center[1] - length)), 0)
    y2 = min(int(round(center[1] + length)), image.shape[0])
    return image[y1:y2, x1:x2]


def smooth_coord(last_coord, current_coord, smooth_factor=0.2):
    change = np.array(current_coord) - np.array(last_coord)
    # smooth the change to 0.1 times
    change = change * smooth_factor
    return (np.array(last_coord) + np.array(change)).astype(int).tolist()


class FaceAnimationClass:
    def __init__(self, source_image_path=None, use_sr=False):
        assert source_image_path is not None, "source_image_path is None, please set source_image_path"
        config_path = os.path.join(os.path.dirname(__file__), "face_vid2vid/config/vox-256-spade.yaml")
        # save to local cache to speed loading
        checkpoint_path = os.path.join(os.path.expanduser("~"), ".cache/torch/hub/checkpoints/FaceMapping.pth.tar")
        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            from gdown import download
            file_id = "11ZgyjKI5OcB7klcsIdPpCCX38AIX8Soc"
            download(id=file_id, output=checkpoint_path, quiet=False)
        if use_sr:
            from face_vid2vid.GPEN.face_enhancement import FaceEnhancement

            self.faceenhancer = FaceEnhancement(
                size=256, model="GPEN-BFR-256", use_sr=False, sr_model="realesrnet_x2", channel_multiplier=1, narrow=0.5, use_facegan=True
            )

        # load checkpoints
        self.generator, self.kp_detector, self.he_estimator = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path)
        source_image = cv2.cvtColor(cv2.imread(source_image_path), cv2.COLOR_RGB2BGR).astype(np.float32) / 255.
        source_image = cv2.resize(source_image, (256, 256), interpolation=cv2.INTER_AREA)
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        self.source = source.cuda()

        # initilize face detectors
        self.face_detector = RetinaFace()
        self.detect_interval = 8
        self.smooth_factor = 0.2

        # load base frame and blank frame
        self.base_frame = cv2.imread(source_image_path) if not use_sr else self.faceenhancer.process(cv2.imread(source_image_path))[0]
        self.base_frame = cv2.resize(self.base_frame, (256, 256))
        self.blank_frame = np.ones(self.base_frame.shape, dtype=np.uint8) * 255
        cv2.putText(self.blank_frame, "Face not", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(self.blank_frame, "detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # count for frame
        self.n_frame = 0

        # initilize variables
        self.first_frame = True
        self.last_coords = None
        self.coords = None
        self.use_sr = use_sr
        self.kp_source = None
        self.kp_driving_initial = None


    def _conver_input_frame(self, frame):
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
        return torch.tensor(frame[np.newaxis]).permute(0, 3, 1, 2).cuda()

    def _process_first_frame(self, frame):
        print("Processing first frame")
        # function to process the first frame
        faces = self.face_detector(frame, cv=True)
        if len(faces) == 0:
            raise ValueError("Face is not detected")
        else:
            self.coords = faces[0][0]
        face = get_square_face(self.coords, frame)
        self.last_coords = self.coords

        # get the keypoint and headpose from the source image
        with torch.no_grad():
            self.kp_canonical = self.kp_detector(self.source)
            self.he_source = self.he_estimator(self.source)

            face_input = self._conver_input_frame(face)
            he_driving_initial = self.he_estimator(face_input)
            self.kp_driving_initial, coordinates = keypoint_transformation(self.kp_canonical, he_driving_initial, output_coord=True)
            self.kp_source = keypoint_transformation(
                self.kp_canonical, self.he_source, free_view=True, yaw=coordinates["yaw"], pitch=coordinates["pitch"], roll=coordinates["roll"]
            )

    def _inference(self, frame):
        # function to process the rest frames
        with torch.no_grad():
            self.n_frame += 1
            if self.first_frame:
                self._process_first_frame(frame)
                self.first_frame = False
            else:
                pass
            if self.n_frame % self.detect_interval == 0:
                faces = self.face_detector(frame, cv=True)
                if len(faces) == 0:
                    raise ValueError("Face is not detected")
                else:
                    self.coords = faces[0][0]
                    self.coords = smooth_coord(self.last_coords, self.coords, self.smooth_factor)
            face = get_square_face(self.coords, frame)
            self.last_coords = self.coords
            face_input = self._conver_input_frame(face)

            he_driving = self.he_estimator(face_input)
            kp_driving = keypoint_transformation(self.kp_canonical, he_driving)
            kp_norm = normalize_kp(
                kp_source=self.kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=self.kp_driving_initial,
                use_relative_movement=True,
                adapt_movement_scale=True,
            )

            out = self.generator(self.source, kp_source=self.kp_source, kp_driving=kp_norm, fp16=True)
            image = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
            image = (np.array(image).astype(np.float32) * 255).astype(np.uint8)
            result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return face, result

    def inference(self, frame):
        # function to inference, input frame, output cropped face and its result
        try:
            if frame is not None:
                face, result = self._inference(frame)
                if self.use_sr:
                    result, _, _ = self.faceenhancer.process(result)
                    result = cv2.resize(result, (256, 256))
                return face, result
        except Exception as e:
            print(e)
            self.first_frame = True
            self.n_frame = 0
            return self.blank_frame, self.base_frame


if __name__ == "__main__":
    from tqdm import tqdm
    import time
    faceanimation = FaceAnimationClass(source_image_path="tmp.png", use_sr=False)

    video_path = "driver.mp4"
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    _, frame = capture.read()
    while frame is not None:
        frames.append(frame)
        _, frame = capture.read()
    capture.release()

    output_frames = []
    time_start = time.time()
    for frame in tqdm(frames):
        face, result = faceanimation.inference(frame)
        # show = cv2.hconcat([cv2.resize(face, (result.shape[1], result.shape[0])), result])
        output_frames.append(result)
    time_end = time.time()
    print("Time cost: %.2f" % (time_end - time_start), "FPS: %.2f" % (len(frames) / (time_end - time_start)))
    writer = imageio.get_writer("result2.mp4", fps=fps, quality=9, macro_block_size=1, codec="libx264", pixelformat="yuv420p")
    for frame in output_frames:
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # writer.append_data(frame)
    writer.close()
    print("Video saved to result2.mp4")
