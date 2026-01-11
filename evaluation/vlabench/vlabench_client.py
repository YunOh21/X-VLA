import os
import argparse
from pathlib import Path
os.environ["MUJOCO_GL"]= "egl"

if "VLABENCH_ROOT" not in os.environ:
    os.environ["VLABENCH_ROOT"] = str(
        Path(__file__).resolve().parent / "VLABench" / "VLABench"
    )

from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.tasks import *
from VLABench.robots import *

import json_numpy
import collections
import requests
import PIL.Image as Image
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import datetime

# ---- Visual Prompting Start ----
from scipy.spatial.transform import Rotation as R

def get_matrix_from_mujoco_config(pos_str, xyaxes_str):
    """
    이미지의 로직을 반영하여 MuJoCo config 문자열로부터
    OpenCV 스타일의 Extrinsic Matrix (World -> Camera)를 계산
    """
    # 1. Parse config
    cam_pos = np.array([float(x) for x in pos_str.split()])
    vals = [float(x) for x in xyaxes_str.split()]
    
    # MuJoCo Camera axes (Camera to World orientation)
    xaxis = np.array(vals[0:3])
    yaxis = np.array(vals[3:6])
    
    # Normalize
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis) # MuJoCo looks towards -Z
    
    # R_mujoco (Camera to World Rotation Matrix)
    # Columns are the axes of the camera in world frame
    R_c2w = np.vstack([xaxis, yaxis, zaxis]).T
    
    # 2. Convert to OpenCV Coordinate System
    # MuJoCo: Right(X), Up(Y), Back(Z) -> Look at -Z
    # OpenCV: Right(X), Down(Y), Forward(Z) -> Look at +Z
    
    R_ex = R_c2w.T.copy()
    R_ex[1, :] = -R_ex[1, :] # Flip Y (Up -> Down)
    R_ex[2, :] = -R_ex[2, :] # Flip Z (Back -> Forward)
    
    # 3. Calculate Translation t
    # t = -R * pos (Standard extrinsic translation formula)
    t = -np.dot(R_ex, cam_pos)
    
    # 4. Construct 4x4 Extrinsic Matrix (World -> Camera)
    ex_mat = np.eye(4)
    ex_mat[:3, :3] = R_ex
    ex_mat[:3, 3] = t
    
    return ex_mat

def get_extrinsics_matrix(cam_pos, cam_quat):
    """
    카메라의 Pose(World -> Camera)를 4x4 행렬로 변환 (Quaternion 입력용)
    """
    rotation = R.from_quat(cam_quat).as_matrix() 
    
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = rotation
    cam_to_world[:3, 3] = cam_pos
    
    world_to_cam = np.linalg.inv(cam_to_world)
    return world_to_cam

def add_visual_prompt(image, ee_pos, camera_intrinsics, extrinsic_matrix, gripper_state):
    """
    extrinsic_matrix: World -> Camera 4x4 행렬
    """
    img = image.copy()
    
    ee_2d = project_3d_to_2d(ee_pos, camera_intrinsics, extrinsic_matrix)
    
    if ee_2d is None:
        return img
    
    x, y = int(ee_2d[0]), int(ee_2d[1])
    h, w = img.shape[:2]
    
    # 화면 밖 체크
    if not (0 <= x < w and 0 <= y < h):
        return img
    
    is_open = gripper_state > 0.04 # 보통 Robosuite에서 닫히면 0에 가까움
    
    if is_open:
        # Open: Red (opencv)
        cv2.circle(img, (x, y), radius=7, color=(0, 0, 255), thickness=-1) 
        cv2.circle(img, (x, y), radius=8, color=(255, 255, 255), thickness=1)
    else:
        # Closed: Blue (opencv)
        cv2.circle(img, (x, y), radius=7, color=(255, 0, 0), thickness=-1) 
        cv2.circle(img, (x, y), radius=8, color=(255, 255, 255), thickness=1)
        
    return img

def project_3d_to_2d(point_3d, intrinsics, world_to_cam_pose):
    """
    Args:
        point_3d: (x, y, z) 월드 좌표
        intrinsics: 3x3 Camera Intrinsics
        world_to_cam_pose: 4x4 Extrinsics Matrix (World -> Camera)
    """
    # 1. Homogeneous 좌표로 변환 [x, y, z, 1]
    point_3d_homo = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    
    # 2. World 좌표 -> Camera 좌표 변환
    # world_to_cam_pose (ex_mat)가 이미 MuJoCo -> OpenCV 변환을 포함하고 있으므로
    # 별도의 축 반전(point_cam[1] *= -1 등)이 필요 없음
    point_cam = world_to_cam_pose @ point_3d_homo
    
    # Z값(깊이)이 0보다 작거나 같으면(카메라 뒤 혹은 평면) 투영하지 않음
    if point_cam[2] <= 0:
        return None 

    # 3. Camera 좌표 -> Image 픽셀 (Intrinsics 적용)
    uv_homo = intrinsics @ point_cam[:3]
    u = uv_homo[0] / uv_homo[2]
    v = uv_homo[1] / uv_homo[2]
    
    return np.array([u, v])

def get_camera_intrinsics(fov, img_width, img_height):
    """
    Get camera intrinsic matrix from FOV, using the centering logic from the provided image.
    
    Args:
        fov: Field of view in degrees (Vertical FOV assumed usually, or Horizontal depending on config)
        img_width, img_height: Image dimensions
    
    Returns:
        K: 3x3 intrinsic matrix
    """
    # Focal length calculation based on FOV (assuming fovy here as strictly typical in Mujoco)
    # fovy = 2 * arctan(h / (2 * fy))  => fy = h / (2 * tan(fovy/2))
    # Assuming square pixels, fx = fy
    
    focal_length = (img_height / 2) / np.tan(np.radians(fov / 2))
    
    # Center calculation updated based on image reference: (WIDTH - 1) / 2
    cx = (img_width - 1) / 2
    cy = (img_height - 1) / 2

    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ])
    
    return K

# ---- Visual Prompting End ----

def get_tcp_position(ee_pos, ee_quat, offset_dist=0.15):
    rot_mat = R.from_quat(ee_quat).as_matrix()
    offset_vec = np.array([0, 0, 1]) * offset_dist
    tcp_pos = ee_pos + (rot_mat @ offset_vec)
    return tcp_pos

def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def rotate6D_to_euler(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    euler = R.from_matrix(rot_mats).as_euler('xyz', degrees=False)
    return euler


class ClientModel():
    def __init__(self,
                 host,
                 port,
                 control_mode = 'ee'):
        
        self.url = f"http://{host}:{port}/act"
        assert control_mode in ['ee', 'joint']
        self.control_mode = control_mode
        self.name = 'hdp'
        
        self.reset()
        
    def reset(self):
        """
        This is called
        """
        # currently, we dont use historical observation, so we dont need this fc
        
        self.action_plan = collections.deque()
        return None
    
    def _post(self, payload: Dict) -> np.ndarray:
        resp = requests.post(self.url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        action = np.array(data["action"])  # shape (T, 10) expected: [pos3, rot6d, grip1]
        attn_map = np.array(data.get("attn_map")) if data.get("attn_map") is not None else None

        # If server returned an overlay image, decode and save it for debugging
        overlay_img = None
        if data.get("overlay") is not None:
            try:
                overlay_arr = json_numpy.loads(data["overlay"])
                # Ensure uint8
                if overlay_arr.dtype != np.uint8:
                    if overlay_arr.max() <= 1.0:
                        overlay_arr = (overlay_arr * 255).astype(np.uint8)
                    else:
                        overlay_arr = overlay_arr.astype(np.uint8)
                # Save overlay
                    save_dir = os.path.join("logs", "prompt")
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{save_dir}/{timestamp}_overlay.jpg"
                # overlay_arr is RGB -> convert to BGR for cv2
                try:
                    bgr = cv2.cvtColor(overlay_arr, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filename, bgr)
                except Exception:
                    # fallback: save with PIL
                    Image.fromarray(overlay_arr).save(filename)
                overlay_img = overlay_arr
            except Exception:
                overlay_img = None

        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
        return action, attn_map, overlay_img
    
    def save_attention_overlay(self, language_instruction, image_rgb, attn_map):
        if attn_map is None:
            return

        save_dir = os.path.join("logs", "attention")
        os.makedirs(save_dir, exist_ok=True)
        
        attn = np.array(attn_map).flatten()
        # print("--------- print attn start ------------")
        # print(attn)
        # print("--------- print attn end ------------")
        seq_len = len(attn)
        side = int(np.sqrt(seq_len))

        if side * side != seq_len:
            target_dim = 16 
            if seq_len >= target_dim**2:
                attn = attn[-(target_dim**2):]
                side = target_dim
            else:
                target_dim = 14
                if seq_len >= target_dim**2:
                     attn = attn[-(target_dim**2):]
                     side = target_dim

        try:
            attn_map = attn.reshape(side, side)
        except:
            print(f"[Skip] attention map size unmatch: {seq_len}")
            return

        h, w = image_rgb.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)

        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        attn_uint8 = (attn_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)

        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{save_dir}/{language_instruction}_{timestamp}.jpg"
        cv2.imwrite(filename, overlay)
        print(f"Saved: {filename}")

    def predict(self, obs, **kwargs):

        """
        Args:
            obs: (dict) environment observations
        Returns:
            action: (np.array) predicted action
        """
        if not self.action_plan:
            multiview = obs['rgb']  # # np.ndarray with shape (4, 480, 480, 3)
            
            main_view = multiview[0]   # np.ndarray with shape (480, 480, 3)
            front_view = multiview[2]   # np.ndarray with shape (480, 480, 3)
            wrist_view = multiview[-1]   # np.ndarray with shape (480, 480, 3)
            
            # proprio
            proprio = obs['ee_state'] # np.ndarray with shape (1, 8)
            ee_pos, ee_quat, gripper = proprio[:3], proprio[3:7], proprio[7:8]
                        
            # ===== VISUAL PROMPTING 추가 =====
            # from camera.xml
            pos = "-0.016 1.223 1.644"
            xyaxes = "-1.000 -0.015 -0.000 0.008 -0.551 0.834"
            fovy = 45.0  # mujoco default fov
            
            # 행렬 계산 (새로운 helper 함수 사용)
            # World -> Camera Extrinsic Matrix directly computed
            extrinsic_matrix = get_matrix_from_mujoco_config(pos, xyaxes)
            K = get_camera_intrinsics(fov=fovy, img_width=480, img_height=480)

            current_gripper_val = float(gripper[0])

            # 점 찍기
            front_view_prompted = add_visual_prompt(front_view, ee_pos, K, extrinsic_matrix, gripper_state=current_gripper_val)
            
            instruction_with_prompt = f"""
            Your body is Franka robot.
            Blue dot shows your gripper is open.
            Green dot shows your gripper is closed.
            {obs['instruction']}
            """

            # ===== END VISUAL PROMPTING =====
            
            ee_6d = np.array(quat_to_rotate6d(ee_quat))
            ee_pos -= np.array([0, -0.4, 0.78])
            ee_state = np.concatenate([ee_pos, ee_6d, gripper], axis=0)
            proprio = np.concatenate([ee_state, np.zeros_like(ee_state)], axis=0).copy()
            
            # ==== 전송 전 이미지 저장 ====
            debug_dir = os.path.join("logs", "prompt")
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            def save_check_image(name, img_arr):
                to_save = img_arr.copy()
                if to_save.dtype != np.uint8:
                    if to_save.max() <= 1.0:
                        to_save = (to_save * 255).astype(np.uint8)
                    else:
                        to_save = to_save.astype(np.uint8)
                
                to_save = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)
                
                filename = f"{debug_dir}/{timestamp}_{obs['instruction']}_{name}.jpg"
                cv2.imwrite(filename, to_save)
                # print(f"[DEBUG] Saved prompt image: {filename}")

            save_check_image("front_view", front_view_prompted)
            # =================================================================

            query = {
                "proprio": json_numpy.dumps(proprio),
                "language_instruction": instruction_with_prompt,
                "image0": json_numpy.dumps(main_view),
                "image1": json_numpy.dumps(front_view_prompted),
                "image2": json_numpy.dumps(wrist_view),
                "domain_id": 8,
                "steps": 10,
            }
            # Include camera parameters so server can perform accurate projection
            query["camera_intrinsics"] = json_numpy.dumps(K)
            query["camera_extrinsics"] = json_numpy.dumps(extrinsic_matrix)

            action, attn_map, overlay_img = self._post(query)

            self.save_attention_overlay(obs['instruction'], main_view, attn_map)
            if overlay_img is not None:
                # also save the overlay next to other debug images
                debug_dir = "debug_prompts_logs"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                overlay_filename = f"{debug_dir}/{timestamp}_{obs['instruction']}_server_overlay.jpg"
                try:
                    bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(overlay_filename, bgr)
                except Exception:
                    Image.fromarray(overlay_img).save(overlay_filename)

            target_eef = action[:, :3]
            target_euler = rotate6D_to_euler(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_euler, target_act], axis=-1)

            for row in final_action.tolist():
                self.action_plan.append(row)

        action_predict = np.array(self.action_plan.popleft())
       
        pos, euler, open_close = action_predict[:3], action_predict[3:-1], action_predict[-1]
        open_close = float(open_close) 
        
        if open_close <= 0.5:
            gripper_state = np.ones(2) * 0.04
        else:
            gripper_state = np.zeros(2)

        pos = np.array(pos) + np.array([0, -0.4, 0.78])  # transform from world cordinates to robot cordinates
        euler = np.array(euler)
        return pos, euler, gripper_state
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-track', nargs='+', default=["track_1_in_distribution"], type=str, choices=["track_1_in_distribution", "track_2_cross_category", "track_3_common_sense", "track_4_semantic_instruction"], help="The evaluation track to run")
    parser.add_argument('--n-episode', default=10, type=int, help="The number of episodes to evaluate for a task")
    parser.add_argument('--visulization', action="store_true", default=True, help="Whether to save the visualized episodes")
    parser.add_argument('--metrics', nargs='+', default=["success_rate"], choices=["success_rate", "intention_score", "progress_score"], help="The metrics to evaluate")
    
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    parser.add_argument("--eval_log_dir", default='results/test', type=str, help="Where to log the evaluation results.")
    args = parser.parse_args()
    return args

def evaluate(args):
    kwargs = vars(args)
    episode_config = None
    
    for eval_track in args.eval_track:
        save_dir = os.path.join(args.eval_log_dir, eval_track)
        with open(os.path.join("./VLABench/VLABench", "configs/evaluation/tracks", f"{eval_track}.json"), "r") as f:
            episode_config = json.load(f)
            tasks = list(episode_config.keys())

        assert isinstance(tasks, list)

        evaluator = Evaluator(
            tasks=tasks,
            n_episodes=args.n_episode,
            episode_config=episode_config,
            max_substeps=10,   
            save_dir=save_dir,
            visulization=args.visulization,
            metrics=args.metrics
        )

        policy = ClientModel(host=kwargs['host'], port=kwargs['port'])

        result = evaluator.evaluate(policy)
        
        # average score
        totals = {
            "success_rate": 0.0,
            "intention_score": 0.0,
            "progress_score": 0.0
        }
        count = len(result)
        for item in result.values():
            for key in totals:
                totals[key] += item.get(key, 0.0)

        averages = {key: total / count for key, total in totals.items()}

        print("average:")
        for key, avg in averages.items():
            print(f"{key}: {avg:.4f}")
        
        # save
        result["averages"] = averages
        with open(os.path.join(save_dir, "evaluation_result.json"), "w") as f:
            json.dump(result, f)

if __name__ == "__main__":
    args = get_args()
    evaluate(args)