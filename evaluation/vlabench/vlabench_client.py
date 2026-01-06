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
    pos = np.array([float(x) for x in pos_str.split()])
    vals = [float(x) for x in xyaxes_str.split()]
    
    xaxis = np.array(vals[0:3])
    yaxis = np.array(vals[3:6])
    
    # 정규화
    xaxis /= np.linalg.norm(xaxis)
    yaxis /= np.linalg.norm(yaxis)
    zaxis = np.cross(xaxis, yaxis)
    
    # MuJoCo(Right, Up, Back) -> OpenCV(Right, Down, Forward) 좌표계 변환
    R_mujoco = np.vstack([xaxis, yaxis, zaxis]).T
    R_cv_correction = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    R_final = R_mujoco @ R_cv_correction
    
    # 4x4 행렬 구성 (Camera -> World)
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = R_final
    cam_to_world[:3, 3] = pos
    
    # World -> Camera (역행렬)
    return np.linalg.inv(cam_to_world)

def get_extrinsics_matrix(cam_pos, cam_quat):
    """
    카메라의 Pose(World -> Camera)를 4x4 행렬로 변환
    cam_pos: [x, y, z]
    cam_quat: [w, x, y, z] or [x, y, z, w] (Scipy 포맷에 맞게 조정 필요)
    """
    # 1. Camera to World Matrix (카메라가 월드의 어디에 있는지)
    rotation = R.from_quat(cam_quat).as_matrix() # 쿼터니언 순서 주의 (x,y,z,w) or (w,x,y,z)
    
    # 4x4 행렬 구성
    cam_to_world = np.eye(4)
    cam_to_world[:3, :3] = rotation
    cam_to_world[:3, 3] = cam_pos
    
    # 2. World to Camera Matrix (월드 좌표를 카메라 기준으로 가져오기 위해 역행렬)
    world_to_cam = np.linalg.inv(cam_to_world)
    return world_to_cam

# [수정] add_visual_prompt 함수 교체
def add_visual_prompt(image, ee_pos, camera_intrinsics, extrinsic_matrix, prompt_type="blue_dot"):
    """
    extrinsic_matrix: World -> Camera 4x4 행렬
    """
    img = image.copy()
    
    # 이미 계산된 행렬을 사용해서 바로 투영
    # project_3d_to_2d 함수는 기존에 있는 것을 그대로 사용 (단, 인자 순서 주의)
    ee_2d = project_3d_to_2d(ee_pos, camera_intrinsics, extrinsic_matrix)
    
    if ee_2d is None:
        return img
    
    x, y = int(ee_2d[0]), int(ee_2d[1])
    h, w = img.shape[:2]
    
    # 화면 밖 체크
    if not (0 <= x < w and 0 <= y < h):
        return img
    
    if prompt_type == "blue_dot":
        # 잘 보이게 빨간색 테두리 + 파란색 점
        cv2.circle(img, (x, y), radius=7, color=(0, 0, 255), thickness=-1) # Red (BGR)
        cv2.circle(img, (x, y), radius=9, color=(255, 255, 255), thickness=1) # White rim
        
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
    point_cam = world_to_cam_pose @ point_3d_homo
    
    # 3. 좌표계 보정 (MuJoCo/OpenGL -> OpenCV)
    # MuJoCo는 -Z가 전방, +Y가 위쪽 / OpenCV는 +Z가 전방, +Y가 아래쪽
    # 따라서 Y축과 Z축을 뒤집어야 합니다.
    # (카메라 행렬 자체에 이 변환이 포함되어 있지 않다면 수동으로 수행)
    point_cam[1] *= -1 
    point_cam[2] *= -1 

    # Z값(깊이)이 0보다 작으면(카메라 뒤에 있으면) 투영하지 않음
    if point_cam[2] <= 0:
        return None 

    # 4. Camera 좌표 -> Image 픽셀 (Intrinsics 적용)
    uv_homo = intrinsics @ point_cam[:3]
    u = uv_homo[0] / uv_homo[2]
    v = uv_homo[1] / uv_homo[2]
    
    return np.array([u, v])

def get_camera_intrinsics(fov, img_width, img_height):
    """
    Get camera intrinsic matrix from FOV
    
    Args:
        fov: Field of view in degrees
        img_width, img_height: Image dimensions
    
    Returns:
        K: 3x3 intrinsic matrix
    """
    focal_length = (img_width / 2) / np.tan(np.radians(fov / 2))
    
    K = np.array([
        [focal_length, 0, img_width / 2],
        [0, focal_length, img_height / 2],
        [0, 0, 1]
    ])
    
    return K

# ---- Visual Prompting End ----

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
                 control_mode = 'ee',
                 episode_config = None):
        
        self.url = f"http://{host}:{port}/act"
        assert control_mode in ['ee', 'joint']
        self.control_mode = control_mode
        self.name = 'hdp'
        self.episode_config = episode_config
        
        # load camera_config
        try:
            config_path = Path(os.environ["VLABENCH_ROOT"]) / "configs" / "camera" / "camera_config.json"
            with open(config_path, "r") as f:
                self.camera_db = json.load(f)
            print(f"[Client] Camera config loaded successfully.")
        except Exception as e:
            print(f"[Error] Failed to load camera config: {e}")
            self.camera_db = {}
        
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
        
        # try:
        #     resp = requests.post(self.url, json=payload)
        #     resp.raise_for_status()
        #     data = resp.json()
        # except Exception as e:
        #     raise RuntimeError(f"Policy server request failed: {e}") from e

        action = np.array(data["action"])  # shape (T, 10) expected: [pos3, rot6d, grip1]
        attn_map = np.array(data["attn_map"])
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
        return action, attn_map
    
    def save_attention_overlay(self, language_instruction, image_rgb, attn_map):
        if attn_map is None:
            return

        save_dir = "attention_logs"
        os.makedirs(save_dir, exist_ok=True)
        
        attn = np.array(attn_map).flatten()
        print("--------- print attn start ------------")
        print(attn)
        print("--------- print attn end ------------")
        seq_len = len(attn)
        print("seq_len: ", seq_len)
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

        # 이미지 크기에 맞춰 확대 (Bicubic Interpolation)
        h, w = image_rgb.shape[:2]
        attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)

        # 컬러맵 적용 (파란색: 낮음, 빨간색: 높음)
        # 정규화 (0~1)
        attn_norm = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        attn_uint8 = (attn_norm * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(attn_uint8, cv2.COLORMAP_JET)

        # 원본 이미지와 합성
        # 입력이 0~1 Float라면 0~255 Uint8로 변환
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
        
        # RGB -> BGR (OpenCV용)
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # 합성 (원본 60% + 히트맵 40%)
        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

        # 파일 저장
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
        episode_config = self.episode_config

        # print(self.action_plan)
        if not self.action_plan:
            multiview = obs['rgb']  # # np.ndarray with shape (4, 480, 480, 3)
            
            main_view = multiview[0]   # np.ndarray with shape (480, 480, 3)
            front_view = multiview[2]   # np.ndarray with shape (480, 480, 3)
            wrist_view = multiview[-1]   # np.ndarray with shape (480, 480, 3)
            
            # proprio
            proprio = obs['ee_state'] # np.ndarray with shape (1, 8)
            ee_pos, ee_quat, gripper = proprio[:3], proprio[3:7], proprio[7:8]
                        
            # ===== VISUAL PROMPTING 추가 =====
            cam_cfg = self.camera_db.get(obs['instruction'], None)
            
            if cam_cfg:
                # 2. 행렬 계산 (helper 함수 사용)
                extrinsic_matrix = get_matrix_from_mujoco_config(cam_cfg["pos"], cam_cfg["xyaxes"])
                
                # 3. Intrinsics (FOV 값도 json에 있으면 가져다 쓰기)
                fovy = float(cam_cfg.get("fovy", 60))
                K = get_camera_intrinsics(fov=fovy, img_width=480, img_height=480)

                # 4. 점 찍기 (Main View)
                # Wrist View는 카메라가 움직이므로 투영하면 안됨! (원본 유지 or 고정점)
                main_view_prompted = add_visual_prompt(main_view, ee_pos, K, extrinsic_matrix, "blue_dot")
                front_view_prompted = front_view.copy() # 필요하면 front_camera config도 가져와서 똑같이 적용
                wrist_view_prompted = wrist_view.copy() # Wrist는 건드리지 않음
            else:
                # Config 로드 실패 시 원본 그대로 사용
                main_view_prompted = main_view
                front_view_prompted = front_view
                wrist_view_prompted = wrist_view
            
            # Camera intrinsics
            K = get_camera_intrinsics(fov=60, img_width=480, img_height=480)
            
            # Add blue dot to images
            main_view_prompted = add_visual_prompt(main_view, ee_pos, K, "blue_dot")
            front_view_prompted = add_visual_prompt(front_view, ee_pos, K, "blue_dot")
            wrist_view_prompted = add_visual_prompt(wrist_view, ee_pos, K, "blue_dot")
            
            instruction_with_prompt = f"Your body is Franka robot. Blue dot is your end effector. {obs['instruction']}"
            # ===== END VISUAL PROMPTING =====
            
            ee_6d = np.array(quat_to_rotate6d(ee_quat))
            ee_pos -= np.array([0, -0.4, 0.78])
            ee_state = np.concatenate([ee_pos, ee_6d, gripper], axis=0)
            proprio = np.concatenate([ee_state, np.zeros_like(ee_state)], axis=0).copy()
            
            # ==== 전송 전 이미지 저장 ====
            debug_dir = "debug_prompts_logs"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            def save_check_image(name, img_arr):
                # 원본 배열 보존을 위해 copy
                to_save = img_arr.copy()
                
                # 0~1 float라면 0~255 uint8로 변환
                if to_save.dtype != np.uint8:
                    if to_save.max() <= 1.0:
                        to_save = (to_save * 255).astype(np.uint8)
                    else:
                        to_save = to_save.astype(np.uint8)
                
                # RGB -> BGR 변환 (OpenCV 저장을 위해)
                to_save = cv2.cvtColor(to_save, cv2.COLOR_RGB2BGR)
                
                filename = f"{debug_dir}/{timestamp}_{obs['instruction']}_{name}.jpg"
                cv2.imwrite(filename, to_save)
                print(f"[DEBUG] Saved prompt image: {filename}")

            # 변환된 이미지 3장 저장
            save_check_image("main_view", main_view_prompted)
            save_check_image("front_view", front_view_prompted)
            save_check_image("wrist_view", wrist_view_prompted)
            # =================================================================

            query = {
                "proprio": json_numpy.dumps(proprio),
                # language instruction with visual prompt
                "language_instruction": instruction_with_prompt,
                # send images as prompted
                "image0": json_numpy.dumps(main_view_prompted),
                "image1": json_numpy.dumps(front_view_prompted),
                "image2": json_numpy.dumps(wrist_view_prompted),
                "domain_id": 8,
                "steps": 10,
            }

            action, attn_map = self._post(query)
            
            self.save_attention_overlay(obs['instruction'], main_view, attn_map)

            target_eef = action[:, :3]
            target_euler = rotate6D_to_euler(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_euler, target_act], axis=-1)

            # Queue up the plan
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
    # parser.add_argument('--tasks', nargs='+', default=None, help="Specific tasks to run, work when eval-track is None")
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

        policy = ClientModel(host=kwargs['host'], port=kwargs['port'], episode_config=episode_config)

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
