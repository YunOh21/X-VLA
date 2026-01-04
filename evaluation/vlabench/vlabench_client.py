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
        atten_map = np.array(data["attn_map"])
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
        return action, attn_map
    
    def save_attention_overlay(self, image_rgb, attn_list, step_idx):
        if attn_list is None:
            return

        save_dir = "attention_logs"
        os.makedirs(save_dir, exist_ok=True)
        
        attn = np.array(attn_list)
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
        filename = f"{save_dir}/step_{step_idx:04d}.jpg"
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
            ee_6d = np.array(quat_to_rotate6d(ee_quat))
            ee_pos -= np.array([0, -0.4, 0.78])
            ee_state = np.concatenate([ee_pos, ee_6d, gripper], axis=0)
            proprio = np.concatenate([ee_state, np.zeros_like(ee_state)], axis=0).copy()
            
            target_pos = [-0.00938768644869489, -0.20980599916902634, 0.75] # 버튼 표면보다 아래로 가까이 갈 것

            prefix = ""
            # prefix = (f"The target object is located at {target_pos}. "
            #   + "Move near the target object. ")           
            
            suffix = " Press red button in front of the painting."
            
            language_instruction = prefix + obs['instruction'] + prefix

            query = {
                "proprio": json_numpy.dumps(proprio),
                "language_instruction": language_instruction,
                "image0": json_numpy.dumps(main_view),
                "image1": json_numpy.dumps(front_view),
                "image2": json_numpy.dumps(wrist_view),
                "domain_id": 8,
                "steps": 10,
            }
            
            print(language_instruction)

            action, attn_map = self._post(query)
            
            if attn_list is not None:
                step_count = len(self.action_plan)
                self.save_attention_overlay(main_view, attn_list, step_count)

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
