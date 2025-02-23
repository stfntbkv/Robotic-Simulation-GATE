"""
The scripts to launch auto scene load and key-point based trajectory generation.
"""
import numpy as np
import os
import open3d as o3d
import mediapy
import argparse
import traceback
from dm_control import viewer
from tqdm import tqdm
from datetime import datetime
from scipy.spatial.transform import Rotation as R
from VLABench.robots import Franka
from VLABench.tasks import *
from VLABench.utils.data_utils import save_single_data, process_observations
from VLABench.utils.utils import find_key_by_value, get_logger
from VLABench.envs import load_env
from VLABench.utils.skill_lib import SkillLib
from VLABench.configs import name2config

os.environ["MUJOCO_GL"] = "egl"

def get_args():
    parser = argparse.ArgumentParser(description='Generate trajectory for a task')
    parser.add_argument('--task-name', default="select_fruit", type=str, help='task name')
    parser.add_argument('--record-video', default=True, help='record video')
    parser.add_argument('--save-dir', default="/media/shiduo/LENOVO_USB_HDD/dataset/VLABench")
    parser.add_argument('--n-sample', default=1, type=int, help='number of samples to generate')
    parser.add_argument('--start-id', default=0, type=int, help='start index for data storage')
    parser.add_argument('--robot', default="franka", type=str, help='robot name')
    parser.add_argument('--debug', action="store_true", default=False, help='debug mode')
    parser.add_argument('--early-stop', action="store_true", default=False, help='whether use early stop when skill failed to carry out')
    args = parser.parse_args()
    return args


def generate_trajectory(args, index, logger):
    env = load_env(args.task_name, robot=args.robot)
    env.reset()
    
    # load key prior information and task specific variables
    target_entity = env.task.config_manager.target_entity   
    instruction = env.task.get_instruction()
    meta_info = dict(
        target_entity=[target_entity],
        entities=list(env.task.entities.keys()),
        instruction=[instruction],
    )
    
    # register the expert sequence
    skill_seq = env.get_expert_skill_sequence()

    # start auto trajectory generation
    observations, waypoints= [], []
    if skill_seq is not None: # normal case
        for skill in skill_seq:
            obs, waypoint, stage_success, task_success = skill(env)
            if args.debug:
                for o in obs: observations.append(dict(rgb=o["rgb"]))
            else:
                observations.extend(obs)
                waypoints.extend(waypoint)
            if args.early_stop and not stage_success:
                logger.warning(f"{skill} failed, early quit...")
                break
            if task_success:
                break
    else: # TODO: some special tasks should be handled based on the feedback
        raise NotImplementedError("No expert skill sequence found")
    
    task_dir = os.path.join(args.save_dir, args.task_name)
    if args.record_video:
        frames = [] 
        for o in observations:
            frames.append(np.vstack([np.hstack(o["rgb"][:2]), np.hstack(o["rgb"][2:4])]))
        if not os.path.exists(task_dir):
            os.makedirs(task_dir)
        mediapy.write_video(os.path.join(task_dir, f"demo_{index}_success_{task_success}.mp4"), 
                            frames, fps=10) 
    if not task_success:
        logger.warning("Task failed, skip saving data")
        return
    else:
        logger.info("Task success, saving data")
    
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_to_save = process_observations(observations)
    
    robot_position = env.robot.robot_config["position"]
    robot_frame_waypoints = [np.array(waypoint) - np.concatenate([robot_position, np.zeros(5)]) for waypoint in waypoints]
    data_to_save["trajectory"] = robot_frame_waypoints
    data_to_save["entities"] = meta_info["entities"]
    data_to_save["target_entity"] = meta_info["target_entity"]
    data_to_save["episode_config"] = json.dumps(env.save())
    data_to_save["instruction"] =meta_info["instruction"]
    save_single_data(data_to_save, 
                     save_dir=task_dir,
                     filename=f"data_{index}.hdf5",
                     )
    env.close()
    
        
if __name__ == "__main__":
    args = get_args()
    logger = get_logger()
    for i in tqdm(range(args.n_sample)):
        i += args.start_id
        try:
            generate_trajectory(args, i, logger)
        except Exception as e:
            err = traceback.TracebackException.from_exception(e)
            print("".join(err.format()))
            continue
    # ===================================================================================
    # BUG: mujoco render will not work after open3d visulization
    # ===================================================================================