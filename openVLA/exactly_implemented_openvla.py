#!/usr/bin/env python3
"""
Exact implementation of OpenVLA LIBERO evaluation approach but with robosuite environment.
Based on: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py
"""

import os
import time
import numpy as np
import torch
import robosuite as suite
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# Constants from OpenVLA repo
ACTION_DIM = 7  # [x, y, z, roll, pitch, yaw, gripper]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# OpenVLA v0.1 system prompt (from their code)
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def create_libero_env():
    """
    Create environment exactly like OpenVLA LIBERO evaluation
    """
    print("🏗️ Creating LIBERO environment (matching OpenVLA setup)")
    
    # Use simplified controller configuration for compatibility
    try:
        # Try newer robosuite API first
        from robosuite.controllers import load_controller_config
        controller_config = load_controller_config(default_controller="OSC_POSE")
        print("✅ Using load_controller_config")
    except ImportError:
        try:
            # Try alternative import
            from robosuite import load_controller_config
            controller_config = load_controller_config(default_controller="OSC_POSE")
            print("✅ Using alternative controller config")
        except ImportError:
            # Fallback to string-based configuration
            controller_config = "OSC_POSE"
            print("✅ Using string-based controller config")
    
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,  # Use the controller config
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],  # OpenVLA LIBERO uses agentview
        camera_heights=128,  # Changed to match more common setup
        camera_widths=128,
        control_freq=20,  # Reduced for stability
        horizon=500,  # Reduced for initial testing
        reward_shaping=True,
    )
    
    print("✅ LIBERO environment created successfully")
    return env

def simple_center_crop(image, crop_scale=0.9):
    """Simple center crop implementation"""
    import torchvision.transforms as transforms
    
    # Convert PIL to tensor
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    
    tensor_image = to_tensor(image)
    
    # Calculate crop size
    h, w = tensor_image.shape[1], tensor_image.shape[2]
    crop_h, crop_w = int(h * crop_scale), int(w * crop_scale)
    
    # Center crop
    center_crop = transforms.CenterCrop((crop_h, crop_w))
    resize = transforms.Resize((h, w))
    
    cropped = center_crop(tensor_image)
    resized = resize(cropped)
    
    return to_pil(resized)

def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """
    Generates an action with the VLA policy.
    Exact implementation from OpenVLA repository.
    """
    # Extract image from observation
    if "agentview_image" in obs:
        image_array = obs["agentview_image"]
    elif "frontview_image" in obs:
        image_array = obs["frontview_image"]
    else:
        raise ValueError("No camera image found in observation")
    
    # Convert to PIL Image
    image_corrected = np.rot90(image_array, 2)  # 180-degree rotation
    image = Image.fromarray(image_corrected)
    image = image.convert("RGB")

    # Center crop (simplified implementation)
    if center_crop:
        image = simple_center_crop(image, crop_scale=0.9)

    # Build VLA prompt (exact from OpenVLA code)
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action

def normalize_gripper_action(action, binarize=True):
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
    Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
    the dataset wrapper.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
    """
    # Just normalize the last action to [-1,+1].
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1.
        action[..., -1] = np.sign(action[..., -1])

    return action


def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action


def get_action(cfg, model, obs, task_label, processor=None):
    """
    Queries the model to get an action.
    Exact implementation from OpenVLA repository.
    """
    if cfg["model_family"] == "openvla":
        action = get_vla_action(
            model, processor, cfg["pretrained_checkpoint"], obs, task_label, cfg["unnorm_key"], center_crop=cfg.get("center_crop", False)
        )
        assert action.shape == (ACTION_DIM,)
        
        # Convert to numpy if tensor
        if torch.is_tensor(action):
            action = action.cpu().numpy()
        
        print(f"Gripper action before normalization: {action[-1]:.4f}")
        # Normalize gripper action from [0,1] to [-1,+1] for simulation
        action = normalize_gripper_action(action, binarize=True)
        
        # Optional: Invert gripper if needed (uncomment if gripper behavior is inverted)
        action = invert_gripper_action(action)
        
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action

def setup_openvla_model():
    """Load OpenVLA model and processor"""
    print("📦 Loading OpenVLA model...")
    
    model_name = "openvla/openvla-7b"
    checkpoint_name = "openvla/openvla-7b-finetuned-libero-goal"
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_name,
        #attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    print(f"✅ OpenVLA model loaded on {DEVICE}")
    return model, processor, model_name

def run_libero_evaluation():
    """
    Run LIBERO evaluation exactly like OpenVLA repository
    """
    print("🚀 Starting OpenVLA LIBERO Evaluation")
    
    # Setup model
    model, processor, base_model_name = setup_openvla_model()
    
    # Configuration (matching OpenVLA LIBERO config)
    cfg = {
        "model_family": "openvla",
        "pretrained_checkpoint": base_model_name,
        "unnorm_key": "libero_goal",
        "center_crop": True  # Set to True if you want center cropping
    }
    
    # Create environment
    env = create_libero_env()
    
    # Task configuration
    task_label = "pick up the object"  # LIBERO task description
    max_steps = 6000
    num_episodes = 1
    
    total_successes = 0
    
    for episode in range(num_episodes):
        print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        obs = env.reset()
        done = False
        t = 0
        episode_reward = 0
        
        try:
            while not done and t < max_steps:
                # Get action from VLA (exact OpenVLA approach)
                action = get_action(cfg, model, obs, task_label, processor)
                
                if torch.is_tensor(action):
                    action = action.cpu().numpy()
                
                # Print action details for every step
                print(f"   Step {t}: Action = {action}")
                print(f"   Step {t}: Position = [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
                print(f"   Step {t}: Rotation = [{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
                print(f"   Step {t}: Gripper = {action[6]:.4f}")
                
                # Execute action in environment (EXACT like OpenVLA)
                obs, reward, done, info = env.step(action.tolist())
                episode_reward += reward
                
                
                if done:
                    print(f"✅ Episode {episode + 1} completed successfully!")
                    total_successes += 1
                    break
                    
                t += 1
                
        except Exception as e:
            print(f"❌ Episode {episode + 1} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    # Results
    success_rate = total_successes / num_episodes
    print(f"\n📊 LIBERO Evaluation Results:")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Successful episodes: {total_successes}")
    print(f"   Success rate: {success_rate:.2%}")
    
    env.close()

def test_single_step():
    """Test a single step to debug the setup"""
    print("🧪 Testing single step")
    
    # Setup
    model, processor, base_model_name = setup_openvla_model()
    env = create_libero_env()
    obs = env.reset()
    
    cfg = {
        "model_family": "openvla",
        "pretrained_checkpoint": base_model_name,
        "unnorm_key": "libero_goal",
        "center_crop": False
    }
    
    # Get single action
    task_label = "push the cube to the right"
    action = get_action(cfg, model, obs, task_label, processor)
    
    if torch.is_tensor(action):
        action = action.cpu().numpy()
    
    print(f"🎯 Single step test:")
    print(f"   Task: {task_label}")
    print(f"   Action: {action}")
    print(f"   Action shape: {action.shape}")
    print(f"   Action magnitude: {np.linalg.norm(action[:6]):.4f}")
    
    # Execute one step
    obs, reward, done, info = env.step(action.tolist())
    print(f"   Reward: {reward:.4f}")
    print(f"   Done: {done}")
    
    env.close()

if __name__ == "__main__":
    # Choose what to run
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_single_step()
    else:
        run_libero_evaluation()