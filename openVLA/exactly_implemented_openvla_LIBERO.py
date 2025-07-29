#!/usr/bin/env python3
"""
Exact implementation of OpenVLA LIBERO evaluation approach with LIBERO dataset
Based on: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py
"""

import os
import time
import numpy as np
import torch
import robosuite as suite
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from torch.serialization import safe_globals
import cv2
from datetime import datetime

# Monkey-patch torch.load for LIBERO initial states (allows numpy globals and forces weights_only=False)
_orig_torch_load = torch.load

def _libero_safe_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    with safe_globals([np.core.multiarray._reconstruct, np.ndarray]):
        return _orig_torch_load(f, *args, **kwargs)

torch.load = _libero_safe_load

# Fix PyTorch 2.6+ security issue for LIBERO initial states loading
# This must be done before importing LIBERO modules
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray])

# Import libero
import sys
sys.path.insert(0,'/home/dimitar/Robotics/ViewImagesLibero/LIBERO')
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark

# Constants from OpenVLA repo
ACTION_DIM = 7  # [x, y, z, roll, pitch, yaw, gripper]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# OpenVLA v0.1 system prompt (from their code)
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

def quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    # Handle the case where the quaternion is [x, y, z, w]
    if len(quat) == 4:
        x, y, z, w = quat
    else:
        raise ValueError("Quaternion must have 4 elements")
    
    # Normalize quaternion
    norm = np.sqrt(x*x + y*y + z*z + w*w)
    if norm > 0:
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # Convert to axis-angle
    angle = 2 * np.arccos(np.abs(w))
    if angle < 1e-6:
        # No rotation
        return np.array([0, 0, 0])
    
    s = np.sqrt(1 - w*w)
    if s < 1e-6:
        # Axis is arbitrary
        return np.array([1, 0, 0]) * angle
    
    axis = np.array([x, y, z]) / s
    return axis * angle

def create_libero_env(task_suite_name="libero_goal", task_id=0, with_renderer=True):
    """
    Create LIBERO environment exactly like OpenVLA LIBERO evaluation
    """
    print(f"🏗️ Creating LIBERO environment: {task_suite_name}, task {task_id}")
    
    # Get benchmark and task
    benchmark_class = get_benchmark(task_suite_name)
    task_suite = benchmark_class()
    task = task_suite.get_task(task_id)
    
    # Get environment and task description exactly like OpenVLA
    env, task_description = get_libero_env(task, "openvla", resolution=256, with_renderer=with_renderer)
    
    # Get initial states (safe globals already set at top of file)
    try:
        initial_states = task_suite.get_task_init_states(task_id)
    except Exception:
        # Fallback: use safe torch.load (monkey patched) directly
        init_path = task_suite._get_init_states_path(task_id)
        initial_states = torch.load(init_path)

    print(f"✅ LIBERO environment created successfully")
    print(f"📝 Task description: {task_description}")
    return env, task_description, initial_states

def get_libero_env(task, model_family, resolution=256, with_renderer=True):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution,
        "has_renderer": with_renderer,  # Enable renderer for visualization
        "has_offscreen_renderer": True,
        "render_camera": "frontview",  # Camera for rendering
        "render_gpu_device_id": 0,
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    # show environment renderer setting
    print("Available attributes:", [attr for attr in dir(env) if 'render' in attr.lower()])
    print("Env attributes:", [attr for attr in dir(env.env) if 'render' in attr.lower()])
    print("Env has renderer:", env.env.has_renderer)
    return env, task_description

def save_video_from_frames(frames, filename, fps=20):
    """Save a list of frames as an MP4 video file."""
    if not frames:
        print("No frames to save")
        return
    
    # Convert PIL images to numpy arrays if needed
    if isinstance(frames[0], Image.Image):
        frames = [np.array(frame) for frame in frames]
    
    # Get video dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Failed to open video writer for {filename}")
        return
    
    # Write frames
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"📹 Video saved: {filename}")

def get_libero_dummy_action(model_family):
    """Get a dummy action for waiting periods."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # No movement, gripper open

def get_horizontal_movement_action(step, max_steps, amplitude=0.1):
    """
    Generate dummy action that moves the robot arm horizontally (x-axis only).
    Creates smooth horizontal movement from -amplitude to +amplitude.
    
    Args:
        step: Current step number
        max_steps: Total number of steps for the movement
        amplitude: Maximum movement range (default 0.1 for safety)
    
    Returns:
        List of 7 action values [x, y, z, roll, pitch, yaw, gripper]
    """
    # Create smooth sinusoidal movement in x-axis
    progress = step 
    x_movement = amplitude * step
    
    # Keep other movements at zero, gripper open
    action = [
        x_movement,  # x-axis movement (-amplitude to +amplitude)
        0.0,         # y-axis (no movement)
        0.0,         # z-axis (no movement)
        0.0,         # roll (no rotation)
        0.0,         # pitch (no rotation)
        0.0,         # yaw (no rotation)
        1.0          # gripper open
    ]
    
    return action

def get_libero_image(obs, resize_size=None):
    """Extract and preprocess image from LIBERO observation."""
    # Get the frontview image
    if "frontview_image" in obs:
        image = obs["frontview_image"]
    elif "agentview_image" in obs:
        image = obs["agentview_image"]
    else:
        raise ValueError("No camera image found in observation")
    
    # Convert to PIL and rotate (LIBERO images need 180-degree rotation)
    image_corrected = np.rot90(image, 2)
    pil_image = Image.fromarray(image_corrected).convert("RGB")
    
    if resize_size is not None:
        pil_image = pil_image.resize((resize_size, resize_size))
    
    return pil_image

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
    # Get preprocessed image
    image = get_libero_image(obs, resize_size=224)  # OpenVLA uses 224x224
    
    # Center crop if specified
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


def get_action(cfg, model, observation, task_label, processor=None):
    """
    Queries the model to get an action.
    Exact implementation from OpenVLA repository.
    """
    if cfg["model_family"] == "openvla":
        # For LIBERO, we need to create obs dict compatible with the VLA function
        obs = {"frontview_image": np.array(observation["full_image"])}
        
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
        
        # Invert gripper for LIBERO environment
        action = invert_gripper_action(action)
        
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action

def setup_openvla_model():
    """Load OpenVLA model and processor"""
    print("📦 Loading OpenVLA model...")
    
    model_name = "openvla/openvla-7b"
    checkpoint_name = "openvla/openvla-7b-finetuned-libero-goal"  # Use goal for this example
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_name,
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)
    
    print(f"✅ OpenVLA model loaded on {DEVICE}")
    return model, processor, model_name

def run_libero_evaluation(use_dummy_actions=False):
    """
    Run LIBERO evaluation exactly like OpenVLA repository
    
    Args:
        use_dummy_actions: If True, use horizontal movement dummy actions instead of OpenVLA model
    """
    print("🚀 Starting OpenVLA LIBERO Evaluation")
    
    if use_dummy_actions:
        print("🤖 Using dummy horizontal movement actions")
        model, processor, base_model_name = None, None, None
        cfg = None
    else:
        print("🧠 Using OpenVLA model")
        # Setup model
        model, processor, base_model_name = setup_openvla_model()
        
        # Configuration (matching OpenVLA LIBERO config)
        cfg = {
            "model_family": "openvla",
            "pretrained_checkpoint": base_model_name,
            "unnorm_key": "libero_goal",  # Match the task suite
            "center_crop": True  # Set to True if you want center cropping
        }
    
    # Create environment with visualization
    env, task_description, initial_states = create_libero_env(
        task_suite_name="libero_goal", 
        task_id=0,
        with_renderer=True  # Enable visualization
    )
    
    # Task configuration
    max_steps = 2000  # For libero_goal
    num_steps_wait = 10  # Wait for objects to stabilize
    num_episodes = 1
    
    total_successes = 0
    
    for episode in range(num_episodes):
        print(f"\n🎯 Episode {episode + 1}/{num_episodes}")
        print(f"📝 Task: {task_description}")
        
        # Reset environment
        env.reset()
        
        # Set initial state (like OpenVLA)
        obs = env.set_init_state(initial_states[episode])
        
        done = False
        t = 0
        episode_reward = 0
        replay_images = []
        video_frames = []  # Store frames for video generation
        
        try:
            while t < max_steps + num_steps_wait:
                # Wait for objects to stabilize (like OpenVLA)
                if t < num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action("openvla"))
                    env.env.render()
                    t += 1
                    continue
                
                # Get preprocessed image
                img = get_libero_image(obs, resize_size=224)
                replay_images.append(img)
                
                # Get action based on mode
                if use_dummy_actions:
                    # Use horizontal movement dummy action
                    action = get_horizontal_movement_action(
                        step=t - num_steps_wait, 
                        max_steps=max_steps, 
                        amplitude=0.05  # Small safe movement
                    )
                    action = np.array(action)
                    print(f"   Step {t-num_steps_wait}: Using dummy horizontal action")
                else:
                    # Prepare observation (like OpenVLA)
                    observation = {
                        "full_image": img,
                        "state": np.concatenate((
                            obs["robot0_eef_pos"], 
                            quat2axisangle(obs["robot0_eef_quat"]), 
                            obs["robot0_gripper_qpos"]
                        )),
                    }
                    
                    # Get action from VLA (exact OpenVLA approach)
                    action = get_action(cfg, model, observation, task_description, processor)
                
                if torch.is_tensor(action):
                    action = action.cpu().numpy()
                
                # Print action details for every step
                print(f"   Step {t-num_steps_wait}: Action = {action}")
                print(f"   Step {t-num_steps_wait}: Position = [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
                print(f"   Step {t-num_steps_wait}: Rotation = [{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
                print(f"   Step {t-num_steps_wait}: Gripper = {action[6]:.4f}")
                
                # Execute action in environment (EXACT like OpenVLA)
                obs, reward, done, info = env.step(action.tolist())
                env.env.render()  # Render after action
                episode_reward += reward
                
                # Capture frame for video (get RGB frame from environment)
                try:
                    # Get frame for video recording
                    if hasattr(env, 'render'):
                        # Try to get RGB array directly
                        frame = env.render(mode='rgb_array')
                        if frame is not None:
                            video_frames.append(frame)
                    else:
                        # Fallback: use the frontview image from observation
                        if "frontview_image" in obs:
                            video_frames.append(obs["frontview_image"])
                        elif "agentview_image" in obs:
                            video_frames.append(obs["agentview_image"])
                except Exception as e:
                    print(f"Failed to capture frame: {e}")
                
                # Render for visualization
                try:
                    env.render()
                except:
                    pass  # In case rendering fails
                
                if done:
                    print(f"✅ Episode {episode + 1} completed successfully!")
                    total_successes += 1
                    break
                    
                t += 1
                
        except Exception as e:
            print(f"❌ Episode {episode + 1} failed with exception: {e}")
            import traceback
            traceback.print_exc()
        
        # Save replay images
        if replay_images:
            print(f"💾 Saving {len(replay_images)} replay images")
        
        # Save video from captured frames
        if video_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"libero_episode_{episode+1}_{timestamp}.mp4"
            save_video_from_frames(video_frames, video_filename, fps=10)
        else:
            print("⚠️ No video frames captured")
    
    # Results
    success_rate = total_successes / num_episodes
    print(f"\n📊 LIBERO Evaluation Results:")
    print(f"   Total episodes: {num_episodes}")
    print(f"   Successful episodes: {total_successes}")
    print(f"   Success rate: {success_rate:.2%}")
    
    env.close()

def test_single_step(use_dummy_actions=False):
    """Test a single step to debug the setup"""
    print("🧪 Testing single step")
    
    if use_dummy_actions:
        print("🤖 Using dummy horizontal movement action")
        model, processor, base_model_name = None, None, None
        cfg = None
    else:
        print("🧠 Using OpenVLA model")
        # Setup
        model, processor, base_model_name = setup_openvla_model()
        cfg = {
            "model_family": "openvla",
            "pretrained_checkpoint": base_model_name,
            "unnorm_key": "libero_goal",
            "center_crop": False
        }
    
    env, task_description, initial_states = create_libero_env(
        task_suite_name="libero_goal", 
        task_id=0,
        with_renderer=True
    )
    
    # Reset and set initial state
    env.reset()
    obs = env.set_init_state(initial_states[0])
    
    # Get preprocessed image
    img = get_libero_image(obs, resize_size=224)
    
    # Get single action based on mode
    if use_dummy_actions:
        # Use horizontal movement dummy action
        action = get_horizontal_movement_action(step=0, max_steps=300, amplitude=0.5)
        action = np.array(action)
        print(f"🤖 Using dummy horizontal action")
    else:
        # Prepare observation
        observation = {
            "full_image": img,
            "state": np.concatenate((
                obs["robot0_eef_pos"], 
                quat2axisangle(obs["robot0_eef_quat"]), 
                obs["robot0_gripper_qpos"]
            )),
        }
        
        # Get single action
        action = get_action(cfg, model, observation, task_description, processor)
    
    if torch.is_tensor(action):
        action = action.cpu().numpy()
    
    print(f"🎯 Single step test:")
    print(f"   Task: {task_description}")
    print(f"   Action: {action}")
    print(f"   Action shape: {action.shape}")
    print(f"   Action magnitude: {np.linalg.norm(action[:6]):.4f}")
    
    # Execute one step
    obs, reward, done, info = env.step(action.tolist())
    print(f"   Reward: {reward:.4f}")
    print(f"   Done: {done}")
    
    # Capture frame for single step video
    try:
        frame = env.render(mode='rgb_array')
        if frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"libero_single_step_{timestamp}.mp4"
            save_video_from_frames([frame], video_filename, fps=1)
    except Exception as e:
        print(f"Failed to capture single step frame: {e}")
    
    # Render
    try:
        env.render()
        print("   Rendering successful!")
    except Exception as e:
        print(f"   Rendering failed: {e}")
    
    env.close()

if __name__ == "__main__":
    # Choose what to run
    import sys
    
    # Parse command line arguments
    use_dummy = False
    test_mode = False
    
    for arg in sys.argv[1:]:
        if arg == "test":
            test_mode = True
        elif arg == "dummy" or arg == "--dummy":
            use_dummy = True
        elif arg == "--help" or arg == "-h":
            print("Usage: python exactly_implemented_openvla_LIBERO.py [test] [dummy]")
            print("  test:  Run single step test instead of full evaluation")
            print("  dummy: Use dummy horizontal movement actions instead of OpenVLA model")
            print("Examples:")
            print("  python exactly_implemented_openvla_LIBERO.py")
            print("  python exactly_implemented_openvla_LIBERO.py dummy")
            print("  python exactly_implemented_openvla_LIBERO.py test")
            print("  python exactly_implemented_openvla_LIBERO.py test dummy")
            sys.exit(0)
    
    if test_mode:
        test_single_step(use_dummy_actions=use_dummy)
    else:
        run_libero_evaluation(use_dummy_actions=use_dummy)