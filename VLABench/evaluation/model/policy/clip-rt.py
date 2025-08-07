# Set MuJoCo environment variables BEFORE any imports
import os
os.environ["MUJOCO_GL"] = "osmesa"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
# os.environ["DISPLAY"] = ":99"

# Start virtual display for headless operation
# import subprocess
# import time
# try:
#     subprocess.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"], 
#                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     time.sleep(2)  # Give Xvfb time to start
# except:
#     pass

import torch
import torch.nn.functional as F
import numpy as np
import collections
from typing import Dict, Optional
from PIL import Image
import open_clip
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler

# Set device for model inference
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

class ClipRTPolicy:
    def __init__(
        self, 
        model_path: str,
        device: torch.device = device,
        replan_steps: int = 8,
        chunk_cut: int = 8,
        model_name: str = 'ViT-H-14-378-quickgelu',
        #pretrained: str = "openai"
    ) -> None:
        """
        CLIP-RT Policy implementation with action chunking.
        
        Args:
            model_path: Path to the pretrained CLIP-RT checkpoint
            device: Device to run inference on
            replan_steps: Number of steps to execute before replanning
            chunk_cut: Number of action chunks to predict at once
            model_name: CLIP model architecture name
            pretrained: Pretrained weights source
        """
        self.device = device
        self.replan_steps = replan_steps
        self.chunk_cut = chunk_cut
        self.name = "clip-rt"
        self.control_mode = 'ee'
        self.timestep = 0
        
        # Action chunking configuration
        self.num_action_chunk = 8
        self.num_action_dim = 7
        self.action_seq_len = self.num_action_chunk * self.num_action_dim
        
        # Initialize action buffer
        self.action_plan = collections.deque(maxlen=replan_steps)
        
        # Load model and preprocessor
        self._load_model(model_path, model_name) # removed pretrained argument
        
        # CLIP-RT prompt template
        self.prompt_template = "what motion should the robot arm perform to complete the instruction '{}'?"
        
    def _load_model(self, model_path: str, model_name: str):
        """Load CLIP-RT model and preprocessing pipeline."""
        try:
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}")
                print("Attempting to load model without pretrained weights...")
                
                # Try to load without pretrained weights
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name = model_name,
                    device=self.device,
                    use_action_decoder=True
                )
            else:
                # Load CLIP model with action decoder
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name = model_name,
                    pretrained= model_path,
                    device=self.device,
                    use_action_decoder=True
                )
            
            # Load tokenizer
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            self.model.eval()
            self.model = self.model.to(self.device)
            print(f"Successfully loaded CLIP-RT model")
            
        except Exception as e:
            print(f"Error loading CLIP-RT model: {e}")
            print("This might be due to missing model files or incompatible model architecture.")
            print("Please check if the model path is correct and the model files exist.")
            raise e
    
    def predict(self, observation, **kwargs):
        """
        Predict robot actions using CLIP-RT with action chunking.
        
        Args:
            observation: Dictionary containing:
                - rgb: Tuple of (right, left, front, wrist) camera images
                - ee_state: End-effector state [pos(3), quat(4), gripper(1)]
                - instruction: Natural language instruction
        
        Returns:
            Tuple of (target_pos, target_euler, gripper_state)
        """
        # Replan when action buffer is empty or at replan intervals
        if self.timestep % self.replan_steps == 0 or len(self.action_plan) == 0:
            # Extract observations
            right, left, front, image_wrist = observation["rgb"]
            state = observation["ee_state"]
            instruction = observation["instruction"]
            
            # Process end-effector state
            pos, quat, gripper_state = state[:3], state[3:7], state[-1]
            ee_euler = quaternion_to_euler(quat)
            
            # Coordinate transformation (adjust based on your robot setup)
            pos_transformed = pos - np.array([0, -0.4, 0.78])
            state_vec = np.concatenate([pos_transformed, ee_euler, np.array([gripper_state])])
            
            # Get action chunks from model
            action_chunk = self._get_action_chunk(front,  instruction)

            print(f"Generated action chunk: {action_chunk}")
            
            # Add chunks to action plan (limit to replan_steps)
            self.action_plan.extend(action_chunk[:self.replan_steps])
        
        self.timestep += 1
        
        # Pop and execute next action
        if len(self.action_plan) > 0:
            raw_action = self.action_plan.popleft()
        else:
            # Fallback: return current state
            state = observation["ee_state"]
            pos, quat, gripper_state = state[:3], state[3:7], state[-1]
            return pos, quaternion_to_euler(quat), np.array([gripper_state, gripper_state])
        
        # Parse action components
        target_pos, target_euler, gripper = raw_action[:3], raw_action[3:6], raw_action[-1]
        
        # Process gripper command
        if gripper >= 0.1:
            gripper_state = np.ones(2) * 0.04  # Open gripper
        else:
            gripper_state = np.zeros(2)  # Close gripper
        
        # Transform position back to robot coordinates
        target_pos = target_pos.copy()
        target_pos += np.array([0, -0.4, 0.78])
        
        return target_pos, target_euler, gripper_state

    def _get_action_chunk(self, image: np.ndarray, instruction: str):
        """
        Generate action chunk using CLIP-RT model.

        Args:
            image: Front camera image 
            instruction: Natural language instruction
            state: Robot state vector
            
        Returns:
            List of action arrays
        """
        try:
            with torch.no_grad():
                # Prepare image - use front camera as primary
                if isinstance(image, np.ndarray):
                    pil_image = Image.fromarray(image.astype(np.uint8))
                else:
                    pil_image = image
                
                pil_image = pil_image.convert("RGB")
                
                # Preprocess image
                image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                
                # Format instruction with prompt template
                formatted_instruction = self.prompt_template.format(instruction.lower())
                
                # Tokenize instruction
                text_tokens = self.tokenizer([formatted_instruction]).to(self.device)
                
                # Generate action chunks using autocast for efficiency
                with torch.amp.autocast("cuda"):
                    # Encode image and text features
                    image_features = self.model.encode_image(image_tensor, normalize=True)
                    text_features = self.model.encode_text(text_tokens, normalize=True)
                    
                    # Create dummy tokens for action sequence
                    dummy_tokens = torch.full(
                        (image_features.shape[0], self.action_seq_len), 
                        self.model.pad_id
                    ).to(device=image_features.device)
                    
                    # Decode action features
                    out_features = self.model.decode_action(dummy_tokens, image_features, text_features)
                    
                    # Reshape and predict actions
                    batch_size = out_features.shape[0]
                    out_features = out_features[:, 2:, :]  # Remove special tokens [batch, 56, embed_dim]
                    out_features = out_features.reshape(batch_size, self.num_action_chunk, -1)
                    
                    # Generate action predictions
                    action_tensor = self.model.action_head(out_features)  # [batch, 8, 7]
                    
                    # Convert to numpy and extract action chunks
                    actions = action_tensor.squeeze(0).cpu().numpy()  # [8, 7]
                    
                    # Convert to list of individual actions
                    action_chunks = [actions[i] for i in range(min(self.chunk_cut, len(actions)))]
                    
                    return action_chunks
                    
        except Exception as e:
            print(f"Error generating action chunks: {e}")
            # Return fallback action (stay in place)
            fallback_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return [fallback_action] * self.replan_steps
    
    def reset(self) -> None:
        """Reset the policy state."""
        self.timestep = 0
        self.action_plan = collections.deque(maxlen=self.replan_steps)
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "name": self.name,
            "device": self.device,
            "replan_steps": self.replan_steps,
            "chunk_cut": self.chunk_cut,
            "num_action_chunk": self.num_action_chunk,
            "num_action_dim": self.num_action_dim,
            "control_mode": self.control_mode
        }

    
if __name__ == "__main__":
    # Additional environment setup for headless operation
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    
    import numpy as np
    from PIL import Image
    from VLABench.evaluation.evaluator import Evaluator
    from VLABench.tasks import *
    from VLABench.robots import *
    
    demo_tasks = ["select_fruit"]
    unseen = True
    save_dir = "VLABench/downloads/"

    policy = ClipRTPolicy(
        model_path = "VLABench/downloads/clip-rt-finetuned-libero-goal/models--clip-rt--clip-rt-finetuned-libero-goal/snapshots/62c1c50b5897922455606d3a237e8329df78b2e5/cliprt_libero_goal.pt",
        replan_steps=4,
        chunk_cut=4,
    )
    
    # Initialize the evaluator
    evaluator = Evaluator(
        tasks=demo_tasks,
        n_episodes=1,
        max_substeps=10,   
        save_dir=save_dir,
        visulization=True  # Enable visualization for headless mode
    )
    
    result = evaluator.evaluate(policy)