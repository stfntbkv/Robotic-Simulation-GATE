from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.openvla import OpenVLA
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.tasks import *
from VLABench.robots import *

demo_tasks = ["select_fruit"]
unseen = True
save_dir = "VLABench/downloads/"

model_ckpt = "/remote-home1/pjliu/openvla-7b"
lora_ckpt = "/remote-home1/pjliu/openvla/weights/select_fruit+CSv1+lora/"

import os
os.environ["MUJOCO_GL"] = "egl"

evaluator = Evaluator(
    tasks=demo_tasks,
    n_episodes=2,
    max_substeps=10,   
    save_dir=save_dir,
    visulization=True
)

random_policy = RandomPolicy(model=None)
result = evaluator.evaluate(random_policy)