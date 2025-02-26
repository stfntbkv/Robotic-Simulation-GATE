import argparse
import json
import os
from VLABench.evaluation.evaluator import VLMEvaluator
from VLABench.evaluation.model.vlm import *

# Dynamically instantiate a VLMmodel class
def initialize_model(model_name, *args, **kwargs):
    # Use globals() to check if the specified class exists in the current namespace
    cls = globals().get(model_name)
    if cls is None:
        raise ValueError(f"Model '{model_name}' not found in the current namespace.")
    
    # Return an instance of the VLMmodel class
    return cls(*args, **kwargs)

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run VLM benchmark with specified model and parameters.")
    parser.add_argument("--vlm_name", type=str, required=True, help="Name of the model class to instantiate")
    parser.add_argument("--save_interval", type=int, default=1, help="Interval for saving benchmark results")
    parser.add_argument("--few_shot_num", type=int, default=1, help="Number of few-shot examples")
    parser.add_argument("--with_CoT", type=int, default=0, help="Whether to use CoT")
    parser.add_argument("--task_list_json", type=str, default=None, help="Path to the task list JSON file (optional)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save benchmark results")
    return parser.parse_args()

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Instantiate VLMEvaluator
    vlm_evaluator = VLMEvaluator(tasks = [], n_episodes = 0, save_path = args.save_path)
    
    # Iinstantiate the VLMmodel
    vlm = initialize_model(args.vlm_name)
    
    # Read the task list
    if args.task_list_json is not None:
        try:
            pwd = os.getcwd()
            task_list_path = os.path.join(pwd, "./task_list", args.task_list_json)
            with open(task_list_path, 'r') as f:
                task_list = json.load(f)['task_list']
        except:
            task_list = None
    else:
        task_list = None

    # Evaluate the VLMmodel using the VLMEvaluator
    vlm_evaluator.evaluate(
        vlm,
        task_list=task_list,
        save_interval=args.save_interval,
        few_shot_num=args.few_shot_num,
        with_CoT=False if args.with_CoT == 0 else True
    )

if __name__ == "__main__":
    main()
