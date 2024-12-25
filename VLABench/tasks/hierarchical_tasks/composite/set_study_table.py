import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.hierarchical_tasks.primitive.select_book_series import SelectSpecificTypeBookConfigManager, SelectBookTask


@register.add_config_manager("set_study_table")
class SelectBookConfigManager(SelectSpecificTypeBookConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[3, 4],
                 **kwargs):
        """select the book with specific name"""
        super().__init__(task_name, num_objects, **kwargs)
        
    def load_objects(self, target_entity):
        super().load_objects(target_entity)
        init_container_config = self.config["task"]["components"][-1]
        for i, subentity_config in enumerate(init_container_config["subentities"]):
            subentity_config["position"][1] -= 0.1
        laptop_config = self.get_entity_config("laptop", position=[random.uniform(-0.2, -0.1), random.uniform(-0.2, -0.15), 0.8])
        self.config["task"]["components"].append(laptop_config) 
    
    def get_instruction(self, target_entity, **kwargs):
        instruction = [f"Today I want to review the content about ... Please help me to manage the study table before I come."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, init_container, **kwargs):
        condition_config = dict(
            not_contain=dict(
                container=init_container,
                entities=[target_entity]
            ),
            joint_in_range=dict(
                target_pos_range=[1.2, 1.5],
                entities=["laptop"]
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_task("set_study_table")
class SetStudyTableTask(SelectBookTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)