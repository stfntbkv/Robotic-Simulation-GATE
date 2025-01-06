import random
from VLABench.tasks.dm_task import LM4ManipBaseTask, SpatialMixin, SemanticMixin, CommonSenseReasoningMixin
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.utils.register import register
from VLABench.utils.utils import flatten_list, grid_sample

@register.add_config_manager("select_fruit")
class SelectFruitConfigManager(BenchTaskConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=[3, 4],
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs) 
    
    def load_init_containers(self, init_container):
        pass
    
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = [f"Put the {target_entity} into the {target_container}"]
        self.config["task"]["instructions"] = instruction

    def get_condition_config(self, target_entity, target_container, **kwargs):
        conditions_config = dict(
            contain=dict(
                container=f"{target_container}",
                entities=[f"{target_entity}"]
            )
        )
        self.config["task"]["conditions"] = conditions_config

@register.add_config_manager("select_fruit_common_sense")
class SelectFruitCommonSeneseConfigManager(SelectFruitConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        objects = [conf["name"] for conf in self.config["task"]["components"][2:]]
        instruction = [f"tell some specifics of {target_entity} between {objects}"]
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("select_fruit_spatial")
class SelectFruitSpatialConfigManager(LM4ManipBaseTask):
    def load_objects(self, target_entity):
        # init container add subentities
        self.config["task"]["components"][-1]["subentities"] = []
        in_objects = [target_entity]
        other_objects = flatten_list(self.seen_object) + flatten_list(self.unseen_object)
        other_objects.remove(target_entity)
        in_objects.extend(random.sample(other_objects, self.num_object-2))
        self.target_entity = target_entity if random.random() < 0.3 else f"{target_entity}_inside"
        positions = grid_sample([-0.1, 0.1, -0.1, 0.1, 0, 0], [3, 3], self.num_object - 1)
        for i, (in_object, pos) in enumerate(zip(in_objects, positions)):
            object_config = self.get_entity_config(in_object,
                                                   position=[pos[0] + random.uniform(-0.02, 0.02), 
                                                             pos[1] + random.uniform(-0.02, 0.02), 
                                                             0.1], 
                                                   specific_name=f"{in_object}_inside",)
            self.config["task"]["components"][-1]["subentities"].append(object_config)
        super().load_objects(target_entity)

    def get_instruction(self, target_entity, target_container, init_container, **kwargs):
        if "inside" in self.target_entity:
            instruction = [f"Put the {target_entity} in {init_container} into the {target_container}"]
        else:
            instruction = [f"Put the {target_entity} on the table into the {target_container}"]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_container, **kwargs):
        conditions_config = dict(
            contain=dict(
                container=f"{target_container}",
                entities=[f"{self.target_entity}"]
            )
        )
        self.config["task"]["conditions"] = conditions_config

@register.add_config_manager("select_fruit_semantic")
class SelectFruitSemanticConfigManager(SelectFruitConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        objects = [conf["name"] for conf in self.config["task"]["components"][2:]]
        instruction = [f"tell some specifics of {target_entity} between {objects}"]
        self.config["task"]["instructions"] = instruction
    
@register.add_task("select_fruit")
class SelectFruitTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("select_fruit") 
        super().__init__(task_name, robot=robot, **kwargs)
        
@register.add_task("select_fruit_common_sense")
class SelectFruitCommonSenseTask(SelectFruitTask, CommonSenseReasoningMixin):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("select_fruit_common_sense")
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_fruit_spatial")
class SelectFruitSpatialTask(SelectFruitTask, SpatialMixin):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("select_fruit_spatial")
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("select_fruit_semantic")
class SelectFruitSemanticTask(SelectFruitTask, SemanticMixin):
    def __init__(self, task_name, robot, **kwargs):
        self.config_manager_cls = register.load_config_manager("select_fruit_semantic")
        super().__init__(task_name, robot=robot, **kwargs)
        