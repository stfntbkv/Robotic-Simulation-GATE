import random
import numpy as np
from VLABench.tasks.dm_task import LM4ManipBaseTask, SpatialMixin, SemanticMixin, CommonSenseReasoningMixin
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager

@register.add_config_manager("insert_flower")
class InsertFlowerConfigManager(BenchTaskConfigManager):
    def __init__(self,
                 config,
                 num_objects=[3],
                 **kwargs):
        super().__init__(config, num_objects, **kwargs)
    
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1].update(
            position=[random.uniform(-0.3, 0.3), random.uniform(0.15, 0.25), 0.78]
        )
        
    def load_objects(self, target_entity):
        super().load_objects(target_entity)
        for i in range(self.num_object):
            self.config["task"]["components"][-i-1]["position"] = [-0.3 + i * 0.3 + random.uniform(-0.05, 0.05), 
                                                                   random.uniform(-0.15, -0.05), 0.9]
        
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = f"Insert the {target_entity} into the {target_container}."
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                container=target_container,
                entities=[target_entity]
            )
        )
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("insert_flower_spatial")
class InsertFlowerSpatialConfigManager(InsertFlowerConfigManager):
    pass


@register.add_config_manager("insert_flower_common_sense")
class InsertFlowerCommonSenseConfigManager(InsertFlowerConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = f"Insert the flower with ... into vase."
        self.config["task"]["instructions"] = instruction
        
@register.add_config_manager("insert_bloom_flower")
class InsertBloomFlowerConfigManager(InsertFlowerConfigManager):
    def load_objects(self, target_entity, **kwargs):
        entities = [target_entity]
        entities.extend([f"wilted_flower" for i in range(self.num_object - 1)])
        random.shuffle(entities)
        for i, entity in enumerate(entities):
            wilted_flower_config = self.get_entity_config(entity,
                                                          position=[-0.3 + i * 0.3 + random.uniform(-0.05, 0.05), 
                                                                   random.uniform(-0.15, -0.05), 0.9])
            self.config["task"]["components"].append(wilted_flower_config)

@register.add_config_manager("insert_flower_semantic")
class InsertFlowerFlowerSemanticConfigManager(InsertFlowerConfigManager):
    def get_instruction(self, target_entity, target_container, **kwargs):
        instruction = f"Insert the flower into the vase."
        self.config["task"]["instructions"] = instruction

@register.add_config_manager("replace_wilted_flower")
class ReplaceWiltedFlowerConfigManager(InsertFlowerConfigManager):
    def load_containers(self, target_container):
        super().load_containers(target_container)
        container_config = self.config["task"]["components"][-1]
        wilter_flower_config = self.get_entity_config("wilted_flower", position=[0., 0., 0.05], orientation=[np.pi/2, 0 ,0])
        container_config["subentities"] = [wilter_flower_config]
    
    def get_condition_config(self, target_entity, target_container, **kwargs):
        condition = dict(
            contain=dict(
                container=target_container,
                entities=[target_entity]
            ),
            not_contain=dict(
                container=target_container,
                entities=[f"wilted_flower"]
            )
        )
        self.config["task"]["conditions"] = condition

@register.add_task("insert_flower")
class InsertFlowerTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)
        
    def build_from_config(self, config, eval=False):
        """
        Attach the vase for stable interaction.
        """
        # FIXME delete the re-attachment of the vase for more realistic interaction
        super().build_from_config(config, eval)
        for key, entity in self.entities.items():
            if "vase" in key:
                entity.detach()
                self._arena.attach(entity)

@register.add_task("insert_flower_common_sense")
class InsertFlowerCommonSenseTask(InsertFlowerTask, CommonSenseReasoningMixin):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("insert_flower_semantic")
class InsertFlowerSemanticTask(InsertFlowerTask, SemanticMixin):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("insert_flower_spatial")
class InsertFlowerSpatialTask(InsertFlowerTask, SpatialMixin):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("insert_bloom_flower")
class InsertBloomFlowerTask(InsertFlowerTask, CommonSenseReasoningMixin):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

@register.add_task("replace_wilted_flower")
class ReplaceWiltedFlowerTask(InsertFlowerTask):
    """
    Drop the wilted flower in vase into dustbin and replace it with a blooming flower
    """
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot=robot, **kwargs)

