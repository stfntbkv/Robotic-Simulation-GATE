import random
import numpy as np
from VLABench.utils.register import register
from VLABench.tasks.config_manager import BenchTaskConfigManager
from VLABench.tasks.dm_task import LM4ManipBaseTask

@register.add_config_manager("book_rearrange")
class BookRearrangeConfigManager(BenchTaskConfigManager):
    """
    Rearrange the books on the bookshelf by their publication year.
    TODO: add order by author name and etc.
    """
    def __init__(self, 
                 config,
                 num_objects=[3],
                 **kwargs
                 ):
        super().__init__(config, num_objects, **kwargs)
    
    def get_seen_task_config(self):
        target_indices = random.sample(range(len(self.seen_object)), self.num_object)
        # order indices
        target_indices.sort()
        target_entities = [self.seen_object[i] for i in target_indices]
        target_container = random.choice(self.seen_container)
        return self.get_task_config(target_entities, target_container)
    
    def get_unseen_task_config(self):
        target_indices = random.sample(range(len(self.unseen_object)))
        # order indices
        target_indices.sort()
        target_entities = [self.unseen_object[i] for i in target_indices]
        target_container = random.choice(self.unseen_container)
        return self.get_task_config(target_entities, target_container)
    
    def get_task_config(self, target_entities, target_container, **kwargs):
        self.target_entities = target_entities
        self.target_container, self.target_entity, self.init_container = target_container, None, None
        self.load_containers(target_container)
        self.load_objects(target_entities.copy()) # copy for shuffle the order position
        self.get_instruction()
        self.get_condition_config(target_entities, target_container)
        return self.config
        
    def load_containers(self, target_container):
        super().load_containers(target_container)
        self.config["task"]["components"][-1]["position"] = [random.uniform(-0.1,0.1), 
                                                             random.uniform(0.35, 0.4), 
                                                             0.78]
    
    def load_objects(self, target_entities, **kwargs):
        random.shuffle(target_entities)
        for i, entity in enumerate(target_entities):
            entity_config = self.get_entity_config(entity, 
                                                   position=[-0.2+0.2*i+random.uniform(-0.05, 0.05),
                                                                     random.uniform(-0.1, 0.),
                                                                     0.85],
                                                   orientation=[0, np.pi/2, np.pi/2])
            self.config["task"]["components"].append(entity_config)
    
    def get_instruction(self, **kwargs):
        instruction = ["Rearrange the books in the bookshelf."]
        self.config["task"]["instructions"] = instruction
    
    def get_condition_config(self, target_entities, target_container, **kwargs):
        condition_config = dict(
            contain=dict(
                container=target_container,
                entities=target_entities
            ),
            order=dict(
                entities=target_entities,
                axis=[0]
            )
        )
        self.config["task"]["conditions"] = condition_config
        
@register.add_task("book_rearrange")
class BookRearrangeTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name=task_name, robot=robot, **kwargs)        
    
    def build_from_config(self, config, eval=False):
        for key, entity in self.entities.items():
            if "shelf" in key:
                entity.detach()
                self._arena.attach(entity)
        return super().build_from_config(config, eval)