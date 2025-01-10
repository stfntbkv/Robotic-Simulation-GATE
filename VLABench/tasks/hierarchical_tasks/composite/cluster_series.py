import numpy as np
import random
from VLABench.tasks.dm_task import ClusterTask
from VLABench.tasks.config_manager import ClusterConfigManager
from VLABench.tasks.hierarchical_tasks.primitive.select_billiards_series import SOLID, STRIPED
from VLABench.utils.register import register
from VLABench.utils.utils import grid_sample

@register.add_config_manager("cluster_book")
class ClusterBookConfigManager(ClusterConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=2,
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)      

    def load_containers(self, target_container, **kwargs):
        if isinstance(target_container, list):
            target_container = random.choice(target_container)
            self.target_container = target_container
        container_config = self.get_entity_config(target_container, 
                                                  position=[random.uniform(-0.1, 0.1), 
                                                            random.uniform(0.3, 0.35), 0.78],
                                                  randomness=None
                                                  )
        self.config["task"]["components"].append(container_config)
    
    def load_objects(self, target_entity, **kwargs):
        super().load_objects(target_entity, **kwargs)
        # change default positions
        index_to_start = - self.num_object * 2
        for i in range(self.num_object * 2):
            position = [-0.2+i*0.13+random.uniform(-0.03, 0.03), 
                     random.uniform(-0.1, -0.), 
                     0.85]
            orientation = [0, np.pi/2, np.pi/2]
            self.config["task"]["components"][i + index_to_start]["position"] = position
            self.config["task"]["components"][i + index_to_start]["orientation"] = orientation
            self.config["task"]["components"][i + index_to_start]["randomness"] = None
             
    def get_condition_config(self, target_entity, target_container, **kwargs):
        assert isinstance(target_entity[-1], list) and isinstance(target_container, list), "target entities and containers should be in list"
        condition_config = dict()
        # or condition, either one of the two conditions is satisfied
        condition_config["or"] = [
        dict(
            contain_1=dict(
                entities=self.entities_to_load["cls_1"],
                container=target_container[-1],
                layer=0,
            ),
            contain_2=dict(
                entities=self.entities_to_load["cls_2"],
                container=target_container[-1],
                layer=1,
                
            )
        ),
        dict(
            contain_1=dict(
                entities=self.entities_to_load["cls_1"],
                container=target_container[-1],
                layer=1,
            ),
            contain_2=dict(
                entities=self.entities_to_load["cls_2"],
                container=target_container[-1],
                layer=0,
            )
        )]        
        
        self.config["task"]["conditions"] = condition_config

@register.add_config_manager("cluster_billiards")
class ClusterBilliardsConfigManager(ClusterConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=2,
                 **kwargs):
        super().__init__(task_name, num_objects, seen_object=[SOLID[::2], STRIPED[::2]], unseen_object=[SOLID[1::2], STRIPED[1::2]], **kwargs)      

    def load_containers(self, target_container):
        super().load_containers(target_container)
        for idx in [1, 2]:
            self.config["task"]["components"][idx]["randomness"].update(dict(
                scale=[0.5, 0.6]
            ))
    
    def load_objects(self, target_entity, **kwargs):
        assert isinstance(target_entity, list) and isinstance(target_entity[0], list), "target entities should be a list"
        self.entities_to_load = dict(
            cls_1 = [], cls_2 = []
        )
        for i, entities in enumerate(target_entity):
            entities_in_same_class = random.sample(entities, self.num_object)   
            self.entities_to_load[f"cls_{i+1}"] = entities_in_same_class
        entities_to_load = []
        for ls in self.entities_to_load.values():
            entities_to_load.extend(ls)
        random.shuffle(entities_to_load)
        positions = grid_sample(workspace=self.config["task"]["workspace"], 
                                grid_size=self.config["task"]["ngrid"],
                                n_samples=self.num_object+self.num_object)
        for i, (entity, pos) in enumerate(zip(entities_to_load, positions)):
            pos = [pos[0], pos[1], 0.8]
            object_config = self.get_entity_config("billiards", 
                                                    position=pos,
                                                    specific_name=entity,
                                                    value=entity
                                                   )
            self.config["task"]["components"].append(object_config)

@register.add_config_manager("cluster_toy")
class ClusterToyConfigManager(ClusterConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=2,
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)      

    def load_objects(self, target_entity, **kwargs):
        super().load_objects(target_entity, **kwargs)
        for index in range(-2*self.num_objects, 0):
            self.config["task"]["components"][index]["orientation"] = random.choice([[np.pi/2, 0, 0], [np.pi/2, 0, np.pi]])
    
    def load_containers(self, target_container):
        super().load_containers(target_container)
        for i, index in enumerate([-2, -1]):
            self.config["task"]["components"][index]["position"] = [(i-0.5)*0.7, random.uniform(-0.1, 0.1), 0.8],
            self.config["task"]["components"][index]["randomness"]["scale"] = [1.2, 1.4]

@register.add_config_manager("cluster_dessert")
class ClusterDessertConfigManager(ClusterConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=2,
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)      

    def load_objects(self, target_entity, **kwargs):
        super().load_objects(target_entity, **kwargs)
        for index in range(-2*self.num_objects, 0):
            self.config["task"]["components"][index]["position"][-1] = 0.85

@register.add_config_manager("cluster_drink")
class ClusterDrinkConfigManager(ClusterConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=2,
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)      

    def load_containers(self, target_container):
        super().load_containers(target_container)
        for index in range(-2, 0):
            self.config["task"]["components"][index]["z_threshold"] = 0.2
    
    def load_objects(self, target_entity, **kwargs):
        super().load_objects(target_entity, **kwargs)
        for index in range(-2*self.num_objects, 0):
            self.config["task"]["components"][index]["position"][-1] = 0.85

@register.add_config_manager("cluster_ingredients")
class ClusterIngredientsConfigManager(ClusterConfigManager):
    def __init__(self, 
                 task_name,
                 num_objects=2,
                 **kwargs):
        super().__init__(task_name, num_objects, **kwargs)      
    
    def load_containers(self, target_container):
        super().load_containers(target_container)  
        for i, index in enumerate([-2, -1]):
            self.config["task"]["components"][index]["position"] = [(i-0.5)*0.7, random.uniform(-0.1, 0.1), 0.8],
            
@register.add_task("cluster_book")
class ClusterBookTask(ClusterTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)
    
    def build_from_config(self, eval=False):
        for key, entity in self.entities.items():
            if "shelf" in key:
                entity.detach()
                self._arena.attach(entity)
        return super().build_from_config(eval)

@register.add_task("cluster_billiards")
class ClusterBilliardsTask(ClusterTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("cluster_toy")
class ClusterToyTask(ClusterTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("cluster_dessert")
class ClusterDessertTask(ClusterTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("cluster_drink")
class ClusterDrinkTask(ClusterTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("cluster_ingredients")
class ClusterIngredientsTask(ClusterTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)