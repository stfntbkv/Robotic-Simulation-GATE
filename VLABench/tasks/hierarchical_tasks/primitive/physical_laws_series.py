import random
from VLABench.tasks.dm_task import PressButtonTask, LM4ManipBaseTask
from VLABench.tasks.config_manager import PressButtonConfigManager
from VLABench.utils.register import register

@register.add_task("friction_qa")
class FrictionQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("density_qa")
class DensityQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("weight_qa")
class WeightQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("thermal_expansion_qa")
class ThermalExpansionQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("speed_of_sound_qa")
class SoundSpeedQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("magnetism_qa")
class MagnetismQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("reflection_qa")
class ReflectionQATask(PressButtonTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("simple_seesaw_use")
class SimpleSeesawUseTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

@register.add_task("simple_cuestick_use")
class SimpleCueStickUseTask(LM4ManipBaseTask):
    def __init__(self, task_name, robot, **kwargs):
        super().__init__(task_name, robot, **kwargs)

# @register.add_task("simple_pulley_use")