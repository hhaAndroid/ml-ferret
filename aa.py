
from ferret.model import *
import os

model = FERRETLlamaForCausalLM.from_pretrained('ferret-7b-v1-3', low_cpu_mem_usage=True)
vision_tower = model.get_vision_tower()
vision_tower_path = os.path.join('ferret-7b-v1-3', 'vision_tower')
if os.path.exists(vision_tower_path):
    print(f'Start Loading vision tower from {vision_tower_path}')
    vision_tower.load_model(vision_tower_path=vision_tower_path)
    print(f'Finish Loading vision tower from {vision_tower_path}')
else:
    print('=======')
    vision_tower.load_model()
    print('==xxxx===')
