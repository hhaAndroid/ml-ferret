from ferret.model.builder import load_pretrained_model
from ferret.conversation import (conv_templates, SeparatorStyle)
from ferret.serve.gradio_web_server import format_region_prompt, resize_bbox
from PIL import Image
import torch
from ferret.mm_utils import tokenizer_image_token
from ferret.constants import IMAGE_TOKEN_INDEX

# 虽然图片输入实际上是 336x336，但是你要假装告诉模型，实际上图片是 1000x1000 的
VOCAB_IMAGE_W = 1000  # 224
VOCAB_IMAGE_H = 1000  # 224
DEFAULT_REGION_REFER_TOKEN = "[region]"
DEFAULT_REGION_FEA_TOKEN = "<region_fea>"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    '/nvme/data/huanghaian/code/ml-ferret/ferret-7b-v1-3', None, 'ferret', False, False)
print(model.dtype)
image_path = 'both.png'
images = Image.open(image_path)
raw_w = images.width
raw_h = images.height
print(raw_w, raw_h)  # 224,224

region00 = [232, 8, 808, 660]  # 1000x1000 尺度
region0 = resize_bbox(region00, raw_w, raw_h)  # 还原到 224x224 尺度

coor_mask = torch.zeros((raw_h, raw_w))
coor_mask[region0[0]:region0[2] + 1, region0[1]:region0[3] + 1] = 1
coor_mask = coor_mask.tolist()

state = conv_templates['ferret_v1'].copy()  # 得到输入的模板
state.append_message(state.roles[0], ('what is this [region0]', images, 'Raw+Processor'))
state.append_message(state.roles[1], None)

refer_input_state = {'region_placeholder_tokens': [], 'region_masks': [], 'region_coordinates': []}
cur_region_token = DEFAULT_REGION_REFER_TOKEN.split(']')[0] + str(0) + ']'  # [region0]
refer_input_state['region_placeholder_tokens'].append(cur_region_token)
# 此时的坐标需要是 1000x1000 尺度
cur_region_coordinates = f'[{int(region00[0])}, {int(region00[1])}, {int(region00[2])}, {int(region00[3])}]'
refer_input_state['region_coordinates'].append(cur_region_coordinates)
# mask 是原图尺度的
refer_input_state['region_masks'].append(coor_mask) # 224x224 的 二维列表

prompt = state.get_prompt()
prompt = format_region_prompt(prompt, refer_input_state)
print(prompt)
# A chat between a human and an AI that understands visuals. In images, [x, y] denotes points: top-left [0, 0],
# bottom-right [width-1, height-1]. Increasing x moves right; y moves down.
# Bounding box: [x1, y1, x2, y2]. Image size: 1000x1000. Follow instructions.
# USER: <image>\nwhat is this [232, 8, 808, 660] <region_fea> ASSISTANT:

pload = {
    "model": 'ferret_v1',
    "prompt": prompt,
    "temperature": float(0.2),
    "top_p": float(0.7),
    "max_new_tokens": min(int(512), 1536),
    "stop": state.sep if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT] else state.sep2, # '</s>'
    "images": image_path,
}

pload['region_masks'] = refer_input_state['region_masks_in_prompts']
# print(pload)
print('final prompt', prompt)

# 直接 resize 为正方形
images = image_processor(images, return_tensors='pt', do_resize=True, do_center_crop=False, size=[336, 336])[
    'pixel_values']
images = images.to(model.device, dtype=torch.float16) # 1 3 336 336
image_args = {"images": images}

region_masks = [[torch.Tensor(region_mask_i).cuda().half() for region_mask_i in pload['region_masks']]]
image_args["region_masks"] = region_masks # list[list[224x224]] 第一个 list 是 batch，第二个 list 是表示区域数

temperature = float(pload.get("temperature", 1.0))
top_p = float(pload.get("top_p", 1.0))
max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
max_new_tokens = min(int(pload.get("max_new_tokens", 256)), 1024)
stop_str = pload.get("stop", None)

stop_idx = None  # 遇到 </s> 停止
if stop_str is not None:
    stop_idx = tokenizer(stop_str).input_ids
    if len(stop_idx) == 1:
        stop_idx = stop_idx[0]
    else:
        stop_idx = None # True

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
output_ids = list(input_ids)
pred_ids = []

max_src_len = context_len - max_new_tokens - 8  # 不能超过长度
input_ids = input_ids[-max_src_len:]

past_key_values = None
for i in range(max_new_tokens):  # 一个一个 token 进行预测
    if i == 0:
        with torch.no_grad():
            out = model(
                torch.as_tensor([input_ids]).cuda(),
                use_cache=True,
                **image_args) # 开始一次推理，image/region_masks
        logits = out.logits
        past_key_values = out.past_key_values
    else:
        attention_mask = torch.ones(
            1, past_key_values[0][0].shape[-2] + 1, device="cuda")
        with torch.no_grad():
            out = model(input_ids=torch.as_tensor([[token]], device="cuda"),
                        use_cache=True,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        region_masks=region_masks)
        logits = out.logits
        past_key_values = out.past_key_values

    # 自己写后处理
    last_token_logits = logits[0][-1] # 取最后一个预测的 logits
    if temperature < 1e-4:
        token = int(torch.argmax(last_token_logits))
    else:
        probs = torch.softmax(last_token_logits / temperature, dim=-1)
        token = int(torch.multinomial(probs, num_samples=1))

    output_ids.append(token)
    pred_ids.append(token)

    if stop_idx is not None and token == stop_idx:
        stopped = True  # 停止
    elif token == tokenizer.eos_token_id:
        stopped = True
    else:
        stopped = False

    if i % 1 == 0 or i == max_new_tokens - 1 or stopped:
        cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
        pos = cur_out.rfind(stop_str)
        if pos != -1:
            cur_out = cur_out[:pos]
            stopped = True
        output = pload["prompt"] + cur_out
        print(output)

    if stopped:
        break

if past_key_values is not None:
    del past_key_values

