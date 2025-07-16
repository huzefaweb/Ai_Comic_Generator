import os
import sys# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Apply the patch
import gradio_client.utils as client_utils
from patch_utils import json_schema_to_python_type, _json_schema_to_python_type, get_type, get_desc

# Override the functions with your patched versions
client_utils.json_schema_to_python_type = json_schema_to_python_type
client_utils._json_schema_to_python_type = _json_schema_to_python_type
client_utils.get_type = get_type
# Add the missing get_desc function
if not hasattr(client_utils, 'get_desc'):
    client_utils.get_desc = get_desc

from email.policy import default
from json import encoder
import gradio as gr
import spaces
import numpy as np
import torch
import requests
import random
import pickle
from PIL import Image
from tqdm.auto import tqdm
from datetime import datetime
from gradio_utils import is_torch2_available
if is_torch2_available():
    from gradio_utils import \
        AttnProcessor2_0 as AttnProcessor
    # from gradio_utils import SpatialAttnProcessor2_0
else:
    from gradio_utils import AttnProcessor

import diffusers
from diffusers import StableDiffusionXLPipeline
from pipeline import PhotoMakerStableDiffusionXLPipeline
from diffusers import DDIMScheduler
import torch.nn.functional as F
from gradio_utils import cal_attn_mask_xl
import copy
import os
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from utils import get_comic
from style_template import styles
image_encoder_path = "./data/models/ip_adapter/sdxl_models/image_encoder"
ip_ckpt = "./data/models/ip_adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Japanese Anime"
global models_dict
use_va = True
models_dict = {
#    "Juggernaut": "RunDiffusion/Juggernaut-XL-v8",
#   "RealVision": "SG161222/RealVisXL_V4.0" ,
    "SDXL":"stabilityai/stable-diffusion-xl-base-1.0" ,
#   "Unstable": "stablediffusionapi/sdxl-unstable-diffusers-y"
}
photomaker_path =  hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
MAX_SEED = np.iinfo(np.int32).max
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def set_text_unfinished():
    return gr.update(visible=True, value="<h3>(Not Finished) Generating ···  The intermediate results will be shown.</h3>")
def set_text_finished():
    return gr.update(visible=True, value="<h3>Generation Finished</h3>")
#################################################

class SpatialAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        text_context_len (`int`, defaults to 77):
            The context length of the text features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size = None, cross_attention_dim=None,id_length = 4,device = "cuda",dtype = torch.float16):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.total_length = id_length + 1
        self.id_length = id_length
        self.id_bank = {}

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        # un_cond_hidden_states, cond_hidden_states = hidden_states.chunk(2)
        # un_cond_hidden_states = self.__call2__(attn, un_cond_hidden_states,encoder_hidden_states,attention_mask,temb)
        global total_count,attn_count,cur_step,mask1024,mask4096
        global sa32, sa64
        global write
        global height,width
        global num_steps
        if write:
            # print(f"white:{cur_step}")
            self.id_bank[cur_step] = [hidden_states[:self.id_length], hidden_states[self.id_length:]]
        else:
            encoder_hidden_states = torch.cat((self.id_bank[cur_step][0].to(self.device),hidden_states[:1],self.id_bank[cur_step][1].to(self.device),hidden_states[1:]))
        if cur_step <=1:
            hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        else:   # 256 1024 4096
            random_number = random.random()
            if cur_step <0.4 * num_steps:
                rand_num = 0.3
            else:
                rand_num = 0.1
            # print(f"hidden state shape {hidden_states.shape[1]}")
            if random_number > rand_num:
                # print("mask shape",mask1024.shape,mask4096.shape)
                if not write:
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[mask1024.shape[0] // self.total_length * self.id_length:]
                    else:
                        attention_mask = mask4096[mask4096.shape[0] // self.total_length * self.id_length:]
                else:
                    # print(self.total_length,self.id_length,hidden_states.shape,(height//32) * (width//32))
                    if hidden_states.shape[1] == (height//32) * (width//32):
                        attention_mask = mask1024[:mask1024.shape[0] // self.total_length * self.id_length,:mask1024.shape[0] // self.total_length * self.id_length]
                    else:
                        attention_mask = mask4096[:mask4096.shape[0] // self.total_length * self.id_length,:mask4096.shape[0] // self.total_length * self.id_length]
                   # print(attention_mask.shape)
                # print("before attention",hidden_states.shape,attention_mask.shape,encoder_hidden_states.shape if encoder_hidden_states is not None else "None")
                hidden_states = self.__call1__(attn, hidden_states,encoder_hidden_states,attention_mask,temb)
            else:
                hidden_states = self.__call2__(attn, hidden_states,None,attention_mask,temb)
        attn_count +=1
        if attn_count == total_count:
            attn_count = 0
            cur_step += 1
            mask1024,mask4096 = cal_attn_mask_xl(self.total_length,self.id_length,sa32,sa64,height,width, device=self.device, dtype= self.dtype)

        return hidden_states
    def __call1__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        # print("hidden state shape",hidden_states.shape,self.id_length)
        residual = hidden_states
        # if encoder_hidden_states is not None:
        #     raise Exception("not implement")
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            total_batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(total_batch_size, channel, height * width).transpose(1, 2)
        total_batch_size,nums_token,channel = hidden_states.shape
        img_nums = total_batch_size//2
        hidden_states = hidden_states.view(-1,img_nums,nums_token,channel).reshape(-1,img_nums * nums_token,channel)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,nums_token,channel).reshape(-1,(self.id_length+1) * nums_token,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # print(key.shape,value.shape,query.shape,attention_mask.shape)
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        #print(query.shape,key.shape,value.shape,attention_mask.shape)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(total_batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # if input_ndim == 4:
        #     tile_hidden_states = tile_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # if attn.residual_connection:
        #     tile_hidden_states = tile_hidden_states + residual

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(total_batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        # print(hidden_states.shape)
        return hidden_states
    def __call2__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, channel = (
            hidden_states.shape
        )
        # print(hidden_states.shape)
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states  # B, N, C
        else:
            encoder_hidden_states = encoder_hidden_states.view(-1,self.id_length+1,sequence_length,channel).reshape(-1,(self.id_length+1) * sequence_length,channel)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

def set_attention_processor(unet,id_length,is_ipadapter = False):
    global total_count
    total_count = 0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            if name.startswith("up_blocks") :
                attn_procs[name] = SpatialAttnProcessor2_0(id_length = id_length)
                total_count +=1
            else:    
                attn_procs[name] = AttnProcessor()
        else:
            if is_ipadapter:
                attn_procs[name] = IPAttnProcessor2_0(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1,
                    num_tokens=4,
                ).to(unet.device, dtype=torch.float16)
            else:
                attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))
    print("successsfully load paired self-attention")
    print(f"number of the processor : {total_count}")
#################################################
#################################################
canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
load_js = """
async () => {
const url = "https://huggingface.co/datasets/radames/gradio-components/raw/main/sketch-canvas.js"
fetch(url)
  .then(res => res.text())
  .then(text => {
    const script = document.createElement('script');
    script.type = "module"
    script.src = URL.createObjectURL(new Blob([text], { type: 'application/javascript' }));
    document.head.appendChild(script);
  });
}
"""

get_js_colors = """
async (canvasData) => {
  const canvasEl = document.getElementById("canvas-root");
  return [canvasEl._data]
}
"""

css = '''
#color-bg{display:flex;justify-content: center;align-items: center;}
.color-bg-item{width: 100%; height: 32px}
#main_button{width:100%}
<style>
'''


#################################################
title = r"""
<h1 align="center">Ai Comic Generator</h1>
"""

description = r"""
<br>❗️❗️❗️[<b>Important</b>] Personalization steps:<br>
1: Enter the prompt array, each line corrsponds to one generated image.<br>
2: Choose your preferred style template.<br>
3: Click the <b>Submit</b> button to start customizing.
"""

article = r"""
<br>If you have any questions, please feel free to reach me out at <b>huzefa.ahmed.web@gmail.com</b>.
"""
version = r"""
<h3 align="center">Ai Comic Generator</h3>
<h5 >1. Support Typesetting Style and Captioning.(By default, the prompt is used as the caption for each image. If you need to change the caption, add a # at the end of each line. Only the part after the # will be added as a caption to the image.)</h5>
<h5 >2. [NC]symbol (The [NC] symbol is used as a flag to indicate that no characters should be present in the generated scene images. If you want do that, prepend the "[NC]" at the beginning of the line. For example, to generate a scene of falling leaves without any character, write: "[NC] The leaves are falling."),Currently, support is only using Textual Description</h5>
<h5>Tips: Not Ready Now! Just Test! It's better to use prompts to assist in controlling the character's attire. Depending on the limited code integration time, there might be some undiscovered bugs. If you find that a particular generation result is significantly poor, please email me (huzefa.ahmed.web@gmail.com)  Thank you very much.</h4>
"""
#################################################
global attn_count, total_count, id_length, total_length,cur_step, cur_model_type
global write
global  sa32, sa64
global height,width
attn_count = 0
total_count = 0
cur_step = 0
id_length = 4
total_length = 5
cur_model_type = ""
device="cuda"
global attn_procs,unet
attn_procs = {}
###
write = False
###
sa32 = 0.5
sa64 = 0.5
height = 768
width = 768
###
global sd_model_path
sd_model_path = models_dict["SDXL"]#"SG161222/RealVisXL_V4.0"
use_safetensors= False
### LOAD Stable Diffusion Pipeline
# pipe1 = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16, use_safetensors= use_safetensors)
# pipe1 = pipe1.to("cpu")
# pipe1.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
# # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe1.scheduler.set_timesteps(50)
### 
''''pipe2 = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    models_dict["Juggernaut"], torch_dtype=torch.float16, use_safetensors=use_safetensors)
pipe2 = pipe2.to("cpu")
pipe2.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img"  # define the trigger word
)
pipe2 = pipe2.to("cpu")
pipe2.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe2.fuse_lora()'''

pipe4 = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    models_dict["SDXL"], torch_dtype=torch.float32, use_safetensors=True)
pipe4 = pipe4.to("cpu")
pipe4.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img"  # define the trigger word
)
pipe4 = pipe4.to("cpu")
pipe4.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
pipe4.fuse_lora()

# pipe3 = StableDiffusionXLPipeline.from_pretrained("SG161222/RealVisXL_V4.0", torch_dtype=torch.float16)
# pipe3 = pipe3.to("cpu")
# pipe3.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
# # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe3.scheduler.set_timesteps(50)
######### Gradio Fuction #############

def remove_tips():
    return gr.update(visible=False)

def apply_style_positive(style_name: str, positive: str):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive)

def apply_style(style_name: str, positives: list, negative: str = ""):
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return [p.replace("{prompt}", positive) for positive in positives], n + ' ' + negative

def change_visible_by_model_type(_model_type):
    # Since you are **only using text**, always hide ref image uploads
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


######### Image Generation ##############
@spaces.GPU(duration=120)
def process_generation(_sd_type, _num_steps, style_name, guidance_scale, seed_, sa32_, sa64_, id_length_, general_prompt, negative_prompt, prompt_array, G_height, G_width, _comic_type):
    global sa32, sa64, id_length, total_length, attn_procs, unet, cur_model_type, device
    global num_steps
    global write
    global cur_step, attn_count
    global height, width
    height = G_height
    width = G_width
    global pipe2, pipe4
    global sd_model_path, models_dict

    sd_model_path = models_dict[_sd_type]
    num_steps = _num_steps
    use_safe_tensor = True

    if style_name == "(No style)":
        sd_model_path = models_dict["SDXL"]

    pipe = StableDiffusionXLPipeline.from_pretrained(sd_model_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    set_attention_processor(pipe.unet, id_length_, is_ipadapter=False)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    cur_model_type = _sd_type + "-original-" + str(id_length_)
    
    prompts = prompt_array.splitlines()
    if len(prompts) > 10:
        raise gr.Error(f"No more than 10 prompts in the Hugging Face demo for speed! But found {len(prompts)} prompts!")

    generator = torch.Generator(device="cuda").manual_seed(seed_)
    sa32, sa64 = sa32_, sa64_
    id_length = id_length_
    clipped_prompts = prompts[:]

    prompts = [general_prompt + "," + prompt if "[NC]" not in prompt else prompt.replace("[NC]", "") for prompt in clipped_prompts]
    prompts = [prompt.rpartition('#')[0] if "#" in prompt else prompt for prompt in prompts]

    id_prompts = prompts[:id_length]
    real_prompts = prompts[id_length:]

    torch.cuda.empty_cache()
    write = True
    cur_step = 0
    attn_count = 0

    id_prompts, negative_prompt = apply_style(style_name, id_prompts, negative_prompt)

    setup_seed(seed_)

    total_results = []
    
    # Generate ID images
    id_images = pipe(id_prompts, num_inference_steps=num_steps, guidance_scale=guidance_scale,
                     height=height, width=width, negative_prompt=negative_prompt, generator=generator).images
    total_results = id_images + total_results
    yield total_results

    # Generate real comic images
    real_images = []
    write = False
    for real_prompt in real_prompts:
        setup_seed(seed_)
        cur_step = 0
        real_prompt = apply_style_positive(style_name, real_prompt)
        
        real_images.append(pipe(real_prompt, num_inference_steps=num_steps, guidance_scale=guidance_scale,
                                height=height, width=width, negative_prompt=negative_prompt, generator=generator).images[0])
        total_results = [real_images[-1]] + total_results
        yield total_results

    # Comic typesetting if selected
    if _comic_type != "No typesetting (default)":
        captions = prompt_array.splitlines()
        captions = [caption.replace("[NC]", "") for caption in captions]
        captions = [caption.split('#')[-1] if "#" in caption else caption for caption in captions]
        from PIL import ImageFont
        total_results = get_comic(id_images + real_images, _comic_type, captions=captions,
                                  font=ImageFont.truetype("./Inkfree.ttf", int(45))) + total_results

    yield total_results

def array2string(arr):
    return "\n".join(arr)

#################################################
#################################################
### define the interface
with gr.Blocks(css=css) as demo:
    binary_matrixes = gr.State([])
    color_layout = gr.State([])

    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Row():
        with gr.Group(elem_id="main-image"):

            with gr.Column(visible=True):
                sd_type = gr.Dropdown(choices=list(models_dict.keys()), value="SDXL", label="sd_type", info="Select pretrained model")

                general_prompt = gr.Textbox(value='', label="(1) Textual Description for Character", interactive=True)
                negative_prompt = gr.Textbox(value='', label="(2) Negative Prompt", interactive=True)
                style = gr.Dropdown(label="Style Template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)
                prompt_array = gr.Textbox(lines=3, value='', label="(3) Comic Description (each line = one frame)", interactive=True)

                with gr.Accordion("(4) Tune the Hyperparameters", open=True):
                    sa32_ = gr.Slider(label="Paired Attention at 32x32 layers", minimum=0, maximum=1., value=0.7, step=0.1)
                    sa64_ = gr.Slider(label="Paired Attention at 64x64 layers", minimum=0, maximum=1., value=0.7, step=0.1)
                    id_length_ = gr.Slider(label="Number of id images", minimum=2, maximum=4, value=3, step=1)
                    seed_ = gr.Slider(label="Seed", minimum=-1, maximum=MAX_SEED, value=0, step=1)
                    num_steps = gr.Slider(label="Number of Sample Steps", minimum=25, maximum=50, step=1, value=50)
                    G_height = gr.Slider(label="Height", minimum=256, maximum=1024, step=32, value=1024)
                    G_width = gr.Slider(label="Width", minimum=256, maximum=1024, step=32, value=1024)
                    comic_type = gr.Radio(["No Typesetting (default)", "Four Panel", "Classic Comic Style"], value="Classic Comic Style", label="Typesetting Style")
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=10.0, step=0.1, value=5)

                final_run_btn = gr.Button("Generate ! 😺")

        with gr.Column():
            out_image = gr.Gallery(label="Result", columns=2, height='auto')
            generated_information = gr.Markdown(label="Generation Details", value="", visible=False)
            gr.Markdown(version)

    final_run_btn.click(fn=set_text_unfinished, outputs=generated_information
    ).then(
        process_generation,
        inputs=[sd_type, num_steps, style, guidance_scale, seed_, sa32_, sa64_, id_length_, general_prompt, negative_prompt, prompt_array, G_height, G_width, comic_type],
        outputs=out_image
    ).then(fn=set_text_finished, outputs=generated_information)

    gr.Examples(
        examples=[
            [0, 0.5, 0.5, 2, "a young girl with short hair, wearing a jacket and boots",
             "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
             array2string([
                 "exploring an abandoned library at night #This place is full of secrets.",
                 "discovers a hidden stairway beneath a broken bookshelf #Where does this go?",
                 "descends the stairs into a glowing underground room #Is this... magic?",
                 "touches a floating book, causing symbols to light up around her #Something is awakening!"
             ]),
             "Japanese Anime", 768, 768],
            [0, 0.7, 0.7, 2, "a man, wearing black suit",
             "bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, three crus, fused feet, fused thigh, extra crus, ugly fingers, horn, cartoon, cg, 3d, unreal, animate, amputation, disconnected limbs",
             array2string([
                 "at home, read new paper #at home, The newspaper says there is a treasure house in the forest.",
                 "on the road, near the forest",
                 "[NC] The car on the road, near the forest #He drives to the forest in search of treasure.",
                 "[NC]A tiger appeared in the forest, at night",
                 "very frightened, in the forest, at night",
                 "running very fast, in the forest, at night",
                 "[NC] A house in the forest, at night #Suddenly, he discovers the treasure house!",
                 "in the house filled with  treasure, laughing, at night #He is overjoyed inside the house."
             ]),
             "Japanese Anime", 768, 768],
            [0, 0.6, 0.4, 3, "a cyberpunk hacker, glowing wires, neon glasses",
             "bad anatomy, blurred face, extra limbs, poorly drawn, bad proportions, cartoon, fake",
             array2string([
             "[NC]In a dark room filled with monitors #She types rapidly on a neon-lit keyboard",
             "neon city street at night, people walking by",
             "a robot chases her through a back alley",
             "[NC]She jumps onto a rooftop, escaping"
             ]),
            "Comic book", 768, 768],
            [1, 0.7, 0.3, 3, "an astronaut in white spacesuit",
             "bad anatomy, floating limbs, poorly drawn face, disconnected limbs, cartoon",
             array2string([
             "floating in space above Earth",
             "[NC]Spots a mysterious alien ship in the distance",
             "enters the ship cautiously",
             "[NC]finds a message written in glowing symbols"
             ]),
             "Digital/Oil Painting", 768, 768],
        ],
        inputs=[seed_, sa32_, sa64_, id_length_, general_prompt, negative_prompt, prompt_array, style, G_height, G_width],
        label='😺 Examples 😺',
    )

    gr.Markdown(article)

demo.launch()