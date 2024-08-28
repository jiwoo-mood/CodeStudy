import os
import random
import datetime
import argparse
import numpy as np

import torch
from diffusers.models import AutoencoderKL

from pipeline_demofusion_sdxl import DemoFusionSDXLPipeline

# args
parser = argparse.ArgumentParser(description='DemoFusion Test')

parser.add_argument('--seed', type=int, default=1, help='SEED')
parser.add_argument('--height', type=int, default=2048, help='target height')
parser.add_argument('--width', type=int, default=2048, help='target width')
parser.add_argument('--stage1_resolution', type=int, default=1024, help='stage 1: pretrained image resolution')
parser.add_argument('--stride', type=float, default=0.5, help='local patch overlap ratio (0~1)')
parser.add_argument('--jump_step', type=int, default=1, help='progressive jump size')
parser.add_argument('--sr_strength', type=float, default=3, help='strength of Skip Residual (0 ~), default=3')
parser.add_argument('--use_DS', action='store_true', help='Dilated Sampling')

args = parser.parse_args()

# Seed setting
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DemoFusionSDXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16, vae=vae)
pipe = pipe.to("cuda")

#Define a prompt
prompt = "Envision a portrait of an elderly woman, her face a canvas of time, framed by a headscarf with muted tones of rust and cream. Her eyes, blue like faded denim. Her attire, simple yet dignified." 
negative_prompt = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"


latent_size = args.stage1_resolution // 8 # divide by vae_scale_factor
stride = int(latent_size * args.stride)

images, time_cost = pipe(prompt, negative_prompt=negative_prompt,
                        height=args.height, width=args.width, view_batch_size=4, stride=stride,
                        num_inference_steps=40, guidance_scale=7.5,
                        cosine_scale_1=args.sr_strength, cosine_scale_2=1, cosine_scale_3=1, sigma=0.8,
                        multi_decoder=True, show_image=False, lowvram=True,
                        stage1_resolution=args.stage1_resolution, jump_step=args.jump_step,
                        use_DS=args.use_DS,
                        )

ds = "DS" if args.use_DS else "No_DS"
path = f"results/{ds}_SR{args.sr_strength}_PU{args.jump_step}/seed{seed}_trg_{args.height}_{args.width}_src_{args.stage1_resolution}"
os.makedirs(path, exist_ok=True)
with open(f"{path}/time.txt", "w") as f:
    for i, (image, time) in enumerate(zip(images, time_cost)):
        image.save(f'{path}/image_' + str(i) + '.png')
        f.write(f"{datetime.datetime.now()}\n")
        f.write(f"Stage{i * args.jump_step}: {str(time)} sec\n")