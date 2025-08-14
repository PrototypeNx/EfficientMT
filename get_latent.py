import os
import math
import wandb
import copy
import random
import logging
import inspect
import argparse
import datetime
import subprocess

from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from typing import Dict, Optional, Tuple

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.swa_utils import AveragedModel
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.models import UNet2DConditionModel
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.data.dataset import WebVid10M
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.manual_temp_attn import ReferenceTemporalAttentionControl
from animatediff.pipelines.pipeline_my import MotionPipeline
from animatediff.utils.util import save_videos_grid, zero_rank_print
from animatediff.models.dataset import VideoDataset, video_preprocess



def main(
    ref_video,
    fusion_block,
    output_path,
    ref_prompt,

    pretrained_model_path: str,
    motion_module_path: str,
    dreambooth_model_path: str,
    dataset_config: Dict,

    ref_timestep: int,
    inference_config: Dict = {},

    noise_scheduler_kwargs = None,
    
    enable_xformers_memory_efficient_attention: bool = True,

    global_seed: int = 42,
    local_rank: str = "cuda",
    seed: int = 42,
):

    local_rank = "cuda"
    seed = global_seed
    torch.manual_seed(seed)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(noise_scheduler_kwargs))

    vae          = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer    = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")

    unet = UNet3DConditionModel.from_pretrained_2d(
        pretrained_model_path, subfolder="unet", 
        unet_additional_kwargs=OmegaConf.load(inference_config).unet_additional_kwargs
    )
    
    # load motion module
    unet_state_dict = {}
    print(f"load motion module from {motion_module_path}")
    motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
    motion_module_state_dict = motion_module_state_dict["state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
    unet_state_dict.update({name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
    unet_state_dict.pop("animatediff_config", "")
    
    missing, unexpected = unet.load_state_dict(unet_state_dict, strict=False)
    del unet_state_dict

    # load LoRA
    if dreambooth_model_path.endswith(".safetensors"):
        print(f"load dreambooth model from {dreambooth_model_path}")
        dreambooth_state_dict = {}
        with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                dreambooth_state_dict[key] = f.get_tensor(key)
    elif dreambooth_model_path.endswith(".ckpt"):
        dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")
    # 1. vae
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, vae.config)
    vae.load_state_dict(converted_vae_checkpoint)
    # 2. unet
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, unet.config)
    unet.load_state_dict(converted_unet_checkpoint, strict=False)
    # 3. text_model
    text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
    del dreambooth_state_dict
    
    ########### register temporal attn control  ##############
    insert_scaler = False
    insert_ca = False
    print("insert_scaler: ", insert_scaler)
    print("insert_ca: ", insert_ca)
    reference_control_writer = ReferenceTemporalAttentionControl(
        unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks=fusion_block,
        insert_scaler=insert_scaler,
        insert_ca=insert_ca
    )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Set unet trainable parameters
    unet.requires_grad_(False)

    ref_timestep = torch.tensor(ref_timestep).long()

    # Enable xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Move models to GPU
    vae.to(local_rank)
    text_encoder.to(local_rank)
    unet.to(local_rank)

    unet.eval()

    with torch.no_grad():
        height = 512
        width  = 512 
        video_length = dataset_config.video_length

        out_path = output_path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        
        video_path = ref_video
        print(video_path)
        name = Path(video_path).stem

        video_data = video_preprocess(video_path, height, width, video_length).unsqueeze(0).to(local_rank)

        video_data = rearrange(video_data, "b f c h w -> (b f) c h w")
        ref_vid_latents = vae.encode(video_data).latent_dist
        ref_vid_latents= ref_vid_latents.sample()
        ref_vid_latents = rearrange(ref_vid_latents, "(b f) c h w -> b c f h w", f=video_length)
        ref_vid_latents = ref_vid_latents * 0.18215

        noise_ref = torch.randn_like(ref_vid_latents)
        ref_noisy_latents = noise_scheduler.add_noise(ref_vid_latents, noise_ref, ref_timestep)

        input_prompt_ids = tokenizer(
            [ref_prompt], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids.to(ref_vid_latents.device)
        cond_text_embeddings = text_encoder(input_prompt_ids)[0]
        
        unet(
            ref_noisy_latents,
            ref_timestep,
            encoder_hidden_states=cond_text_embeddings,
            return_dict=False,
        )
        reference_control_writer.write_ref_feat(os.path.join(out_path, f"{name}.pt"))
        reference_control_writer.clear()

        torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str, default="configs/get_latent.yaml")
    parser.add_argument("--reference_video",   type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    main(args.reference_video, **config)
