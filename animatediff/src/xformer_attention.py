import math
from typing import Optional, Callable

import xformers
from omegaconf import OmegaConf
import yaml

import numpy as np
import torch
import torch.fft as fft
from einops import rearrange, repeat


def identify_blocks(block_list, name):
    block_name = None
    for block in block_list:
        if block in name:
            block_name = block
            break
    return block_name


def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block


class MySelfAttnProcessor:
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
        

    def __call__(self, attn, hidden_states, query, key, value, attention_mask):
        self.attn = attn
        self.key = key
        self.query = query
        self.value = value
        self.attention_mask = attention_mask
        self.hidden_state = hidden_states.detach()
        return hidden_states
    
    def record_qkv(self, attn, hidden_states, query, key, value, attention_mask):
        self.key = key
        self.query = query
        self.value = value
        self.hidden_state = hidden_states.detach()
        
    def record_attn_mask(self, attn, hidden_states, query, key, value, attention_mask):
        self.attn = attn
        self.attention_mask = attention_mask
        
class MyFeatureFilter():
    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op
    
    def low_pass(self, feature, lpf):
        fftn_result = fft.fftn(feature, dim=(2,3))
        fftn_result = fft.fftshift(feature, dim=(2,3))
        fftn_result = fftn_result * lpf

        result = torch.abs(fft.ifftn(fft.ifftshift(fftn_result, dim=(2,3)), dim=(2,3)))
        return result

    def high_pass(self, feature, hpf):
        fftn_result = fft.fftn(feature, dim=(2,3))
        fftn_result = fft.fftshift(feature, dim=(2,3))
        fftn_result = fftn_result * hpf

        result = torch.abs(fft.ifftn(fft.ifftshift(fftn_result, dim=(2,3)), dim=(2,3)))
        return result
    
    def only_phase(self, feature):
        fftn_result = fft.fftn(feature, dim=(2,3))
        fre_m = torch.abs(fftn_result)   #幅度谱，求模得到
        fre_p = torch.angle(fftn_result) 
        constant = fre_m.mean()
        fre_only_phase = constant * np.e**(1j*fre_p)

        result = torch.abs(fft.ifftn(fre_only_phase, dim=(2,3)))
        return result

    def __call__(self, output):
        
        w = int(math.sqrt(output.shape[1]))
        h = int(output.shape[1] // w)

        lpf = torch.zeros((w, h)).type_as(output).to(output.device)
        R = (h + w) // 4  # pass R
        for x in range(w):
            for y in range(h):
                if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
                    lpf[y, x] = 1
        hpf = 1-lpf
        feature = rearrange(output, "b (w h) c -> b c w h", h=h)
        # result = self.low_pass(feature, lpf)
        result = self.high_pass(feature, hpf)
        result = rearrange(result, "b c w h -> b (w h) c")

        return result

def prep_unet_attention(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if "Attention" in module_name:
            module.set_processor(MySelfAttnProcessor())
    return unet

def prep_unet_feature_filter(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if "TemporalTransformerBlock" in module_name and "up_blocks.2" in name:
            print(name, module_name)
            module.set_filter(MyFeatureFilter())
    return unet


def get_self_attn_feat(unet, injection_config, config):
    hidden_state_dict = dict()
    query_dict = dict()
    key_dict = dict()
    value_dict = dict()
    
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if "CrossAttention" in module_name and 'attn1' in name and classify_blocks(injection_config.blocks, name=name):
            res = int(math.sqrt(module.processor.hidden_state.shape[1]))
            bs = module.processor.hidden_state.shape[0] # 20 * 16 = 320
            hidden_state_dict[name] = module.processor.hidden_state.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            res = int(math.sqrt(module.processor.query.shape[1]))
            query_dict[name] = module.processor.query.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            key_dict[name] = module.processor.key.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)
            value_dict[name] = module.processor.value.cpu().permute(0, 2, 1).reshape(bs, -1, res, res)

    return hidden_state_dict, query_dict, key_dict, value_dict


def clean_attn_buffer(unet):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention" and 'attn' in name:
            if 'injection_config' in module.processor.__dict__.keys():
                module.processor.injection_config = None
            if 'injection_mask' in module.processor.__dict__.keys():
                module.processor.injection_mask = None
            if 'obj_index' in module.processor.__dict__.keys():
                module.processor.obj_index = None
            if 'pca_weight' in module.processor.__dict__.keys():
                module.processor.pca_weight = None
            if 'pca_weight_changed' in module.processor.__dict__.keys():
                module.processor.pca_weight_changed = None
            if 'pca_info' in module.processor.__dict__.keys():
                module.processor.pca_info = None
            if 'step' in module.processor.__dict__.keys():
                module.processor.step = None
