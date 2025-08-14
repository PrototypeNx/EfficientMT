# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional
import math
import imageio
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from animatediff.models.motion_module import TemporalTransformerBlock
from animatediff.models.motion_module import MyVersatileAttention, PositionalEncoding

from .attention import BasicTransformerBlock


class ScalePredictor(nn.Module):
    def __init__(self, query_dim, input_channel, video_length):
        super().__init__()

        self.pos_encoder = PositionalEncoding(
            query_dim*2,
            dropout=0., 
            max_len=32
        )
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()

        input_channel *= 2
        self.spatial_convs = nn.ModuleList([
                            nn.Conv2d(input_channel, input_channel//8, kernel_size=3, padding=1),
                            nn.Conv2d(input_channel//8, input_channel//32, kernel_size=3, padding=1),
                            self.zero_init(nn.Conv2d(input_channel//32, 1, kernel_size=3, padding=1))
                        ])
        
        # self.temporal_convs = nn.ModuleList([
        #                     nn.Conv1d(input_channel, input_channel//8, kernel_size=video_length+1, padding=8),
        #                     nn.Conv1d(input_channel//8, input_channel//32, kernel_size=video_length+1, padding=8),
        #                     nn.Conv1d(input_channel//32, 1, kernel_size=video_length+1, padding=8)
        #                 ])

        self.layer_num = len(self.spatial_convs)
    
    def zero_init(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module

    def forward(self, x, video_length):
        f = video_length # * 2
        d_size = x.shape[2]
        w = int(math.sqrt(d_size))
        h = d_size // w

        hidden_states = rearrange(x, "b f d c -> (b d) f c", f=f)
        hidden_states = self.pos_encoder(hidden_states)
        hidden_states = rearrange(hidden_states, "(b w h) f c -> (b f) c w h", w=w, h=h)
        
        layer = 0
        # for spatial_layer, temporal_layer in zip(self.spatial_convs, self.temporal_convs):
        #     hidden_states = spatial_layer(hidden_states)
        #     hidden_states = self.gelu(hidden_states)
        #     hidden_states = rearrange(hidden_states, "(b f) c w h -> (b w h) c f", f=f)
        #     hidden_states = temporal_layer(hidden_states)
        #     layer += 1
        #     if layer == self.layer_num:
        #         break
        #     hidden_states = self.silu(hidden_states)
        #     hidden_states = rearrange(hidden_states, "(b w h) c f -> (b f) c w h", w=w, h=h, f=f)
        # hidden_states = rearrange(hidden_states, "(b w h) c f -> b f (w h) c", w=w, h=h, f=f)

        for spatial_layer in self.spatial_convs:
            hidden_states = spatial_layer(hidden_states)
            layer += 1
            if layer == self.layer_num:
                break
            hidden_states = self.gelu(hidden_states)
        hidden_states = rearrange(hidden_states, "(b f) c w h -> b f (w h) c", w=w, h=h, f=f)

        output = torch.sigmoid(hidden_states - 5)
        # print(output.shape)
        
        return output #- 0.5


class ScalePredictor_channel_1d(nn.Module):
    def __init__(self, query_dim, input_channel, video_length):
        super().__init__()

        self.pos_encoder = PositionalEncoding(
            query_dim*2,
            dropout=0., 
            max_len=32
        )
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
    
        self.input_c = query_dim
        # input_channel *= 2
        # self.spatial_convs = nn.ModuleList([
        #                     nn.Conv2d(video_length, video_length, kernel_size=3, stride=2, padding=1),
        #                     nn.Conv2d(video_length, video_length, kernel_size=3, stride=2, padding=1),
        #                     self.zero_init(nn.Conv2d(video_length, video_length, kernel_size=3, stride=2, padding=1))
        #                 ])
        self.temporal_convs = nn.ModuleList([
                            nn.Conv1d(input_channel, input_channel//4, kernel_size=video_length-1, padding=7),
                            nn.Conv1d(input_channel//4, input_channel//16, kernel_size=video_length-1, padding=7),
                            nn.Conv1d(input_channel//16, 1, kernel_size=video_length-1, padding=7)
                        ])

        self.layer_num = len(self.temporal_convs)
    
    def zero_init(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module

    def forward(self, x, video_length):
        f = video_length # * 2
        d_size = x.shape[2]
        w = int(math.sqrt(d_size))
        h = d_size // w
        batch = x.shape[0]

        hidden_states = rearrange(x, "b f d c -> (b d) f c", f=f)
        hidden_states = self.pos_encoder(hidden_states)
        hidden_states = rearrange(hidden_states, "(b w h) f c -> (b c) (w h) f", w=w, h=h)

        layer = 0

        for temporal_layer in self.temporal_convs:
            hidden_states = temporal_layer(hidden_states)
            layer += 1
            if layer == self.layer_num:
                break
            hidden_states = self.gelu(hidden_states)
        # hidden_states = rearrange(hidden_states, "(b f) c w h -> b f (w h) c", w=w, h=h, f=f)
        hidden_states = rearrange(hidden_states, "(b c) d f -> b f d c", b=batch, f=f) # b f 1 1 c
        

        output = torch.sigmoid(hidden_states - 5)
        # print(output.shape)
        
        return output[:, :, :, self.input_c:] 


class ScalePredictor_channel(nn.Module):
    def __init__(self, query_dim, input_channel, video_length):
        super().__init__()

        self.pos_encoder = PositionalEncoding(
            query_dim,
            dropout=0., 
            max_len=32
        )
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
    
        self.input_c = input_channel
        input_channel
        self.spatial_convs = nn.ModuleList([
                            nn.Conv2d(video_length, video_length, kernel_size=3, stride=2, padding=1),
                            nn.Conv2d(video_length, video_length, kernel_size=3, stride=2, padding=1),
                            nn.Conv2d(video_length, video_length, kernel_size=3, stride=2, padding=1),
                            # self.zero_init(nn.Conv2d(video_length, video_length, kernel_size=3, stride=2, padding=1))
                        ])

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        # self.temporal_convs = nn.ModuleList([
        #                     nn.Conv1d(input_channel, input_channel//8, kernel_size=video_length+1, padding=8),
        #                     nn.Conv1d(input_channel//8, input_channel//32, kernel_size=video_length+1, padding=8),
        #                     nn.Conv1d(input_channel//32, 1, kernel_size=video_length+1, padding=8)
        #                 ])

        self.layer_num = len(self.spatial_convs)
    
    def zero_init(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module

    def forward(self, x, video_length):
        f = video_length # * 2
        d_size = x.shape[2]
        w = int(math.sqrt(d_size))
        h = d_size // w
        batch = x.shape[0]

        hidden_states = rearrange(x, "b f d c -> (b d) f c", f=f)
        hidden_states = self.pos_encoder(hidden_states)
        hidden_states = rearrange(hidden_states, "(b w h) f c -> (b c) f w h", w=w, h=h)

        layer = 0

        for spatial_layer in self.spatial_convs:
            hidden_states = spatial_layer(hidden_states)
            layer += 1
            if layer == self.layer_num:
                break
            hidden_states = self.gelu(hidden_states)
        hidden_states = self.gap(hidden_states) # (b c) f 1 1
        # hidden_states = rearrange(hidden_states, "(b f) c w h -> b f (w h) c", w=w, h=h, f=f)
        hidden_states = rearrange(hidden_states, "(b c) f w h -> b f (w h) c", b=batch, f=f) # b f 1 1 c
        

        output = torch.sigmoid(hidden_states)
        # print(output.shape)
        
        return output #- 0.5
    


class ScalePredictor_temp(nn.Module):
    def __init__(self, query_dim, input_channel, video_length):
        super().__init__()

        self.pos_encoder = PositionalEncoding(
            query_dim*2,
            dropout=0., 
            max_len=32
        )
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()

        input_channel *= 2
        self.spatial_convs = nn.ModuleList([
                            nn.Conv2d(input_channel, input_channel//8, kernel_size=3, padding=1),
                            nn.Conv2d(input_channel//32, input_channel//128, kernel_size=3, padding=1),
                            # nn.Conv2d(input_channel//128, input_channel//256, kernel_size=3, padding=1)
                        ])
        
        self.temporal_convs = nn.ModuleList([
                            nn.Conv1d(input_channel//8, input_channel//32, kernel_size=video_length-1, padding=7),
                            self.zero_init(nn.Conv1d(input_channel//128, 1, kernel_size=video_length-1, padding=7)),
                            # self.zero_init(nn.Conv1d(input_channel//256, 1, kernel_size=video_length-1, padding=7))
                        ])

        self.layer_num = len(self.spatial_convs)
    
    def zero_init(self, module):
        for p in module.parameters():
            p.detach().zero_()
        return module

    def forward(self, x, video_length):
        f = video_length # * 2
        d_size = x.shape[2]
        w = int(math.sqrt(d_size))
        h = d_size // w

        hidden_states = rearrange(x, "b f d c -> (b d) f c", f=f)
        hidden_states = self.pos_encoder(hidden_states)
        hidden_states = rearrange(hidden_states, "(b w h) f c -> (b f) c w h", w=w, h=h)
        
        layer = 0
        for spatial_layer, temporal_layer in zip(self.spatial_convs, self.temporal_convs):
            hidden_states = spatial_layer(hidden_states)
            hidden_states = self.gelu(hidden_states)
            hidden_states = rearrange(hidden_states, "(b f) c w h -> (b w h) c f", f=f)
            hidden_states = temporal_layer(hidden_states)
            layer += 1
            if layer == self.layer_num:
                break
            hidden_states = self.gelu(hidden_states)
            hidden_states = rearrange(hidden_states, "(b w h) c f -> (b f) c w h", w=w, h=h, f=f)
        hidden_states = rearrange(hidden_states, "(b w h) c f -> b f (w h) c", w=w, h=h, f=f)

        # for spatial_layer in self.spatial_convs:
        #     hidden_states = spatial_layer(hidden_states)
        #     layer += 1
        #     if layer == self.layer_num:
        #         break
        #     hidden_states = self.gelu(hidden_states)
        # hidden_states = rearrange(hidden_states, "(b f) c w h -> b f (w h) c", w=w, h=h, f=f)

        output = torch.sigmoid(hidden_states + 5)
        # print(output.shape)
        
        return output #- 0.5



t = 0

def visulize_scale_map(map, name, dim):
    global t
    print(map.shape)
    d_size = map.shape[2]
    w = int(math.sqrt(d_size))
    h = d_size // w

    map = rearrange(map, "b f (w h) c -> b f w h c", w=w, h=h)
    map = map[1, :, :, :, :]
    map = map.squeeze(-1)
    # map = map[16:, :, :]
    print(map.shape)
    with imageio.get_writer(uri=f"./scaler_map/{t}_{name}_{dim}.mp4", mode='I', fps=8) as writer:
        mx = np.max(map.detach().cpu().numpy())
        mn = np.min(map.detach().cpu().numpy())
        for i in range(map.shape[0]):
            img = map[i, :, :].detach().cpu().numpy()
            print(dim, np.max(img), np.min(img))
            img = (img - np.min(img)) / (np.max(img) - np.min(img))

            img = (img * 255.0).astype(np.uint8)
            img = cv2.resize(img, dsize=(384, 384), interpolation=cv2.INTER_LINEAR)
            # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
            writer.append_data(img)
    t += 1


def classify_blocks(block_list, name):
    is_correct_block = False
    for block in block_list:
        if block in name:
            is_correct_block = True
            break
    return is_correct_block


def torch_dfs(model: torch.nn.Module, name=None):
    result = [model]
    cnt=0
    if name is None:
        name = model.__class__.__name__
    for child in model.children():
        child_name = name + f".{child.__class__.__name__}{cnt}"
        child.regist_name = child_name
        result += torch_dfs(child, child_name)
        cnt+=1
    return result


class ReferenceTemporalAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
        insert_scaler=False,
        insert_ca=False
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full", "up_1", "up_2", "up"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
            insert_scaler=insert_scaler,
            insert_ca=insert_ca
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
        insert_scaler=False,
        insert_ca=False
    ):
        MODE = mode
        INSERT_SCALER = insert_scaler
        INSERT_CA = insert_ca
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            video_length=None,
        ):
            if MODE == "write":
                for attention_block, norm in zip(self.attention_blocks, self.norms):
                    norm_hidden_states = norm(hidden_states)
                    self.bank.append(norm_hidden_states.clone())
                    hidden_states = attention_block(
                        norm_hidden_states,
                        encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                        video_length=video_length,
                    ) + hidden_states
                
                
                hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
                
                output = hidden_states  
                return output
            
            if MODE == "read":
                bank_fea = [
                    rearrange(1.0 * d, "(b f) d c -> b f d c", f=video_length) #(b f) d c
                    for d in self.bank
                ]
                layer = 0
                for attention_block, norm in zip(self.attention_blocks, self.norms):
                    norm_hidden_states = norm(hidden_states)
                    
                    # modify_norm_hidden_states = norm_hidden_states  # not inject motion
                    # print("not inject motion!!")
                    if norm_hidden_states.shape[0] > 16 :
                        do_classifier_free_guidance = False
                    else:
                        do_classifier_free_guidance = False
                    
                    if len(bank_fea) == 0:
                        hidden_states_c = attention_block(
                            norm_hidden_states,
                            video_length=video_length,
                        ) + hidden_states
                    else:
                        ref_fea = bank_fea[layer]
                        norm_hidden_states = rearrange(norm_hidden_states, "(b f) d c -> b f d c", f=video_length) #(b f) d c
                        if INSERT_SCALER:
                            scale_map = self.scaler(torch.cat([norm_hidden_states, ref_fea], dim=-1), video_length)
                            # scale_map = self.scaler(ref_fea, video_length)
                            # ref_fea = scale_map[:, video_length:, :, :] * ref_fea
                            ref_fea = scale_map * ref_fea
                            # for p in self.scaler.spatial_convs[0].parameters():
                            #     print(p)
                            # print("scaler!!!")
                            # module_name = type(attention_block).__name__
                            # visulize_scale_map(scale_map, module_name, self.dim)

                        modify_norm_hidden_states = torch.cat([norm_hidden_states, ref_fea], dim=1) 
                        modify_norm_hidden_states = rearrange(modify_norm_hidden_states, "b f d c -> (b f) d c", f=video_length*2)
                        norm_hidden_states = rearrange(norm_hidden_states, "b f d c -> (b f) d c", f=video_length)
                        hidden_states_c = attention_block(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            video_length=video_length,
                        ) + hidden_states
                    
                    if do_classifier_free_guidance:
                        # print("do cfg")
                        hidden_states_out = hidden_states_c.clone()
                        _uc_mask = uc_mask.clone()
                        
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                ).to(device).bool()
                            )
                        hidden_states_out[_uc_mask] = (
                            attention_block(
                                norm_hidden_states[_uc_mask],
                                # encoder_hidden_states=norm_hidden_states[_uc_mask],
                                video_length=video_length,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_out.clone()
                    else:
                        hidden_states = hidden_states_c

                    layer += 1
                
                hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
                
                output = hidden_states  
                return output
    

        if self.reference_attn:
            if self.fusion_blocks == "up":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up_1":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1")
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up_2":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1") + torch_dfs(self.unet.up_blocks[2], "ModuleList.CrossAttnUpBlock3D2")
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norms[0].normalized_shape[0]
            )

            # print(len(attn_modules))
            spatial_dim = [8*8,8*8,8*8, 16*16,16*16,16*16, 32*32,32*32,32*32, 64*64,64*64,64*64]
            for i, module in enumerate(attn_modules):   ## hook forward
                module._original_inner_forward = module.forward
                if isinstance(module, TemporalTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalTransformerBlock
                    )
                    module.bank = []
                    if MODE == "read" and INSERT_SCALER:
                        module.scaler = ScalePredictor(module.dim, module.dim, 16).cuda()#spatial_dim[i], 16).cuda()
                    if MODE == "read" and INSERT_CA:
                        add_ca_blocks = []
                        for i in range(2):
                            add_ca_blocks.append(
                                MyVersatileAttention(
                                    attention_mode="Temporal",
                                    cross_attention_dim=None,
                                
                                    query_dim=module.dim,
                                    heads=module.num_attention_heads,
                                    dim_head=module.attention_head_dim,
                                    dropout=module.dropout_ratio,
                                    bias=module.attention_bias,
                                    upcast_attention=module.upcast_attention,
                        
                                    cross_frame_attention_mode=module.cross_frame_attention_mode,
                                    temporal_position_encoding=True,
                                    temporal_position_encoding_max_len=32,
                                )
                            )
                        module.add_ca_blocks = nn.ModuleList(add_ca_blocks)

                    
                

    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "up_1":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1")
                    if isinstance(module, TemporalTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up_2":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1") + torch_dfs(self.unet.up_blocks[2], "ModuleList.CrossAttnUpBlock3D2")
                    if isinstance(module, TemporalTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1") + torch_dfs(writer.unet.up_blocks[2], "ModuleList.CrossAttnUpBlock3D2")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks)
                    if isinstance(module, TemporalTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet.up_blocks)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block) + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norms[0].normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norms[0].normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                # print(r.regist_name)
                r.bank = [v.clone().to(dtype) for v in w.bank]
                # w.bank.clear()


    def update_ready_made(self, ref_feat_dict, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "up_1":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up_2":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1") + torch_dfs(self.unet.up_blocks[2], "ModuleList.CrossAttnUpBlock3D2")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norms[0].normalized_shape[0]
            )
            device = next(reader_attn_modules[0].parameters()).device
            for r in reader_attn_modules:
                # print(r.regist_name)
                r.bank = [rearrange(v.clone(), "b f d c -> (b f) d c").to(dtype).to(device) for v in ref_feat_dict[r.regist_name]]
                # w.bank.clear()

    def write_ref_feat(self, output_path, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "up_1":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up_2":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1") + torch_dfs(self.unet.up_blocks[2], "ModuleList.CrossAttnUpBlock3D2")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norms[0].normalized_shape[0]
            )
            write_feat = {}
            for w in reader_attn_modules:
                write_feat[w.regist_name] = [v.clone().to(dtype) for v in w.bank]

            torch.save(write_feat, output_path)


    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "up_1":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up_2":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks[1], "ModuleList.CrossAttnUpBlock3D1") + torch_dfs(self.unet.up_blocks[2], "ModuleList.CrossAttnUpBlock3D2")
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "up":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet.up_blocks)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norms[0].normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()
