from math import sqrt
from animatediff.src.utils import (
    isinstance_str,
    clean_memory, save_video,
)
import torch
import torch.nn.functional as F
from einops import rearrange

import numpy as np
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import numpy as np

index = [
    [35, 934, 625, 1041, 934, 934, 788, 789, 263, 789, 934, 934, 934, 35, 934, 420, 934, 934, 934, 934, 934, 934, 934, 736],
    [323, 199, 199, 199, 199, 721, 199, 199, 789, 323, 41, 721, 273, 1270, 721, 1270, 1270, 1270, 153, 844, 230, 844, 323, 116],
    [449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449, 449],
]

def channel_pca(feature, frame_index):
    fea = feature[frame_index]
    fea = fea.transpose(1, 2, 0) # c,w,h -> w,h,c
    fea = fea.reshape((-1, fea.shape[-1])) # w*h, c

    dim = min(fea.shape[0], fea.shape[1])
    # print(fea.shape)
    pca = PCA(n_components=2)
    pca.fit(fea)

    w = pca.components_
    # print(w.shape)
    w = w[0]
    # print(w)
    # print('max of pc1: ', np.max(w))
    # print('channel index: ', np.where(w==np.max(w)))

    # num = np.zeros(20, dtype=int)
    # for i in range(w.shape[0]):
    #     num[int(abs(float(w[i])) // 0.05 + 1)] += 1

    # plt.hist(np.abs(w), bins=20, color='blue', alpha=0.7)

    # # 设置图表标题和坐标轴标签
    # plt.title(f'hist {w.shape[0]}')
    # plt.xlabel('weight')
    # plt.ylabel('num')

    # # 显示图表
    # plt.savefig(f'./hist_{str(frame_index).zfill(2)}.png')
    # plt.close()
    # print(num)
    w = np.abs(w)
    index = np.argsort(-w)
    
    # return int(np.where(w==np.max(w))[0])
    # print(index[ :int(dim*0.05)])
    return index[ :int(dim*0.05)]

def pca(orig, target, k, step):
    orig = orig.permute(1, 2, 0)    # c,w,h -> w,h,c
    orig = orig.reshape((-1, orig.shape[-1]))  # w*h, c
    U, S, V = torch.pca_lowrank(orig, k, center=True)    
    # print(orig.shape)

    target = target.permute(1, 2, 0)    # c,w,h -> w,h,c
    target = target.reshape((-1, target.shape[-1]))  # w*h, c

    # x = []
    # y = []
    # su = 0
    # for i in range(S.shape[0]):
    #     x.append(i)
    #     su += float(S[i].detach().cpu().numpy())
    #     y.append(su)
    # plt.figure(figsize=(10, 10))  
    # plt.scatter(x,  # 横坐标
    #             y,  # 纵坐标
    #             c='red',  # 点的颜色
    #             label='function')  # 标签 即为点代表的意思

    # plt.legend()
    # plt.savefig(f'./pca_eigen_increase_single_frame_{step}.png')
    # plt.close()

    orig_fea = orig @ V
    target_fea = target @ V

    return orig_fea, target_fea

def pca_all(fea, k):
    fea = fea.permute(0, 2, 3, 1)    # f,c,w,h -> f,w,h,c
    fea = fea.reshape((-1, fea.shape[-1]))  # f*w*h, c

    U, S, V = torch.pca_lowrank(fea, k, center=True)


    x = []
    y = []
    su = 0
    for i in range(S.shape[0]):
        x.append(i)
        su += float(S[i].detach().cpu().numpy())
        y.append(su)
    plt.figure(figsize=(10, 10))  
    plt.scatter(x,  # 横坐标
                y,  # 纵坐标
                c='red',  # 点的颜色
                label='function')  # 标签 即为点代表的意思

    plt.legend()
    plt.savefig(f'./pca_eigen_increase.png')
    plt.close()


    res = fea @ V

    return res

def pca_visualize(orig_all, target_all, k, layer, step):
    orig_res = []
    target_res = []
    features_diff_loss = 0
    for f in range(orig_all.shape[0]):
        orig = orig_all[f]
        target = target_all[f]

        orig = orig.permute(1, 2, 0)    # c,w,h -> w,h,c
        orig = orig.reshape((-1, orig.shape[-1]))  # w*h, c
        U, S, V = torch.pca_lowrank(orig, k, center=True)
        orig_pca = orig @ V
        orig_res.append(orig_pca)

        target = target.permute(1, 2, 0)    # c,w,h -> w,h,c
        target = target.reshape((-1, target.shape[-1]))  # w*h, c
        # U, S, V = torch.pca_lowrank(target, k, center=False)
        target_pca = target @ V
        target_res.append(target_pca)

        features_diff_loss += F.mse_loss(target_pca, orig_pca.detach(), reduction='mean')
    print(orig_res[0].shape)
    if k == 2:
        plt.figure(figsize=(10, 10))  

        plt.scatter(orig_res[0][:, 0].detach().cpu().numpy(),  # 横坐标
                    orig_res[0][:, 1].detach().cpu().numpy(),  # 纵坐标
                    c='red',  # 点的颜色
                    label='function')  # 标签 即为点代表的意思

        plt.scatter(target_res[0][:, 0].detach().cpu().numpy(),  # 横坐标
                    target_res[0][:, 1].detach().cpu().numpy(),  # 纵坐标
                    c='blue',  # 点的颜色
                    label='function')  # 标签 即为点代表的意思

        plt.legend()
        plt.savefig(f'./pca_step_{str(step).zfill(2)}_layer_{str(layer).zfill(2)}.png')
        plt.close()

    return features_diff_loss

def get_pc_direction(fea, k):

    fea = fea.permute(1, 2, 0)    # c,w,h -> w,h,c 
    fea = fea.reshape((-1, fea.shape[-1]))  # w*h, c
    U, S, V = torch.pca_lowrank(fea, k, center=True)
    # print(V.permute(1, 0).shape)
    return V.permute(1, 0)

def get_pc_direction_all(fea, k):
    fea = fea.permute(0, 2, 3, 1)    # f,c,w,h -> f,w,h,c
    fea = fea.reshape((-1, fea.shape[-1]))  # f*w*h, c
    U, S, V = torch.pca_lowrank(fea, k, center=True)
    print(V.permute(1, 0).shape)
    return V.permute(1, 0)

@torch.autocast(device_type="cuda", dtype=torch.float32)
def calculate_losses(orig_features, target_features, config, layer, step):
    orig = orig_features
    target = target_features

    orig = orig.detach()

    total_loss = 0
    losses = {}
    if config["features_loss_weight"] > 0:
        if config["global_averaging"]:
            orig = orig.mean(dim=(2, 3), keepdim=True)
            target = target.mean(dim=(2, 3), keepdim=True)

        features_loss = compute_feature_loss(orig, target)
        total_loss += config["features_loss_weight"] * features_loss
        losses["features_mse_loss"] = features_loss

    if config["features_diff_loss_weight"] > 0:
        features_diff_loss = 0
        print("orig", orig.shape)
        print("target", target.shape)
        orig = orig.mean(dim=(2, 3), keepdim=True)  # t d 1 1
        target = target.mean(dim=(2, 3), keepdim=True)

        for i in range(len(orig)):
            orig_anchor = orig[i]
            target_anchor = target[i]
            orig_diffs = orig - orig_anchor  # t d 1 1
            target_diffs = target - target_anchor  # t d 1 1
            features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=1).mean()
            
        features_diff_loss /= len(orig)
        print(features_diff_loss)
        total_loss += config["features_diff_loss_weight"] * features_diff_loss
        losses["features_diff_loss"] = features_diff_loss
    
    if config["MOFT_diff_loss_weight"] > 0:
        features_diff_loss = 0

        orig_sim = orig.mean(dim=(0), keepdim=False)  # c w h
        target_sim = target.mean(dim=(0), keepdim=False)

        orig = orig - orig_sim  # t c w h
        target = target - target_sim  # t c w h     content removal

        # orig_diffs = orig.mean(dim=(1), keepdim=True)  # t w h
        # target_diffs = target.mean(dim=(1), keepdim=True)
        # channel_index = channel_pca(target.clone().detach().cpu().numpy(), 0)

        orig_select = []
        target_select = []
        channel_index = None
        for f in range(orig.shape[0]):
            channel_index = channel_pca(orig.cpu().numpy(), f)
            # orig_select.append(channel_index)
            # channel_index = index[layer][f]
            orig_ = [orig[f, idx, :, :].unsqueeze(0) for idx in channel_index]
            orig_ = torch.cat(orig_, 0)
            orig_select.append(orig_.unsqueeze(0))
            target_ = [target[f, idx, :, :].unsqueeze(0) for idx in channel_index]
            target_ = torch.cat(target_, 0)
            target_select.append(target_.unsqueeze(0))

        # orig_diffs = orig[:, channel_index, :, :]
        # target_diffs = target[:, channel_index, :, :]   # channel filter

        orig_diffs = torch.cat(orig_select, 0)
        target_diffs = torch.cat(target_select, 0) # channel filter
        print(orig_diffs.shape)

        # features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=1).mean()
        # features_diff_loss += F.mse_loss(target_diffs, orig_diffs.detach(), reduction='mean')
        for f in range(orig.shape[0]):
            sum_ = 0
            for c in range(target_diffs.shape[1]):
                sum_ += F.l1_loss(target_diffs[f, c, :, :], orig_diffs[f, c, :, :].detach(), reduction='mean')
            sum_ /= target_diffs.shape[1]
            features_diff_loss += sum_
        features_diff_loss /= orig.shape[0]
        print(features_diff_loss)

        # features_diff_loss /= len(orig)

        total_loss += config["MOFT_diff_loss_weight"] * features_diff_loss
        losses["features_diff_loss"] = features_diff_loss

    if config["MOFT_PCA_loss_weight"] > 0:
        features_diff_loss = 0

        orig_sim = orig.mean(dim=(0), keepdim=False)  # c w h
        target_sim = target.mean(dim=(0), keepdim=False)

        orig = orig - orig_sim  # t c w h
        target = target - target_sim  # t c w h     content removal

        # features_diff_loss = pca_visualize(orig, target, 2, layer, step)

        for f in range(orig.shape[0]):
            orig_fea, target_fea = pca(orig[f], target[f], 1, step)
            # features_diff_loss += 1 - F.cosine_similarity(target_fea, orig_fea.detach(), dim=1).mean()
            features_diff_loss += F.l1_loss(target_fea, orig_fea.detach(), reduction='mean')
        features_diff_loss /= len(orig)
        print(features_diff_loss)

        # features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=1).mean()

        # orig_fea = pca_all(orig, 200)
        # target_fea = pca_all(target, 200)
        # features_diff_loss += F.mse_loss(target_fea, orig_fea.detach(), reduction='mean')


        total_loss += config["MOFT_PCA_loss_weight"] * features_diff_loss #/ 128.0
        losses["features_diff_loss"] = features_diff_loss

    losses["total_loss"] = total_loss
    return losses


def compute_feature_loss(orig, target):
    features_loss = 0
    for i, (orig_frame, target_frame) in enumerate(zip(orig, target)):
        features_loss += 1 - F.cosine_similarity(target_frame, orig_frame.detach(), dim=0).mean()
    features_loss /= len(orig)
    return features_loss


def get_timesteps(scheduler, num_inference_steps, max_guidance_timestep, min_guidance_timestep):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * max_guidance_timestep), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = int(num_inference_steps * min_guidance_timestep)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]
    if t_end > 0:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order : -t_end * scheduler.order]
    else:
        guidance_schedule = scheduler.timesteps[t_start * scheduler.order :]
    return timesteps, guidance_schedule


def register_time(model, t):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["TransformerWithGuidance", "ModuleWithGuidance", "UpsampleWithGuidance"]):
            setattr(module, "t", t)


def register_batch(model, b):
    for _, module in model.unet.named_modules():
        if isinstance_str(module, ["TransformerWithGuidance", "ModuleWithGuidance", "UpsampleWithGuidance"]):
            setattr(module, "b", b)


def register_guidance(model):
    guidance_schedule = model.guidance_schedule
    num_frames = model.video.shape[0]
    h = model.video.shape[2]
    w = model.video.shape[3]

    class ModuleWithGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.is_cross_attention = module.is_cross_attention
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "temporal_attention",
            ]
            self.module_type = module_type

            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config

        def forward(self, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = self.module(x, *args, **kwargs)
            t = self.num_frames
            if self.module_type == "temporal_attention":
                size = out.shape[1]

            if self.t in self.guidance_schedule:
                h, w = int(sqrt(size * self.h / self.w)), int(sqrt(size * self.h / self.w) * self.w / self.h)
                self.saved_features = rearrange(
                    out, f"(b f) (h w) d -> b f d h w", b=self.b, f=self.num_frames, h=h, w=w
                )
                # print(self.saved_features.shape)
            return out
    class TransformerWithGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "temporal_transformer",
            ]
            self.module_type = module_type

            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config

        def forward(self, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = self.module(x, *args, **kwargs)
            t = self.num_frames
            if self.module_type == "temporal_transformer":
                size = out.shape[1]

            if self.t in self.guidance_schedule:
                self.saved_features = rearrange(
                    out, f"b c f h w -> b f c h w"
                )
                # print(self.saved_features.shape)
            return out

    class UpsampleWithGuidance(torch.nn.Module):
        def __init__(self, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            self.module = module
            self.guidance_schedule = guidance_schedule
            self.num_frames = num_frames
            assert module_type in [
                "temporal_upsample",
            ]
            self.module_type = module_type

            self.h = h
            self.w = w
            self.block_name = block_name
            self.config = config

        def forward(self, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = self.module(x, *args, **kwargs)
            print("!!!!upsample: ", out.shape)
            t = self.num_frames
            if self.module_type == "temporal_upsample":
                size = out.shape[1]

            if self.t in self.guidance_schedule:
                h, w = out.shape[3], out.shape[4]
                self.saved_features = rearrange(
                    out, f"b d f h w -> b f d h w", b=self.b, f=self.num_frames, h=h, w=w
                )
                # print(self.saved_features.shape)
            return out

    up_res_dict = model.config["up_res_dict"]
    for res in up_res_dict:
        module = model.unet.up_blocks[res]
        for block in up_res_dict[res]:
            block_name = f"decoder_res{res}_block{block}"
            if model.config["use_temporal_attention_features"]:
                submodule = module.motion_modules[block].temporal_transformer.transformer_blocks[0]
                # print("!!!!!!!!!temp_attentions:", len(module.motion_modules))
                assert isinstance_str(submodule, "TemporalTransformerBlock")
                block_name_temp = f"{block_name}_temporal_attn1"
                submodule.attention_blocks[0] = ModuleWithGuidance(
                    submodule.attention_blocks[0],
                    guidance_schedule,
                    num_frames,
                    h=h,
                    w=w,
                    block_name=block_name_temp,
                    config=model.config,
                    module_type="temporal_attention",
                )
            if model.config["use_temporal_transformer_features"]:  
                print("!!!!!!")
                submodule = module.motion_modules[block].temporal_transformer
                assert isinstance_str(submodule, "TemporalTransformer3DModel")
                block_name_temp = f"{block_name}_temporal_transformer"
                module.motion_modules[block].temporal_transformer = TransformerWithGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h=h,
                    w=w,
                    block_name=block_name_temp,
                    config=model.config,
                    module_type="temporal_transformer",
                )
                print("module registed!!!")
            if model.config["use_temporal_attention_up_sample_features"]:
                submodule = module.upsamplers[0]
                print("!!!!!!!!!upsamle_attentions:", len(module.upsamplers))
                assert isinstance_str(submodule, "Upsample3D")
                block_name_temp = f"{block_name}_upsample"
                module.upsamplers[0] = UpsampleWithGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h=h,
                    w=w,
                    block_name=block_name_temp,
                    config=model.config,
                    module_type="temporal_upsample",
                )
                print("module registed!!!")
