import json
import random
from typing import List

import pickle
from pathlib import Path
import os
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import decord
decord.bridge.set_bridge('torch')
from PIL import Image
from torch.utils.data import Dataset

def video_preprocess(video_path, height, width, video_length, duration=None, sample_start_idx=0,):
    
    video_name = video_path.split('/')[-1].split('.')[0]
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    if  duration is None:
        # 读取整个视频
        total_frames = len(vr)
    else:
        # 根据给定的时长（秒）计算帧数
        total_frames = int(fps * duration)
        total_frames = min(total_frames, len(vr))  # 确保不超过视频总长度
        
    sample_index = np.linspace(0, total_frames - 1, video_length, dtype=int)
    # print(total_frames,sample_index)
    video = vr.get_batch(sample_index)
    video = rearrange(video, "f h w c -> f c h w")

    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=True)
    
    video = video / 127.5 - 1.0

    return video

class VideoDataset(Dataset):
    def __init__(
        self,
        video_length,
        width,
        height,
        data_path="./samples_data",
        ref_feat_path="/data2/zhangzijian/MotionClone"
    ):
        super().__init__()

        self.data_path = data_path
        self.ref_feat_path = ref_feat_path
        self.groups = os.listdir(data_path)
        self.video_length = video_length
        self.width = width
        self.height = height
        self.vid = {}
        self.vid_list = []
        for group in self.groups:
            if group not in self.vid.keys():
                self.vid[group] = []
            group_path = os.path.join(data_path, group)
            for file in os.listdir(group_path):
                self.vid[group].append(os.path.join(group_path, file))
                self.vid_list.append(os.path.join(group_path, file))

    def __getitem__(self, index):
        # print(index)
        ref_vid_path = self.vid_list[index]
        num = 0
        group = None
        for key in self.vid.keys():
            if num + len(self.vid[key]) - 1 >= index:
                group = key
                ref_index = index - num
                break
            num += len(self.vid[key])

        # ref_index = random.randint(0, len(self.vid[group])-1)
        target_index = random.randint(0, len(self.vid[group])-1)
        while target_index == ref_index:
            target_index = random.randint(0, len(self.vid[group])-1)

        # ref_vid_path = self.vid[group][ref_index]
        target_vid_path = self.vid[group][target_index]
        # print(ref_vid_path, target_vid_path)
        
        ref_text = ref_vid_path.split("/")[-1].split(".")[0].replace(group, "").replace("_"," ")

        
        if "MotionInversion" in self.data_path:
            target_text = target_vid_path.split("/")[-1].split(".")[0].replace("_"," ")
        else:
            target_text = target_vid_path.split("/")[-1].split(".")[0].replace(group, "").replace("_"," ")
        target_text = target_text.split(",")
        target_text = target_text[0] + target_text[-1]
        # print(target_text)

        ref_pixels = video_preprocess(ref_vid_path, self.height, self.width, self.video_length)
        target_pixels = video_preprocess(target_vid_path, self.height, self.width, self.video_length)
        
        ref_feat_name = Path(ref_vid_path).name.replace("mp4", "pt")
        ref_feat_dict = torch.load(os.path.join(os.path.join(self.ref_feat_path, group), ref_feat_name), map_location='cpu')
        
        # print(ref_text, target_text)
        sample = dict(
            ref_feat_dict=ref_feat_dict,
            ref_pixels=ref_pixels,
            target_pixels=target_pixels,
            ref_text=ref_text,
            target_text=target_text
        )

        return sample

    def __len__(self):
        num = 0
        for group in self.vid.keys():
            num += len(self.vid[group])
        # return len(self.groups)
        return num