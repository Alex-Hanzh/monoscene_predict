import os

from monoscene.monoscene import MonoScene
import numpy as np
from torchvision import transforms
import torch
from helpers import *
import sys
import csv
import cv2
from monoscene.unet2d import UNet2D
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time


model_path = r'/home/hzh/monoscene_predict/monoscene_kitti.ckpt'
torch.set_grad_enabled(False)
device = "cuda:0"
model = MonoScene.load_from_checkpoint(
    "monoscene_kitti.ckpt",
    dataset="kitti",
    n_classes=20,
    feature=64,
    project_scale=2,
    full_scene_size=(256, 256, 32),
)
img_W, img_H = 1220, 370
model = model.to(device)


def Trans(img):
    img = np.array(img, dtype=np.float32, copy=False) / 255.0
    normalize_rgb = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    img = normalize_rgb(img)
    img = img.to(device)
    return img


def predict(img):
    batch = get_projections(img_W, img_H)
    batch["img"] = Trans(img)
    for k in batch:
        batch[k] = batch[k].unsqueeze(0).cuda()
    # batch = batch.to(device)
    pred = model(batch).squeeze()
    # print(pred.shape)
    pred = majority_pooling(pred, k_size=2)
    batch['fov_mask_2'] = batch['fov_mask_2'].cpu()
    fig = draw(pred, batch['fov_mask_2'])

    return fig


def generate(new_path, path):
    img = cv2.imread(path)
    fig = predict(img)
    camera = dict(
        eye=dict(x=0, y=0, z=5)
    )
    fig.update_layout(scene_camera=camera)
    fig.write_image(new_path)


# data_path = r'G:\datasets\kitti\image'
data_path = r'/data1/hzh/image'
new_data_path = r'/data1/hzh/repro_image'
ground_pic_fnames = os.listdir(data_path)

for pic_fname in tqdm(ground_pic_fnames):
    path = os.path.join(data_path, pic_fname)
    new_path = os.path.join(new_data_path, pic_fname)
    generate(new_path, path)
