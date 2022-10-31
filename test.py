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

model_path = r'D:\Work\Code\Python\MonoScene\trained_models\monoscene_kitti.ckpt'
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
# model = model.to(device)

net_rgb = UNet2D.build(out_feature=3, use_decoder=True)


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
    return img


def predict(img):
    batch = get_projections(img_W, img_H)
    batch["img"] = Trans(img)
    for k in batch:
        batch[k] = batch[k].unsqueeze(0)  # .cuda()
    # batch = batch.to(device)
    pred = model(batch).squeeze()
    # print(pred.shape)
    pred = majority_pooling(pred, k_size=2)
    fig = draw(pred, batch['fov_mask_2'])

    return fig


img = cv2.imread(r'C:\Users\Alex\Desktop\0000017.jpg')
# img = cv2.imread(r'C:\Users\Alex\Desktop\0000047.jpg')
# img = Trans(img).unsqueeze(0)
# img_rgb = net_rgb(img)

# scale = [1, 2, 4, 8, 16]

# for i in range(0, 5, 1):
#     #     number = i + 1
#     #     plt.subplot(2, 3, number)
#     out = img_rgb['1_{}'.format(scale[i])].squeeze().numpy().transpose(1, 2, 0)
#     plt.imshow(out)
# print(img_rgb)

# out = img_rgb['1_1'].squeeze().numpy().transpose(1, 2, 0)
# plt.imshow(out)
# out = img_rgb['1_2'].squeeze().numpy().transpose(1, 2, 0)
# plt.imshow(out)
# out = img_rgb['1_4'].squeeze().numpy().transpose(1, 2, 0)
# plt.imshow(out)
# plt.show()
# cv2.imshow('image', out)
# cv2.waitKey(0)

img2 = Contrast(ALTM(img))
fig = predict(img2)
# camera = dict(
#     up=dict(x=0, y=0, z=1),
#     center=dict(x=0, y=0, z=0),
#     eye=dict(x=1.25, y=1.25, z=1.25)
# )
camera = dict(
    eye=dict(x=0, y=0, z=3)
)

fig.update_layout(scene_camera=camera)
# fig.write_image()
# fig.write_image('x.png')
# img3 = fig.to_image()
fig.show()
