# -- coding: utf-8 --
import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataset import BuildDataLoader, BuildDataset


imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
paths = [imgs_path, masks_path, bboxes_path, labels_path]
# load the data into data.Dataset
dataset = BuildDataset(paths)

# shape=(3265, ...), ... as(n, 4)
w, h = [], []
# print(dataset.bbox.shape)
for img_bbox in dataset.bbox:
    # print(img_bbox)
    for bbox in img_bbox:
        # print(bbox)
        w.append(bbox[2] - bbox[0])
        h.append(bbox[3] - bbox[1])
w, h = np.array(w), np.array(h)
aspect_ratio = w/h
scale = np.sqrt(w*h)

plt.hist(x=aspect_ratio, bins=15)
plt.savefig("./aspect.png")
plt.close()
plt.hist(x=scale, bins=15)
plt.savefig("./scale.png")

plt.hist(x=w, bins=15)
plt.show()

plt.hist(x=h, bins=15)
plt.show()

print('aspect_ratio', np.mean(aspect_ratio), np.median(aspect_ratio))
print('scale', np.mean(scale), np.median(scale))

print('w', np.mean(w), np.median(w))
print('h', np.mean(h), np.median(h))
print('w/h', np.mean(w)/np.mean(h), np.median(w)/np.median(h))
print('w*h', np.sqrt(np.mean(w)*np.mean(h)), np.sqrt(np.median(w)*np.median(h)))