# -- coding: utf-8 --

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import h5py
from collections import Counter
import pytorch_lightning as pl

from matplotlib import patches
import matplotlib.pyplot as plt

np.random.seed(17)
data_len = 3265

class SoloDataset(Dataset):
    def __init__(self, img, mask, bbox, label, indices=None,
                 transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Resize(size=(800, 1066)),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                     transforms.Pad(padding=(11, 0), fill=0)
                 ])):
        self.imageScale = 1066 / 400
        self.indices = indices
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(800, 1066)),
            transforms.Pad(padding=(11, 0), fill=0)
        ])
        with h5py.File(img, 'r') as img:
            # shape=(3265, 3, 300, 400), type=ndarray
            self.image = img['data'][:]
        with h5py.File(mask, 'r') as mask:
            # shape=(3843, 300, 400)
            self.mask = mask['data'][:]
        # shape=(3265, ...), ... as(n, 4)
        self.bbox = np.load(bbox, allow_pickle=True)
        # shape=(3265, ...), ... as(m, )
        self.label = np.load(label, allow_pickle=True)
        if indices is None:
            self.indices = np.random.permutation(len(self.image))
        else:
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        label_sum = 0
        idx = self.indices[idx]
        for label in self.label[:idx]:
            label_sum += len(label)
        labels = len(self.label[idx])
        masks = self.mask[label_sum:label_sum + labels].astype(np.uint8)
        image = self.transform(self.image[idx].astype(np.uint8).transpose(1, 2, 0))
        for i, mask in enumerate(masks):
            if i == 0:
                res_masks = self.mask_transform(mask)
            else:
                res_masks = torch.vstack([res_masks, self.mask_transform(mask)])

        bbox = torch.tensor(self.bbox[idx] * self.imageScale, requires_grad=False)
        label = torch.tensor(self.label[idx], requires_grad=False)

        return image, label, res_masks, bbox


def draw(imgs, labels, masks, bboxs):
    cmap = ["Blues", "Greens", "Reds"]
    color=['b', 'g', 'r']
    label=['vehicle', 'person', 'animal']
    # col = 6
    # row = len(img) // col + 1
    col = 5
    row = 2
    fig, ax = plt.subplots(row, col, figsize=(18, 5))
    N = len(imgs)
    # imgs = imgs.clone().detach().numpy()
    for i in range(0, N):
        img_ = imgs[i]
        col_ = i % 5
        row_ = i // 5
        img_ = transforms.functional.normalize(img_,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        ax[row_][col_].imshow(img_.permute(1, 2, 0).squeeze())
        label_ = labels[i].clone().detach().numpy()
        label_cls = Counter(label_)
        # print(label_cls)
        begin_ = 0
        masks_ = masks[i]
        bbox = bboxs[i].clone().detach().numpy()

        for cls in label_cls.keys():
            cls_num = label_cls[cls]
            for mask in masks_[begin_:begin_ + cls_num]:
                mask_ = mask.clone().detach().numpy()
                msk = np.ma.masked_where(mask_ == 0, mask_)
                ax[row_][col_].imshow(msk.squeeze(), cmap=cmap[int(cls)-1], alpha=0.7)
            for box in bbox[begin_:begin_ + cls_num]:
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                         edgecolor=color[int(cls) - 1],
                                         facecolor="none")
                ax[row_][col_].add_patch(rect)
                ax[row_][col_].text(box[0], box[1]-5, label[int(cls) - 1], fontsize=8, c=color[int(cls) - 1])
            begin_ += cls_num

    plt.show()

class HW3DataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size=2, val_batch_size=2):
        super().__init__()
        self.indices = np.random.permutation(data_len)
        self.data_train = None
        self.data_val = None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
    
    def train_dataloader(self):
        if self.data_train is None:
            self.data_train = SoloDataset(
                img='data/hw3_mycocodata_img_comp_zlib.h5',
                mask='data/hw3_mycocodata_mask_comp_zlib.h5',
                bbox='data/hw3_mycocodata_bboxes_comp_zlib.npy',
                label='data/hw3_mycocodata_labels_comp_zlib.npy',
                indices=self.indices[:int(0.8*data_len)]
            )
        return DataLoader(self.data_train, batch_size=self.train_batch_size, num_workers=8, collate_fn=collate_fn_solo, shuffle=True)

    def val_dataloader(self):
        if self.data_val is None:
            self.data_val = SoloDataset(
                img='data/hw3_mycocodata_img_comp_zlib.h5',
                mask='data/hw3_mycocodata_mask_comp_zlib.h5',
                bbox='data/hw3_mycocodata_bboxes_comp_zlib.npy',
                label='data/hw3_mycocodata_labels_comp_zlib.npy',
                indices=self.indices[int(0.8*data_len):]
            )
        return DataLoader(self.data_val, batch_size=self.val_batch_size, num_workers=8, collate_fn=collate_fn_solo)
    
    def test_dataloader(self):
        return DataLoader(self.data_val, batch_size=1, num_workers=8)

def collate_fn_solo(batch):
    images, labels, masks, bounding_boxes = list(zip(*batch))
    data_batch = {"img": torch.stack(images), "bounding_boxes": bounding_boxes, "labels": labels, "masks": masks}
    return data_batch


'''
one batch
img: tensor, shape=(bsize, 3, 800, 1088)
label: list, len=bsize
mask: list, len=bsize, item_shape=(label_num, 800, 1088)
bbox: list, len=bsize, item_shape=(label_num, 4)
'''
imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
soloDataset = SoloDataset(img=imgs_path, label=labels_path, mask=masks_path, bbox=bboxes_path)
train_loader = DataLoader(soloDataset, batch_size=10, collate_fn=collate_fn_solo, shuffle=True)
nxt = iter(train_loader).next()
img, label, mask, bbox = nxt['img'], nxt['labels'], nxt['masks'], nxt['bounding_boxes']
print(img.shape)
print(label)
print(mask[0].shape)
print(bbox)
draw(img, label, mask, bbox)
# mask_color_list = ["jet", "ocean", "Spectral"]

