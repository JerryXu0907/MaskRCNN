from os import remove
import torch
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torchvision.models.detection.image_list import ImageList
import time

from dataset import *
from utils import *
from BoxHead import BoxHead
from pretrained_models import pretrained_models_680


imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
paths = [imgs_path, masks_path, bboxes_path, labels_path]
epoch = 100
batch_size = 4
tolerance = 5
keep_topK = 100
torch.manual_seed(17)

def main():
    dataset = BuildDataset(paths)
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    print("Data Loading")
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()
    print("Training Set Size", len(train_dataset), "Validation Set Size", len(test_dataset))
    print("Creating Model")
    rcnn_net = BoxHead()
    print("Setting Optimizer")
    optimizer = Adam(rcnn_net.parameters(), lr=0.001)

    print("Create Backbone")
    backbone, rpn = pretrained_models_680('checkpoint680.pth')

    best_loss = 1000.
    early_stopping = 0.
    best_model = None
    loss_total_train = []
    loss_c_train = []
    loss_r_train = []
    loss_total_val = []
    loss_c_val = []
    loss_r_val = []
    for i in range(epoch):
        print(f"\nEpoch {i} begins")
        print("Train:")
        loss_total, loss_c, loss_r = train(rcnn_net, backbone, rpn, train_loader, optimizer, i)
        loss_total_train += loss_total
        loss_c_train += loss_c
        loss_r_train += loss_r
        print("Validation")
        loss_total, loss_c, loss_r = val(rcnn_net, backbone, rpn, test_loader, i)
        loss_total_val += loss_total
        loss_c_val += loss_c
        loss_r_val += loss_r
        val_loss_mean = np.mean(np.array(loss_total))
        print("Epoch {} Validation Loss Mean: {:.4f}".format(i, val_loss_mean))
        if val_loss_mean < best_loss:
            best_loss = val_loss_mean
            early_stopping = 0
            best_model = rcnn_net.state_dict()
        else:
            early_stopping += 1
        if early_stopping == tolerance:
            break
    torch.save(best_model, "./train_result/best_model.pth")
    np.save("./train_result/total_train.npy", np.array(loss_total_train))
    np.save("./train_result/c_train.npy", np.array(loss_c_train))
    np.save("./train_result/r_train.npy", np.array(loss_r_train))
    np.save("./train_result/total_val.npy", np.array(loss_total_val))
    np.save("./train_result/c_val.npy", np.array(loss_c_val))
    np.save("./train_result/r_val.npy", np.array(loss_r_val))

def train(model: BoxHead, backbone, rpn, loader, optimizer, i):
    loss_t = []
    loss_c = []
    loss_r = []
    model.train()
    for idx, data_batch in enumerate(loader):
        images = data_batch['img'].to(model.device)
        bbox = data_batch["bbox"]
        labels = data_batch["labels"]
        bbox = [b.cuda() for b in bbox]
        labels = [l.cuda() for l in labels]
        a = time.time()
        backout = backbone(images)
        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        rpnout = rpn(im_lis, backout)
        proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
        fpn_feat_list= list(backout.values())
        b = time.time()
        feature_vectors = model.MultiScaleRoiAlign(fpn_feat_list, proposals, P=model.P)
        c = time.time()
        gt_labels, regressor_target = model.create_ground_truth(proposals, labels, bbox)
        d = time.time()
        class_logits, box_pred = model(feature_vectors)
        e = time.time()

        loss, loss1, loss2 = model.compute_loss(class_logits, box_pred, gt_labels, regressor_target)
        f = time.time()
        # print(b-a, c-b, d-c, e-d, f-e)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_t.append(loss.item())
        loss_c.append(loss1.item())
        loss_r.append(loss2.item())
        if idx % 50 == 0:
            print("Epoch {} Batch {}: Total Loss {:.3f} ({:.3f}); Class Loss {:.3f} ({:.3f}); Regressor Loss {:.3f} ({:.3f})".format(i, idx, loss.item(), np.array(loss_t).mean(), 
                                                        loss1.item(), np.array(loss_c).mean(), loss2.item(), np.array(loss_r).mean()))
    return loss_t, loss_c, loss_r

def val(model: BoxHead, backbone, rpn, loader, i):
    loss_t = []
    loss_c = []
    loss_r = []
    model.eval()
    for idx, data_batch in enumerate(loader):
        images = data_batch['img'].to(model.device)
        bbox = data_batch["bbox"]
        labels = data_batch["labels"]
        bbox = [b.cuda() for b in bbox]
        labels = [l.cuda() for l in labels]
        backout = backbone(images)

        # The RPN implementation takes as first argument the following image list
        im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
        rpnout = rpn(im_lis, backout)
        proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
        fpn_feat_list= list(backout.values())
        feature_vectors = model.MultiScaleRoiAlign(fpn_feat_list, proposals, P=model.P)
        gt_labels, regressor_target = model.create_ground_truth(proposals, labels, bbox)
        class_logits, box_pred = model(feature_vectors)

        loss, loss1, loss2 = model.compute_loss(class_logits, box_pred, gt_labels, regressor_target)
        loss_t.append(loss.item())
        loss_c.append(loss1.item())
        loss_r.append(loss2.item())
        if idx % 50 == 0:
            print("Valid Epoch {} Batch {}: Total Loss {:.3f} ({:.3f}); Class Loss {:.3f} ({:.3f}); Regressor Loss {:.3f} ({:.3f})".format(i, idx, loss.item(), np.array(loss_t).mean(), 
                                                        loss1.item(), np.array(loss_c).mean(), loss2.item(), np.array(loss_r).mean()))
    return loss_t, loss_c, loss_r

if __name__ == "__main__":
    main()