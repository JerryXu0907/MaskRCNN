from os import remove
from matplotlib.pyplot import draw
import torch
from torch.nn import functional as F
from torchvision import transforms
from torch import nn, Tensor
from dataset import *
from utils import *

import torchvision


class RPNHead(torch.nn.Module):

    def __init__(self,  device='cuda', anchors_param=dict(ratio=0.8,scale=256, grid_size=(50, 68), stride=1)):
        # Initialize the backbone, intermediate layer clasifier and regressor heads of the RPN
        super(RPNHead,self).__init__()

        self.device=device
        # TODO Define Backbone
        self.backbone = nn.Sequential(nn.Conv2d(3, 16, 5, 1, padding="same"),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(16, 32, 5, 1, padding="same"),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(32, 64, 5, 1, padding="same"),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(64, 128, 5, 1, padding="same"),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(),
                                      nn.MaxPool2d(2, 2),
                                      nn.Conv2d(128, 256, 5, 1, padding="same"),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU()).to(device)
        # TODO  Define Intermediate Layer
        self.intermediate = nn.Sequential(nn.Conv2d(256, 256, 3, padding="same"),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU()).to(device)
        # TODO  Define Proposal Classifier Head
        self.classifier = nn.Sequential(nn.Conv2d(256, 1, 1, 1, padding="same"),
                                        nn.Sigmoid()).to(device)
        # TODO Define Proposal Regressor Head
        self.regressor = nn.Conv2d(256, 4, 1, 1, padding="same").to(device)

        #  find anchors
        self.anchors_param=anchors_param
        self.anchors=self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict={}



    # Forward  the input through the backbone the intermediate layer and the RPN heads
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       logits: (bz,1,grid_size[0],grid_size[1])}
    #       bbox_regs: (bz,4, grid_size[0],grid_size[1])}
    def forward(self, X):
        X = X.to(self.device)
        #TODO forward through the Backbone
        X = self.forward_backbone(X)

        #TODO forward through the Intermediate layer
        feature = self.intermediate(X)

        #TODO forward through the Classifier Head
        logits = self.classifier(feature)

        #TODO forward through the Regressor Head
        bbox_regs = self.regressor(feature)

        assert logits.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert bbox_regs.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return logits, bbox_regs


    # Forward input batch through the backbone
    # Input:
    #       X: (bz,3,image_size[0],image_size[1])}
    # Ouput:
    #       X: (bz,256,grid_size[0],grid_size[1])
    def forward_backbone(self,X):
        #####################################
        # TODO forward through the backbone
        #####################################
        X = self.backbone(X)

        assert X.shape[1:4]==(256,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return X



    # This function creates the anchor boxes
    # Output:
    #       anchors: (grid_size[0],grid_size[1],4)
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        ######################################
        # TODO create anchors
        ######################################
        w = np.sqrt(aspect_ratio*(scale**2))
        h = 1/np.sqrt(aspect_ratio/(scale**2))
        image_x, image_y = int(1088/stride), int(800/stride)
        grid_rows, grid_cols = grid_sizes[0], grid_sizes[1]
        x_span, y_span = image_x/grid_cols, image_y/grid_rows

        anchors= torch.zeros((grid_rows, grid_cols, 4))
        x = x_span*torch.arange(grid_cols) + x_span/2
        y = y_span*torch.arange(grid_rows) + y_span/2
        x, y = torch.meshgrid(x, y)
        anchors[:, :, 0] = x.permute(1, 0)
        anchors[:, :, 1] = y.permute(1, 0)
        anchors[:, :, 2] = w*torch.ones((grid_rows, grid_cols))
        anchors[:, :, 3] = h*torch.ones((grid_rows, grid_cols))

        assert anchors.shape == (grid_sizes[0], grid_sizes[1], 4)

        return anchors



    def get_anchors(self):
        return self.anchors



    # This function creates the ground truth for a batch of images by using
    # create_ground_truth internally
    # Input:
    #      bboxes_list: list:len(bz){(n_obj,4)}
    #      indexes:      list:len(bz)
    #      image_shape:  tuple:len(2)
    # Output:
    #      ground_clas: (bz,1,grid_size[0],grid_size[1])
    #      ground_coord: (bz,4,grid_size[0],grid_size[1])
    def create_batch_truth(self,bboxes_list,indexes,image_shape):
        #####################################
        # TODO create ground truth for a batch of images
        #####################################
        ground_clas = []
        ground_coord = []
        for i in range(len(bboxes_list)):
            clas, coord = self.create_ground_truth(bboxes_list[i], 
                                                   indexes[i], 
                                                   [self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1]],
                                                   self.get_anchors(),
                                                   image_shape)
            ground_clas.append(clas)
            ground_coord.append(coord)
        ground_clas = torch.stack(ground_clas, dim=0).to(self.device)
        ground_coord = torch.stack(ground_coord, dim=0).to(self.device)
        assert ground_clas.shape[1:4]==(1,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])
        assert ground_coord.shape[1:4]==(4,self.anchors_param['grid_size'][0],self.anchors_param['grid_size'][1])

        return ground_clas, ground_coord


    # This function creates the ground truth for one image
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset used for caching)
    #       grid_size:   tuple:len(2)
    #       anchors:     (grid_size[0],grid_size[1],4)
    #       image_size:  tuple:len(2)
    # Output:
    #       ground_clas:  (1,grid_size[0],grid_size[1])  Note that 1 pos, -1 neg, 0 ignore
    #       ground_coord: (4,grid_size[0],grid_size[1])
    def create_ground_truth(self, bboxes, index, grid_size, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        #####################################################
        H, W = image_size
        S_y, S_x = anchors.shape[:2]
        h_grid, w_grid = H // S_y, W // S_x
        w_a, h_a = anchors[0, 0, 2:]

        # calculate the cutoff index for the grids in which the anchor cross
        # the boundary
        h_cutoff = int((h_a / 2 - h_grid / 2) // h_grid)
        w_cutoff = int((w_a / 2 - w_grid / 2) // w_grid)
        ignore_mask = torch.ones(S_y, S_x)
        if h_cutoff >= 0:
            ignore_mask[:h_cutoff + 1] = 0.
            ignore_mask[-h_cutoff - 1:] = 0.
        if w_cutoff >= 0:
            ignore_mask[:, :w_cutoff + 1] = 0.
            ignore_mask[:, -w_cutoff - 1:] = 0.

        # mask-out the cross boundary anchors
        new_anchors = ignore_mask.unsqueeze(2) * anchors

        new_anchors = new_anchors.view(-1, 4) # g0 * g1, 4
        iou = IOU(new_anchors, bboxes) # g0*g1, n_boxes
        max_box_indices = torch.argmax(iou, dim=1)
        pos_labels = torch.zeros(new_anchors.shape[0])

        # record the indices of the anchors that has the largest iou
        # for each bounding box
        highest_iou, _ = torch.max(iou, dim=0)
        for i in range(len(highest_iou)):
            pos_labels += (iou[:, i] == highest_iou[i])
        
        # record the anchors that has iou with bbox larger than 0.7
        iou_over7 = 1 - torch.prod(iou <= 0.7, dim=1)

        # get the positive labels and corresponding anchor box indices
        pos_labels = (pos_labels + iou_over7) > 0
        pos_indices = torch.nonzero(pos_labels).squeeze(1)

        # calculate the regressor groundtruth for each positive anchor box
        ground_coord = torch.zeros_like(new_anchors)
        for i in pos_indices:
            t_x_star = (bboxes[max_box_indices[i], 0] - new_anchors[i, 0]) / new_anchors[i, 2]
            t_y_star = (bboxes[max_box_indices[i], 1] - new_anchors[i, 1]) / new_anchors[i, 3]
            t_w_star = torch.log(bboxes[max_box_indices[i], 2] / new_anchors[i, 2])
            t_h_star = torch.log(bboxes[max_box_indices[i], 3] / new_anchors[i, 3])
            ground_coord[i] = torch.tensor([t_x_star, t_y_star, t_w_star, t_h_star])
        
        # record all non_positive anchor boxes
        neg_labels = ~pos_labels
        # record anchor boxes with all iou < 0.3
        iou_below3 = torch.prod(iou < 0.3, dim=1)
        neg_labels = neg_labels * iou_below3
        # 1 positive, -1 negative, 0 ignore
        ground_clas = pos_labels.int() - neg_labels.int()

        # reshape
        ground_coord = ground_coord.view(S_y, S_x, 4).permute(2, 0, 1)
        ground_clas = ground_clas.view(S_y, S_x).unsqueeze(0) * ignore_mask

        self.ground_dict[key] = (ground_clas, ground_coord)

        assert ground_clas.shape==(1,grid_size[0],grid_size[1])
        assert ground_coord.shape==(4,grid_size[0],grid_size[1])

        return ground_clas, ground_coord





    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self,p_out,n_out):

        #torch.nn.BCELoss()
        # TODO compute classifier's loss

        sum_count = p_out.shape[0] + n_out.shape[0]
        loss = torch.nn.BCELoss(reduction='sum')
        loss_class = loss(p_out, torch.ones_like(p_out)) + loss(n_out, torch.zeros_like(n_out))
        return loss_class, sum_count



    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self,pos_target_coord,pos_out_r):
            #torch.nn.SmoothL1Loss()
            # TODO compute regressor's loss
        sum_count = pos_out_r.shape[0]
        loss = torch.nn.SmoothL1Loss(reduction='sum')
        loss_reg = loss(pos_out_r.reshape(-1, 1), pos_target_coord.reshape(-1, 1))

        return loss_reg, sum_count



    # Compute the total loss
    # Input:
    #       clas_out: (bz,1,grid_size[0],grid_size[1])
    #       regr_out: (bz,4,grid_size[0],grid_size[1])
    #       targ_clas:(bz,1,grid_size[0],grid_size[1])
    #       targ_regr:(bz,4,grid_size[0],grid_size[1])
    #       l: lambda constant to weight between the two losses
    #       effective_batch: the number of anchors in the effective batch (M in the handout)
    # TODO Change effective_batch size
    def compute_loss(self,clas_out,regr_out,targ_clas,targ_regr, l=50, effective_batch=50, eval=False):
        #############################
        # TODO compute the total loss
        #############################
        clas_out = clas_out.permute(0, 2, 3, 1).reshape(-1, 1)
        regr_out = regr_out.permute(0, 2, 3, 1).reshape(-1, 4)
        targ_clas = targ_clas.permute(0, 2, 3, 1).reshape(-1, 1)
        targ_regr = targ_regr.permute(0, 2, 3, 1).reshape(-1, 4)

        pos_targ = torch.nonzero(targ_clas == 1)[:, 0]  # pos targ index, 1d
        neg_targ = torch.nonzero(targ_clas == -1)[:, 0]  # neg targ index, 1d

        if not eval:
            if pos_targ.shape[0] + neg_targ.shape[0] < effective_batch:
                # no enough samples
                mbatch_targ_regr = targ_regr[pos_targ]
                mbatch_regr_out = regr_out[pos_targ]
                mbatch_clas_out_pos = clas_out[pos_targ]
                mbatch_clas_out_neg = clas_out[neg_targ]

            elif pos_targ.shape[0] < effective_batch // 2:
                # no enough pos samples
                neg_targ_sampleidx = np.random.choice(a=neg_targ.shape[0], size=effective_batch - pos_targ.shape[0],
                                                    replace=False)
                mbatch_targ_regr = targ_regr[pos_targ]
                mbatch_regr_out = regr_out[pos_targ]
                mbatch_clas_out_pos = clas_out[pos_targ]
                mbatch_clas_out_neg = clas_out[neg_targ[neg_targ_sampleidx]]
            else:
                pos_sample = effective_batch // 2
                neg_sample = effective_batch - pos_sample
                pos_targ_sampleidx = np.random.choice(a=pos_targ.shape[0], size=pos_sample, replace=False)
                neg_targ_sampleidx = np.random.choice(a=neg_targ.shape[0], size=neg_sample, replace=False)
                mbatch_targ_regr = targ_regr[pos_targ[pos_targ_sampleidx]]
                mbatch_regr_out = regr_out[pos_targ[pos_targ_sampleidx]]
                mbatch_clas_out_pos = clas_out[pos_targ[pos_targ_sampleidx]]
                mbatch_clas_out_neg = clas_out[neg_targ[neg_targ_sampleidx]]

            # minibatch_targ_regr (M, 4)
            loss_c, sum_count_c = self.loss_class(mbatch_clas_out_pos, mbatch_clas_out_neg)
            loss_r, sum_count_r = self.loss_reg(mbatch_targ_regr, mbatch_regr_out)
            loss = loss_c / sum_count_c + l * loss_r / 3400
            return loss, loss_c, loss_r
        else:
            
            loss_c, sum_count_c = self.loss_class(clas_out[pos_targ], clas_out[neg_targ])
            loss_r, sum_count_r = self.loss_reg(targ_regr[pos_targ], regr_out[pos_targ])
            loss = loss_c / sum_count_c + loss_r / 3400
            return loss, loss_c, loss_r
            # loss = (l * loss_c + loss_r) / effective_batch
            



    # Post process for the outputs for a batch of images
    # Input:
    #       out_c:  (bz,1,grid_size[0],grid_size[1])}
    #       out_r:  (bz,4,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinates of the boxes that the NMS kept)
    def postprocess(self,out_c,out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10, draw=False):
       ####################################
       # TODO postprocess a batch of images
       #####################################
        if draw:
            nms_clas_list = []
            nms_prebox_list = []
            prenms_clas_list = []
            prenms_prebox_list = []
            for i in range(len(out_c)):
                nms_clas, nms_prebox, sorted_clas, sorted_coord = self.postprocessImg(out_c[i], out_r[i], IOU_thresh, keep_num_preNMS, keep_num_postNMS, draw=draw)
                nms_clas_list.append(nms_clas)
                nms_prebox_list.append(nms_prebox)
                prenms_clas_list.append(sorted_clas)
                prenms_prebox_list.append(sorted_coord)
            return nms_clas_list, nms_prebox_list, prenms_clas_list, prenms_prebox_list
        else:
            nms_clas_list = []
            nms_prebox_list = []
            for i in range(len(out_c)):
                nms_clas, nms_prebox = self.postprocessImg(out_c[i], out_r[i], IOU_thresh, keep_num_preNMS, keep_num_postNMS)
                nms_clas_list.append(nms_clas)
                nms_prebox_list.append(nms_prebox)
            return nms_clas_list, nms_prebox_list



    # Post process the output for one image
    # Input:
    #      mat_clas: (1,grid_size[0],grid_size[1])}  (scores of the output boxes)
    #      mat_coord: (4,grid_size[0],grid_size[1])} (encoded coordinates of the output boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4) (decoded coordinates of the boxes that the NMS kept)
    def postprocessImg(self,mat_clas,mat_coord, IOU_thresh,keep_num_preNMS, keep_num_postNMS, draw=False):
        ######################################
        # TODO postprocess a single image
        #####################################
        flatten_regr, flatten_clas, flatten_anchors = output_flattening(mat_coord, mat_clas, self.get_anchors())
        boxes = output_decoding(flatten_regr, flatten_anchors, device=self.device)
        sorted_clas, sorted_index = torch.sort(flatten_clas, descending=True)
        sorted_coord = torch.index_select(boxes, 0, sorted_index)
        sorted_clas = sorted_clas[:keep_num_preNMS]
        sorted_coord = sorted_coord[:keep_num_preNMS]
        nms_clas, nms_prebox = self.NMS(sorted_clas, sorted_coord, IOU_thresh)
        nms_clas = nms_clas[:keep_num_postNMS]
        nms_prebox = nms_prebox[:keep_num_postNMS]
        if draw:
            return nms_clas, nms_prebox, sorted_clas, sorted_coord
        return nms_clas, nms_prebox



    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas,prebox, thresh):
        ##################################
        # TODO perform NMS
        ##################################
        # import ipdb; ipdb.set_trace()
        l = list(range(len(prebox)))
        indices = []
        while len(l) > 0:
            box = prebox[l[0]]
            indices.append(l[0])
            l.remove(l[0])
            remove_list = []
            for j in l:
                iou = single_IOU(box, prebox[j])
                if iou > thresh:
                    remove_list.append(j)
            for k in remove_list:
                l.remove(k)
        indices = torch.tensor(indices).to(self.device)
        nms_prebox = torch.index_select(prebox, dim=0, index=indices)
        nms_clas = torch.index_select(clas, dim=0, index=indices)
        return nms_clas,nms_prebox
    
    def top20(self, mat_clas,mat_coord):
        flatten_regr, flatten_clas, flatten_anchors = output_flattening(mat_coord, mat_clas, self.get_anchors())
        boxes = output_decoding(flatten_regr, flatten_anchors, device=self.device)
        sorted_clas, sorted_index = torch.sort(flatten_clas, descending=True)
        sorted_coord = torch.index_select(boxes, 0, sorted_index)
        sorted_clas = sorted_clas[:20]
        sorted_coord = sorted_coord[:20]
        return sorted_clas, sorted_coord

def plot_top20():
    from dataset import BuildDataLoader, BuildDataset
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    rpn = RPNHead()
    rpn.load_state_dict(torch.load("./train_result/best_model.pth"))
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, bboxes_path, labels_path]
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    batch_size = 1
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = test_build_loader.loader()
    rpn.eval()

    for i, batch in enumerate(test_loader, 0):
        logits, bbox_regs = rpn(batch["img"])
        images = batch['img'][0, :, :, :]
        boxes = batch['bbox']
        logits = logits[0]
        bbox_regs = bbox_regs[0]
        sorted_clas, sorted_coord = rpn.top20(logits, bbox_regs)

        images = transforms.functional.normalize(images,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(images.permute(1, 2, 0))

        for j in range(len(sorted_clas)):
            if sorted_clas[j] > 0.5:
                col = 'b'
                coord = sorted_coord[j].cpu().detach().numpy()
                rect = patches.Rectangle((coord[0] - coord[2] / 2, coord[1] - coord[3] / 2), coord[2], coord[3], fill=False,
                                        color=col)
                ax.add_patch(rect)
        for j in range(len(boxes[0])):
            col = 'r'
            rect = patches.Rectangle((boxes[0][j, 0] - boxes[0][j, 2] / 2, boxes[0][j, 1] - boxes[0][j, 3] / 2), boxes[0][j, 2], boxes[0][j, 3],
                                     fill=False, color=col)
            ax.add_patch(rect)
        ax.set_title("Top 20 Proposals")

        plt.savefig(f"./predict_vis/{i}_top20.png")
        plt.close()

        if (i > 20):
            break

def plot_nms():
    rpn = RPNHead()
    # X = torch.randn(2, 3, 800, 1088)
    # logits, bbox_regs = rpn(X)
    # print("logits shape", logits.shape)
    # print("bbox_reg shape", bbox_regs.shape)
    # print(rpn.get_anchors().shape)
    # fake_bbox = torch.tensor([[250, 320, 100, 120], [300, 400, 100, 120]])
    # rpn.create_ground_truth(fake_bbox, 0, [50, 68], rpn.get_anchors(), [800, 1088])
    rpn.load_state_dict(torch.load("./train_result/best_model.pth"))
    
    from dataset import BuildDataLoader, BuildDataset
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, bboxes_path, labels_path]
    dataset = BuildDataset(paths)

    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_loader = train_build_loader.loader()
    rpn.eval()
    for i, batch in enumerate(train_loader, 0):
        logits, bbox_regs = rpn(batch["img"])
        images = batch['img'][0, :, :, :]
        indexes = batch['idx']
        boxes = batch['bbox']
        # gt, ground_coord = rpn.create_batch_truth(boxes, indexes, images.shape[-2:])

        nms_clas_list, nms_prebox_list, prenms_clas_list, prenms_prebox_list = rpn.postprocess(logits, bbox_regs, draw=True)

        images = transforms.functional.normalize(images,
                                                 [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                 [1 / 0.229, 1 / 0.224, 1 / 0.225], inplace=False)
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[1].imshow(images.permute(1, 2, 0))

        for j in range(len(nms_clas_list[0])):
            if nms_clas_list[0][j] > 0.5:
                col = 'b'
                coord = nms_prebox_list[0][j].cpu().detach().numpy()
                rect = patches.Rectangle((coord[0] - coord[2] / 2, coord[1] - coord[3] / 2), coord[2], coord[3], fill=False,
                                        color=col)
                ax[1].add_patch(rect)
        for j in range(len(boxes[0])):
            col = 'r'
            rect = patches.Rectangle((boxes[0][j, 0] - boxes[0][j, 2] / 2, boxes[0][j, 1] - boxes[0][j, 3] / 2), boxes[0][j, 2], boxes[0][j, 3],
                                     fill=False, color=col)
            ax[1].add_patch(rect)
        ax[1].title.set_text("After NMS")

        ax[0].imshow(images.permute(1, 2, 0))
        for j in range(len(prenms_clas_list[0])):
            if prenms_clas_list[0][j] > 0.5:
                col = 'b'
                coord = prenms_prebox_list[0][j].cpu().detach().numpy()
                rect = patches.Rectangle((coord[0] - coord[2] / 2, coord[1] - coord[3] / 2), coord[2], coord[3], fill=False,
                                        color=col)
                ax[0].add_patch(rect)
        ax[0].title.set_text("Before NMS")

        plt.savefig(f"./predict_vis/{i}.png")
        plt.close()

        if (i > 20):
            break
if __name__=="__main__":
    plot_top20()
    plot_nms()