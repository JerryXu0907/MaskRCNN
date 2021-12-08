from matplotlib.pyplot import grid
import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
from dataset import *
from utils import *
import torchvision
from backbone import Resnet50Backbone


class RPNHead(torch.nn.Module):
    # The input of the initialization of the RPN is:
    # Input:
    #       computed_anchors: the anchors computed in the dataset
    #       num_anchors: the number of anchors that are assigned to each grid cell
    #       in_channels: number of channels of the feature maps that are outputed from the backbone
    #       device: the device that we will run the model
    def __init__(self, num_anchors=3, in_channels=256, device='cuda',
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64]),
                #  freeze_backbone=True,
                #  backbone_ckpt=None
                 ):
        ######################################
        # TODO initialize RPN
        #######################################
        super(RPNHead,self).__init__()
        self.anchors_param = anchors_param
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        self.device = device
        self.len_fpn = len(anchors_param['ratio'])
        self.total_anchors = 0
        for i in range(self.len_fpn):
            self.total_anchors += self.anchors_param['grid_size'][i][0] * self.anchors_param['grid_size'][i][1] * 3
        
        # self.backbone = Resnet50Backbone(checkpoint_file=backbone_ckpt, device=device, eval=freeze_backbone)
        self.intermediate = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding="same"),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU()).to(device)
        
        self.classifier = nn.Sequential(nn.Conv2d(in_channels, self.num_anchors, 1, 1, padding="same"),
                                        nn.Sigmoid()).to(device)
        
        self.regressor = nn.Conv2d(in_channels, 4*self.num_anchors, 1, 1, padding="same").to(device)
        self.anchors = self.create_anchors(self.anchors_param['ratio'],self.anchors_param['scale'],self.anchors_param['grid_size'],self.anchors_param['stride'])
        self.ground_dict = {}
        self.backbone_keys = ['0', '1', '2', '3', 'pool']
        

    # Forward each level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
    def forward(self, X):
        logits = []
        bbox_regs = []
        for i in range(self.len_fpn):
            logit, reg = self.forward_single(X[self.backbone_keys[i]])
            logits.append(logit)
            bbox_regs.append(reg)
        return logits, bbox_regs

    # Forward a single level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       feature: (bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logit: (bz,1*num_acnhors,grid_size[0],grid_size[1])
    #       bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])
    def forward_single(self, feature):
        out = self.intermediate(feature)
        logit = self.classifier(out)
        bbox_reg = self.regressor(out)
        return logit, bbox_reg


    # This function creates the anchor boxes for all FPN level
    # Input:
    #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
    #       scale:        list:len(FPN)
    #       grid_size:    list:len(FPN){tuple:len(2)}
    #       stride:        list:len(FPN)
    # Output:
    #       anchors_list: list:len(FPN){(num_anchors, grid_size[0], grid_size[1], 4)}
    # Note:
    #       anchors is in x, y, w, h form
    def create_anchors(self, aspect_ratio, scale, grid_sizes, stride):
        len_FPN = len(aspect_ratio)
        anchor_list = []
        for i in range(len_FPN):
            anchors = self.create_anchors_single(aspect_ratio[i], scale[i], grid_sizes[i], stride[i])
            anchor_list.append(anchors)
        return anchor_list



    # This function creates the anchor boxes for one FPN level
    # Input:
    #      aspect_ratio: list:len(number_of_aspect_ratios)
    #      scale: scalar
    #      grid_size: tuple:len(2)
    #      stride: scalar
    # Output:
    #       anchors: (num_anchors, grid_size[0], grid_size[1], 4)
    def create_anchors_single(self, aspect_ratio, scale, grid_sizes, stride):
        anchors= torch.zeros((3, grid_sizes[0], grid_sizes[1], 4))
        for j in range(3):
            w = np.sqrt(aspect_ratio[j]*(scale**2))
            h = 1/np.sqrt(aspect_ratio[j]/(scale**2))
            grid_rows, grid_cols = grid_sizes[0], grid_sizes[1]
            x_span, y_span = stride, stride
            x = x_span*torch.arange(grid_cols) + x_span/2
            y = y_span*torch.arange(grid_rows) + y_span/2
            x, y = torch.meshgrid(x, y)
            anchors[j, :, :, 0] = x.permute(1, 0)
            anchors[j, :, :, 1] = y.permute(1, 0)
            anchors[j, :, :, 2] = w*torch.ones((grid_rows, grid_cols))
            anchors[j, :, :, 3] = h*torch.ones((grid_rows, grid_cols))
        assert anchors.shape == (3, grid_sizes[0], grid_sizes[1], 4)
        # anchors = anchors.reshape(-1, 4)
        return anchors

    def get_anchors(self):
        return self.anchors

    # This function creates the ground truth for a batch of images
    # Input:
    #      bboxes_list: list:len(bz){(number_of_boxes,4)}
    #      indexes: list:len(bz)
    #      image_shape: list:len(bz){tuple:len(2)}
    # Ouput:
    #      ground: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
    #      ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    def create_batch_truth(self, bboxes_list, indexes, image_shape):
        bz = len(bboxes_list)
        ground = [[], [], [], [], []]
        ground_coord = [[], [], [], [], []]
        for i in range(bz):
            gt_clas, gt_coord = self.create_ground_truth(bboxes_list[i], 
                                                         indexes[i], 
                                                         self.anchors_param['grid_size'],
                                                         self.get_anchors(), 
                                                         image_shape)
            for t in range(self.len_fpn):
                ground[t].append(gt_clas[t])
                ground_coord[t].append(gt_coord[t])
        ground = [torch.stack(i).to(self.device) for i in ground]
        ground_coord = [torch.stack(i).to(self.device) for i in ground_coord]
        return ground, ground_coord

    # This function create the ground truth for one image for all the FPN levels
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset)
    #       grid_size:   list:len(FPN){tuple:len(2)}
    #       anchor_list: list:len(FPN){(num_anchors, grid_size[0], grid_size[1], 4)}
    # Output:
    #       ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
    #       ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
    def create_ground_truth(self, bboxes, index, grid_sizes, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord

        #####################################################
        # TODO create ground truth for a single image
        #####################################################
        H, W = image_size
        gt_coord_list = []
        gt_clas_list = []
        for i in range(len(anchors)):
            anch = anchors[i]
            num_anchors = len(anch)
            S_y, S_x = grid_sizes[i]
            h_grid, w_grid = H // S_y, W // S_x
            w_a = anch[:,0, 0, 2] # num_anchors
            h_a = anch[:,0, 0, 3] # num_anchors

            # calculate the cutoff index for the grids in which the anchor cross
            # the boundary
            h_cutoff = ((h_a / 2 - h_grid / 2) // h_grid).to(torch.int)
            w_cutoff = ((w_a / 2 - w_grid / 2) // w_grid).to(torch.int)
            ignore_mask = torch.ones(num_anchors, S_y, S_x)
            for n in range(num_anchors):
                if h_cutoff[n] >= 0:
                    ignore_mask[n, :h_cutoff[n] + 1] = 0.
                    ignore_mask[n, -h_cutoff[n] - 1:] = 0.
                if w_cutoff[n] >= 0:
                    ignore_mask[n, :, :w_cutoff[n] + 1] = 0.
                    ignore_mask[n, :, -w_cutoff[n] - 1:] = 0.

            # mask-out the cross boundary anchors
            new_anchors = ignore_mask.unsqueeze(-1) * anch

            new_anchors = new_anchors.view(-1, 4) # num_a * g0 * g1, 4
            iou = IOU(new_anchors, bboxes) # num_a*g0*g1, n_boxes
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
            ground_coord = ground_coord.view(num_anchors, S_y, S_x, 4).permute(0, 3, 1, 2).reshape(-1, S_y, S_x)
            ground_clas = ground_clas.view(num_anchors, S_y, S_x) * ignore_mask
            gt_coord_list.append(ground_coord)
            gt_clas_list.append(ground_clas)

        self.ground_dict[key] = (gt_clas_list, gt_coord_list)

        return gt_clas_list, gt_coord_list

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):

        # torch.nn.BCELoss()
        # TODO compute classifier's loss
        sum_count = p_out.shape[0] + n_out.shape[0]
        loss = torch.nn.BCELoss(reduction='sum')
        loss_class = loss(p_out, torch.ones_like(p_out)) + loss(n_out, torch.zeros_like(n_out))
        return loss_class, sum_count

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        # torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss
        sum_count = pos_out_r.shape[0]
        loss = torch.nn.SmoothL1Loss(reduction='sum')
        loss_reg = loss(pos_out_r.reshape(-1, 1), pos_target_coord.reshape(-1, 1))

        return loss_reg, sum_count

    # Compute the total loss for the FPN heads
    # Input:
    #       clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       l: weighting lambda between the two losses
    # Output:
    #       loss: scalar
    #       loss_c: scalar
    #       loss_r: scalar
    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=1, effective_batch=150):
        total_loss, total_loss_c, total_loss_r = 0., 0., 0.
        for n in range(self.len_fpn):
            clas_out = clas_out_list[n]
            regr_out = regr_out_list[n]
            targ_clas = targ_clas_list[n]
            targ_regr = targ_regr_list[n]
            clas_out = clas_out.permute(0, 2, 3, 1).reshape(-1, 1)
            regr_out = regr_out.permute(0, 2, 3, 1).reshape(-1, 4*3).reshape(-1, 4)
            targ_clas = targ_clas.permute(0, 2, 3, 1).reshape(-1, 1)
            targ_regr = targ_regr.permute(0, 2, 3, 1).reshape(-1, 4*3).reshape(-1, 4)

            pos_targ = torch.nonzero(targ_clas == 1)[:, 0]  # pos targ index, 1d
            neg_targ = torch.nonzero(targ_clas == -1)[:, 0]  # neg targ index, 1d

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
            total_loss += loss
            total_loss_c += loss_c
            total_loss_r += loss_r
        return total_loss, total_loss_c, total_loss_r


    # Post process for the outputs for a batch of images
    # Input:
    #       out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)
    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=3):
        bz = len(out_c[0])
        nms_clas_list = []
        nms_prebox_list = []
        for i in range(bz):
            out_c_img = [c[i:i+1] for c in out_c]
            out_r_img = [r[i:i+1] for r in out_r]
            nms_clas, nms_prebox = self.postprocessImg(out_c_img, out_r_img, IOU_thresh, keep_num_preNMS, keep_num_postNMS)
            nms_clas_list.append(nms_clas)
            nms_prebox_list.append(nms_prebox)
        return nms_clas_list, nms_prebox_list

    # Post process the output for one image
    # Input:
    #      mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
    #      mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        flatten_regr, flatten_clas, flatten_anchors = output_flattening(mat_coord, mat_clas, self.get_anchors())
        boxes = output_decoding(flatten_regr, flatten_anchors, device=self.device)
        sorted_clas, sorted_index = torch.sort(flatten_clas, descending=True)
        sorted_coord = torch.index_select(boxes, 0, sorted_index)
        sorted_clas = sorted_clas[:keep_num_preNMS]
        sorted_coord = sorted_coord[:keep_num_preNMS]
        nms_clas, nms_prebox = self.NMS(sorted_clas, sorted_coord, IOU_thresh)
        nms_clas = nms_clas[:keep_num_postNMS]
        nms_prebox = nms_prebox[:keep_num_postNMS]
        return nms_clas, nms_prebox

    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self, clas, prebox, thresh):
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


if __name__ == "__main__":
    device = "cuda:0"
    backbone = Resnet50Backbone(device=device)
    E = torch.ones([2,3,800,1088], device=device)
    backout = backbone(E)

    model = RPNHead(device=device)
    logits, bbox_regs = model(backout)
    print("len(logits): ", len(logits))
    for i in range(model.len_fpn):
        print(f"logits[{i}].shape: ", logits[i].shape)
        print(f"bbox_regs[{i}].shape: ", bbox_regs[i].shape)
        print(f"anchors[{i}].shape: ", model.anchors[i].shape)