import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import torchvision


class BoxHead(torch.nn.Module):
    def __init__(self, Classes=3, P=14, device="cpu"):
        super(BoxHead, self).__init__()
        self.device = device
        self.C = Classes
        self.P = P
        # TODO initialize BoxHead
        self.intermediate = nn.Sequential(
            nn.Linear(in_features=256 * P * P, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU()
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=Classes + 1),
            # nn.Softmax(dim=1)
        ).to(self.device)

        self.regressor = nn.Sequential(
            nn.Linear(in_features=1024, out_features=4 * Classes)
        ).to(self.device)

        self.img_size = (800, 1088)

    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self, proposals, gt_labels, bbox, iou_thresh=0.4):
        b = len(proposals)
        labels = []
        regressor_target = []
        for i in range(b):
            iou = IOU(proposals[i], bbox[i], xaya=True)  # per_img_proposal * n_obj
            max_iou, arg_max_iou = torch.max(iou, dim=1)
            # arg_max_iou += 1
            label = torch.index_select(gt_labels[i], 0, arg_max_iou)
            bboxes = torch.index_select(bbox[i], 0, arg_max_iou)  # per_img_proposal * 4

            target_bbox = torch.zeros_like(bboxes)
            x_p = (proposals[i][:, 0] + proposals[i][:, 2]) / 2
            y_p = (proposals[i][:, 1] + proposals[i][:, 3]) / 2
            w_p = -proposals[i][:, 0] + proposals[i][:, 2]
            h_p = -proposals[i][:, 1] + proposals[i][:, 3]
            target_bbox[:, 0] = (bboxes[:, 0] - x_p) / w_p
            target_bbox[:, 1] = (bboxes[:, 1] - y_p) / h_p
            target_bbox[:, 2] = torch.log(bboxes[:, 2] / w_p)
            target_bbox[:, 3] = torch.log(bboxes[:, 3] / h_p)

            background = (max_iou > iou_thresh).int()
            label = label * background
            labels.append(label)
            regressor_target.append(target_bbox)
        labels = torch.cat(labels, dim=0).unsqueeze(1)
        regressor_target = torch.cat(regressor_target, dim=0)
        return labels, regressor_target

    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, 
                               box_regression, 
                               proposals, 
                               gt_boxes=None,
                               conf_thresh=0.5, 
                               keep_num_preNMS=500, 
                               keep_num_postNMS=100,
                               train=False):
        b = len(proposals)
        boxes = []
        scores = []
        labels = []
        start = 0
        for i in range(b):
            cls = class_logits[start: start + len(proposals[i])]
            reg_box = box_regression[start: start + len(proposals[i])]
            start += len(proposals[i])

            # Confidence cutoff
            non_bg = torch.nonzero(cls[:, 0] < conf_thresh).squeeze(1)
            cls = cls[non_bg]
            reg_box = reg_box[non_bg]
            prop = proposals[i][non_bg]

            # get rid of all background predictions
            conf, cls = torch.max(cls, dim=1)
            nonbg_indices = torch.nonzero(cls).squeeze(1)
            conf = torch.index_select(conf, 0, nonbg_indices)
            cls = torch.index_select(cls, 0, nonbg_indices) - 1
            reg_box = torch.index_select(reg_box, 0, nonbg_indices)
            prop = torch.index_select(prop, 0, nonbg_indices)

            # decode the boxes
            reg_cls = torch.zeros_like(prop)
            reg_box = reg_box.reshape(-1, 3, 4)
            for j in range(len(reg_cls)):
                reg_cls[j] = reg_box[j, cls[j]]
            box = output_decodingd(reg_cls, prop)  # x1 y1 x2 y2 type

            # Cutoff pre NMS
            conf = conf[:keep_num_preNMS]
            cls = cls[:keep_num_preNMS]
            box = box[:keep_num_preNMS]

            left_indices = []
            for c in range(self.C):
                indices = torch.nonzero(cls == c).squeeze(1)
                if len(indices) == 0:
                    continue
                box_list = list(range(len(indices)))
                left_index = []
                while len(box_list) > 0:
                    l = box_list[0]
                    remove_list = [l]
                    for b in range(1, len(box_list)):
                        iou = single_IOU(box[indices[l]], box[indices[box_list[b]]], True, True)
                        if iou > IOU_thresh:
                            remove_list.append(box_list[b])
                    for r in remove_list:
                        box_list.remove(r)
                    left_index.append(l)
                left_indices.append(indices[left_index])
            if len(left_indices) == 0:
                return [], [], []
            left_indices, _ = torch.sort(torch.cat(left_indices))
            conf = (conf[left_indices])[:keep_num_postNMS]
            cls = (cls[left_indices])[:keep_num_postNMS]
            box = (box[left_indices])[:keep_num_postNMS]
            boxes.append(box)
            scores.append(conf)
            labels.append(cls)
        return boxes, scores, labels

    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
        class_criterion = nn.CrossEntropyLoss(reduction='sum')
        regr_criterion = nn.SmoothL1Loss(reduction='sum')

        bg_labels_idx = torch.nonzero(labels==0)[:, 0]
        nonbg_labels_idx = torch.nonzero(labels!=0)[:, 0]
        max_nonbg_sample_size = int(0.75*effective_batch)
        nonbg_sample_size = min(len(nonbg_labels_idx), max_nonbg_sample_size)
        bg_sample_size = effective_batch - nonbg_sample_size

        bg_sample_idx = bg_labels_idx[torch.randperm(len(bg_labels_idx))[:bg_sample_size]]
        nonbg_sample_idx = nonbg_labels_idx[torch.randperm(len(nonbg_labels_idx))[:nonbg_sample_size]]
        loss_class = class_criterion(
                    class_logits[torch.hstack([bg_sample_idx, nonbg_sample_idx])],
                    labels[torch.hstack([bg_sample_idx, nonbg_sample_idx])].squeeze(1))
        
        lab = labels[nonbg_sample_idx] - 1
        preds = torch.zeros_like(regression_targets[nonbg_sample_idx])
        t_preds = box_preds[nonbg_sample_idx].reshape(-1, 3, 4)
        for i in range(len(preds)):
            preds[i] = t_preds[i][lab[i]]
        loss_regr = regr_criterion(preds, regression_targets[nonbg_sample_idx])

        loss_class = loss_class / effective_batch
        loss_regr = loss_regr / effective_batch
        loss = loss_class + l*loss_regr

        return loss, loss_class, loss_regr

    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors, eval=False):
        intermediate_feature = self.intermediate(feature_vectors)
        box_pred = self.regressor(intermediate_feature)

        if not eval:
            class_logits = self.classifier(intermediate_feature)
        else:
            class_logits = self.classifier(intermediate_feature)
            class_logits = F.softmax(class_logits, dim=1)
        return class_logits, box_pred

    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list, proposals, P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        b = len(proposals)
        feature_vectors = []
        for i in range(b):
            feature = torch.zeros(len(proposals[i]), 256, P, P)
            for j in range(len(proposals[i])):
                w = proposals[i][j, 2] - proposals[i][j, 0]
                h = proposals[i][j, 3] - proposals[i][j, 1]
                k = torch.clamp(torch.floor(4 + torch.log2(torch.sqrt(w * h) / 224)).int() - 2, 0, 4)
                feat = fpn_feat_list[k][i].unsqueeze(0)
                scales = feat.shape[3] / self.img_size[1]
                box = [proposals[i][j].unsqueeze(0)]
                feature[j] = torchvision.ops.roi_align(feat, box, output_size=P, spatial_scale=scales)
            feature_vectors.append(feature)
        feature_vectors = torch.cat(feature_vectors, dim=0)
        feature_vectors = feature_vectors.view(len(feature_vectors), -1)
        assert feature_vectors.shape[1] == 256 * P * P
        assert len(feature_vectors.shape) == 2
        return feature_vectors.cuda()

    def preNMS(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50,
               IOU_thresh=0.5):
        b = len(proposals)
        boxes = []
        scores = []
        labels = []
        start = 0
        for i in range(b):
            cls = class_logits[start: start + len(proposals[i])]
            reg_box = box_regression[start: start + len(proposals[i])]
            start += len(proposals[i])

            # Confidence cutoff
            non_bg = torch.nonzero(cls[:, 0] < conf_thresh).squeeze(1)
            cls = cls[non_bg]
            reg_box = reg_box[non_bg]
            prop = proposals[i][non_bg]

            # get rid of all background predictions
            conf, cls = torch.max(cls, dim=1)
            nonbg_indices = torch.nonzero(cls).squeeze(1)
            conf = torch.index_select(conf, 0, nonbg_indices)
            cls = torch.index_select(cls, 0, nonbg_indices) - 1
            reg_box = torch.index_select(reg_box, 0, nonbg_indices)
            prop = torch.index_select(prop, 0, nonbg_indices)

            # decode the boxes
            reg_cls = torch.zeros_like(prop)
            reg_box = reg_box.reshape(-1, 3, 4)
            for j in range(len(reg_cls)):
                reg_cls[j] = reg_box[j, cls[j]]
            box = output_decodingd(reg_cls, prop)  # x1 y1 x2 y2 type

            # Cutoff pre NMS
            conf = conf[:20]
            cls = cls[:20]
            box = box[:20]
            boxes.append(box)
            scores.append(conf)
            labels.append(cls)
        return boxes, scores, labels

    def MultiScaleRoiAlign_mask(self, fpn_feat_list, proposals, P=14):
        b = len(proposals)
        feature_vectors = []
        for i in range(b):
            feature = torch.zeros(len(proposals[i]), 256, P, P)
            for j in range(len(proposals[i])):
                w = proposals[i][j, 2] - proposals[i][j, 0]
                h = proposals[i][j, 3] - proposals[i][j, 1]
                k = torch.clamp(torch.floor(4 + torch.log2(torch.sqrt(w * h) / 224)).int() - 2, 0)
                feat = fpn_feat_list[k][i].unsqueeze(0)
                scales = feat.shape[3] / self.img_size[1]
                box = [proposals[i][j].unsqueeze(0)]
                feature[j] = torchvision.ops.roi_align(feat, box, output_size=P, spatial_scale=scales)
            feature_vectors.append(feature)
        feature_vectors = torch.cat(feature_vectors, dim=0)
        # feature_vectors = feature_vectors.view(len(feature_vectors), -1)
        assert feature_vectors.shape[-1] == P
        assert feature_vectors.shape[1] == 256
        assert len(feature_vectors.shape) == 4
        return feature_vectors.cuda()

    def renew_proposals(self, proposals, box_pred, class_logits):
        # proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
        # return: list: len(bz) {per_image_proposals, int}
        b = len(proposals)
        start = 0
        boxes = []

        for i in range(b):
            cls = class_logits[start: start + len(proposals[i])]
            reg_box = box_pred[start: start + len(proposals[i])]
            prop = proposals[i]
            start += len(proposals[i])

            _, cls = torch.max(cls, dim=1)
            cls = cls - 1  # 0, 1, 2
            # decode the boxes
            # import pdb
            # pdb.set_trace()
            reg_cls = torch.zeros_like(prop)
            reg_box = reg_box.reshape(-1, 3, 4)
            for j in range(len(reg_cls)):
                reg_cls[j] = reg_box[j, cls[j]]
            box = output_decodingd(reg_cls, prop)  # x1 y1 x2 y2 type
            box_num = len(box)
            new_box = torch.zeros_like(box)
            # print(box[:, 0].shape)
            new_box[:, 0] = torch.max(box[:, 0], torch.zeros((1, len(box))).to(self.device))
            new_box[:, 1] = torch.max(box[:, 1], torch.zeros((1, len(box))).to(self.device))
            new_box[:, 2] = torch.min(box[:, 2], 1088 * torch.ones((1, len(box))).to(self.device))
            new_box[:, 3] = torch.min(box[:, 3], 800 * torch.ones((1, len(box))).to(self.device))
            boxes.append(new_box)
        return boxes