from copy import deepcopy
import torch
import torch.nn.functional as F
# from torchvision.transforms import Resize
from torch import nn
from utils import *

class MaskHead(torch.nn.Module):
    def __init__(self,Classes=3,P=14):
        self.C=Classes
        self.P=P
        # TODO initialize MaskHead
        self.net = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding_mode='same'),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, self.C, 1),
            nn.Sigmoid()
        )

    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)   ——[from rpn]
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       masks: list:len(bz){(n_obj,800,1088)}
    #       IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
    def preprocess_ground_truth_creation(self, class_logits, box_regression, proposals, gt_labels, bbox ,masks , IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):
        conf_thresh = 0.5 # this parameter is needed ?
        b = len(proposals)
        boxes = []
        scores = []
        labels = []
        gt_masks = []
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
            cls = torch.index_select(cls, 0, nonbg_indices) - 1  # 0, 1, 2
            reg_box = torch.index_select(reg_box, 0, nonbg_indices)
            prop = torch.index_select(prop, 0, nonbg_indices)

            # decode the boxes
            reg_cls = torch.zeros_like(prop)
            reg_box = reg_box.reshape(-1, 3, 4)
            for j in range(len(reg_cls)):
                reg_cls[j] = reg_box[j, cls[j]]
            box = output_decoding(reg_cls, prop)  # x1 y1 x2 y2 type

            # Cutoff pre NMS
            conf = conf[:keep_num_preNMS]
            cls = cls[:keep_num_preNMS]
            box = box[:keep_num_preNMS]

            left_indices = []
            gt_masks_image = []
            for c in range(self.C):
                indices = torch.nonzero(cls == c).squeeze(1)
                if len(indices) == 0:
                    continue
                box_list = list(range(len(indices)))
                box_indices = deepcopy(indices)
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
                        box_indices = box_indices[torch.tensor(box_list)]
                    left_index.append(l)
                    # assign gt mask to l
                    left_box = box[indices[l]]
                    gt_indices = torch.nonzero(gt_labels[i]==c).squeeze(1)
                    if len(gt_indices) == 0:
                        gt_masks_box = torch.zeros((self.P, self.P))
                    else:
                        _, max_iou_bbox_idx = max_IOU(left_box, masks[i][gt_indices])
                        gt_masks_box = masks[i][gt_indices[max_iou_bbox_idx]]
                        gt_masks_box = gt_masks_box[left_box[0]:left_box[2], left_box[1]:left_box[3]]
                        gt_masks_box = F.interpolate(gt_masks_box, size=(self.P, self.P))
                    gt_masks_image.append(gt_masks_box)
                left_indices.append(indices[left_index])
            if len(left_indices) == 0:
                return [], [], []
            left_indices, left_indices_order = torch.sort(torch.cat(left_indices))
            conf = (conf[left_indices])[:keep_num_postNMS]
            cls = (cls[left_indices])[:keep_num_postNMS]
            box = (box[left_indices])[:keep_num_postNMS]
            gt_masks_image = self.flatten_inputs(gt_masks_image)[left_indices_order]
            gt_masks_image = gt_masks_image[:keep_num_postNMS]
            boxes.append(box)
            scores.append(conf)
            labels.append(cls)
            gt_masks.append(gt_masks_image)
        return boxes, scores, labels, gt_masks

    # general function that takes the input list of tensors and concatenates them along the first tensor dimension
    # Input:
    #      input_list: list:len(bz){(dim1,?)}
    # Output:
    #      output_tensor: (sum_of_dim1,?)
    def flatten_inputs(self,input_list):
        output_tensor = input_list[0]
        for input_tensor in input_list[1:]:
            output_tensor = torch.cat((output_tensor, input_tensor), dim=0)
        return output_tensor


    # This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
    # back to the original image size
    # Use the regressed boxes to distinguish between the images
    # Input:
    #       masks_outputs: (total_boxes,C,2*P,2*P)
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       image_size: tuple:len(2)
    # Output:
    #       projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
    def postprocess_mask(self, masks_outputs, boxes, labels, image_size=(800,1088)):
        start = 0
        projected_masks = []
        for img_idx in range(len(boxes)):
            img_projected_masks = []
            boxes_num = boxes[img_idx].shape[0]
            img_bbox = boxes[img_idx]
            img_labels = labels[img_idx]
            img_masks = masks_outputs[start:start+boxes_num]
            start += boxes_num
            for bbox, idx in zip(img_bbox, range(boxes_num)):
                mask = img_masks[idx][img_labels[idx]]
                mask = F.interpolate(mask, size=(bbox[2]-bbox[0], bbox[3]-bbox[1]))
                project_mask = torch.zeros((image_size[0],image_size[1]))
                project_mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = mask
                img_projected_masks.append(project_mask)
            projected_masks.append(self.flatten_inputs(img_projected_masks))
        return projected_masks

    # Compute the total loss of the Mask Head
    # Input:
    #      mask_output: (total_boxes,C,2*P,2*P)
    #      labels: (total_boxes)
    #      gt_masks: (total_boxes,2*P,2*P)
    # Output:
    #      mask_loss
    def compute_loss(self,mask_output,labels,gt_masks):
        mask_loss = 0
        loss = nn.BCELoss(reduction='sum')
        for c in range(self.C):
            cls = c+1
            cls_idx = torch.nonzero(labels==cls).squeeze(1)
            cls_masks = mask_output[cls_idx][:, c, ...]
            gt_cls_masks = gt_masks[cls_idx]
            mask_loss += loss(cls_masks, gt_cls_masks)
        return mask_loss

    # Forward the pooled feature map Mask Head
    # Input:
    #        features: (total_boxes, 256,P,P)
    # Outputs:
    #        mask_outputs: (total_boxes,C,2*P,2*P)
    def forward(self, features):
        mask_outputs = self.net(features)
        return mask_outputs

if __name__ == '__main__':
