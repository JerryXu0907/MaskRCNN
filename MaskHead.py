import torch
import torch.nn.functional as F
from torch import nn
from utils import *

class MaskHead(torch.nn.Module):
    def __init__(self,Classes=3,P=14):
        self.C=Classes
        self.P=P
        # TODO initialize MaskHead


    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
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
    def preprocess_ground_truth_creation(self, class_logits, box_regression, gt_labels,bbox ,masks , IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):

        return boxes, scores, labels, gt_masks

    # general function that takes the input list of tensors and concatenates them along the first tensor dimension
    # Input:
    #      input_list: list:len(bz){(dim1,?)}
    # Output:
    #      output_tensor: (sum_of_dim1,?)
    def flatten_inputs(self,input_list):

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

        return projected_masks




    # Compute the total loss of the Mask Head
    # Input:
    #      mask_output: (total_boxes,C,2*P,2*P)
    #      labels: (total_boxes)
    #      gt_masks: (total_boxes,2*P,2*P)
    # Output:
    #      mask_loss
    def compute_loss(self,mask_output,labels,gt_masks):

        return mask_loss



    # Forward the pooled feature map Mask Head
    # Input:
    #        features: (total_boxes, 256,P,P)
    # Outputs:
    #        mask_outputs: (total_boxes,C,2*P,2*P)
    def forward(self, features):

        return mask_outputs

if __name__ == '__main__':
