import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
from dataset import *
from utils import *
import torchvision


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
                                    stride=[4, 8, 16, 32, 64])
                 ):
        ######################################
        # TODO initialize RPN
        #######################################

    # Forward each level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
    def forward(self, X):

        return logits, bbox_regs

    # Forward a single level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       feature: (bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logit: (bz,1*num_acnhors,grid_size[0],grid_size[1])
    #       bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])
    def forward_single(self, feature):

        return logit, bbox_reg


    # This function creates the anchor boxes for all FPN level
    # Input:
    #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
    #       scale:        list:len(FPN)
    #       grid_size:    list:len(FPN){tuple:len(2)}
    #       stride:        list:len(FPN)
    # Output:
    #       anchors_list: list:len(FPN){(grid_size[0]*grid_size[1]*num_anchors,4)}
    def create_anchors(self, aspect_ratio, scale, grid_size, stride):

        return anchors_list



    # This function creates the anchor boxes for one FPN level
    # Input:
    #      aspect_ratio: list:len(number_of_aspect_ratios)
    #      scale: scalar
    #      grid_size: tuple:len(2)
    #      stride: scalar
    # Output:
    #       anchors: (grid_size[0]*grid_size[1]*num_acnhors,4)
    def create_anchors_single(self, aspect_ratio, scale, grid_sizes, stride):

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

        return ground, ground_coord

    # This function create the ground truth for one image for all the FPN levels
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset)
    #       grid_size:   list:len(FPN){tuple:len(2)}
    #       anchor_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
    # Output:
    #       ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
    #       ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
    def create_ground_truth(self, bboxes, index, grid_sizes, anchors, image_size):


        return ground_clas, ground_coord

    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):

        # torch.nn.BCELoss()
        # TODO compute classifier's loss

        return loss, sum_count

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        # torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss

        return loss, sum_count

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


        return loss, loss_c, loss_r


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

        return nms_clas_list, nms_prebox_list

    # Post process the output for one image
    # Input:
    #      mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
    #      mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):

        return nms_clas, nms_prebox

    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self, clas, prebox, thresh):

        return nms_clas, nms_prebox


if __name__ == "__main__":



