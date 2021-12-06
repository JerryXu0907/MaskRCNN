import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):

    return iou


# This function flattens the output of the network and the corresponding anchors
# in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
# the FPN levels from all the images into 2D matrices
# Each row correspond of the 2D matrices corresponds to a specific grid cell
# Input:
#       out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}
#       out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
#       anchors: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
# Output:
#       flatten_regr: (total_number_of_anchors*bz,4)
#       flatten_clas: (total_number_of_anchors*bz)
#       flatten_anchors: (total_number_of_anchors*bz,4)
def output_flattening(out_r, out_c, anchors):

    return flatten_regr, flatten_clas, flatten_anchors


# This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it returns the upper left and lower right corner of the bbox
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out, flatten_anchors, device='cpu'):

    return box


# This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
# a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
# Input:
#      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
#      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
#      P: scalar
# Output:
#      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
def MultiScaleRoiAlign( fpn_feat_list,proposals,P=7):
    #####################################
    # Here you can use torchvision.ops.RoIAlign check the docs
    #####################################

    return feature_vectors

