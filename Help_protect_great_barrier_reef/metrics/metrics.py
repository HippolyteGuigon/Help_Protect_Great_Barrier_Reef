import numpy as np 

def order_dicts(ground_truth: dict, pred: dict)->dict:
    """
    The goal of this function is
    to order the dictionnaries
    in terms of bounding boxes. Bounding
    boxes of same objects must be in the
    same order from one dictionnary to the
    other
    
    Arguments:
        -ground_truth: dict: The true localization 
        of a given object inside a given image
        -pred: dict: The prediction of the localization
        of a given object inside the image

    Returns:
        -ground_truth: dict: The sorted ground_truth
        dictionnary
        -pred: dict: The sorted pred dictionnary
    """

    barycenter_gt=[[(d["xmin"]+d["xmax"])/2, (d["ymin"]+d["ymax"])/2]
                    for d in ground_truth]
    barycenter_pred=[[(d["xmin"]+d["xmax"])/2, (d["ymin"]+d["ymax"])/2]
                    for d in pred]
    
def get_iou(ground_truth: dict, pred: dict)->float:
    """
    The goal of this function is to
    calculate the intersection over 
    union (IoU) of two given predictions
    
    Arguments:
        -ground_truth: The true localization 
        of a given object inside a given image
        -pred: The prediction of the localization
        of a given object inside the image
        
    Returns:
        -iou: float: The iou computed    
    """
    
    if (ground_truth and not pred) or (not ground_truth and pred):
        return 0 
    elif len(ground_truth.keys())==0 and len(pred.keys())==0:
        return 1

    ix1 = np.maximum(ground_truth["xmin"], pred["xmin"])
    iy1 = np.maximum(ground_truth["ymin"], pred["ymin"])
    ix2 = np.minimum(ground_truth["xmax"], pred["xmax"])
    iy2 = np.minimum(ground_truth["ymax"], pred["ymax"])
    
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
    
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth["ymax"] - ground_truth["ymin"] + 1
    gt_width = ground_truth["xmax"] - ground_truth["xmin"] + 1
     
    # Prediction dimensions.
    pd_height = pred["ymax"] - pred["ymin"] + 1
    pd_width = pred["xmax"] - pred["xmin"] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou