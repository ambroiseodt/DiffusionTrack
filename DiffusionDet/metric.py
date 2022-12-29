import torch
import torchvision
import pickle

from diffusiondet.util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

path_pred = "data_eval/MOT_16_02_000001_swin_8.pkl"
path_gt = "data_eval/gt.txt"
def jaccard(path_pred, path_gt, frame):
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Boxes should be in (x1, y1, x2, y2) format with 0 <= x1 < x2 and 0 <= y1 < y2.
    """

    # Recover predicted boxes
    pred = torch.Tensor(pickle.load(open(path_pred,'rb')))
    pred = box_cxcywh_to_xyxy(pred).squeeze(dim=0)

    # Open ground truth
    with open(path_gt) as f:
        lines = f.read()
        target = lines.split('\n',frame)[0]

    # Recover ground truth boxes
    target = target.split(',')
    target = list(map(int,target))
    target = torch.Tensor(target[2:-3])
    target = box_cxcywh_to_xyxy(target)
    jaccard_index = torchvision.ops.box_iou(pred[0].unsqueeze(dim=0), target.unsqueeze(dim=0))

    return jaccard_index

jaccard_index = jaccard(path_pred,path_gt, 1)
print(jaccard_index.size())