import numpy as np
from functools import partial
import torch
import torch.nn.functional as F

import SimpleITK as sitk
from SimpleITK import GetArrayViewFromImage as ArrayView
"""
3D
"""
class DiceCoefficient:
    """Computes Dice Coefficient.
    Generalized to multiple channels by computing per-channel Dice Score
    (as described in https://arxiv.org/pdf/1707.03237.pdf) and then simply taking the average.
    Input is expected to be probabilities instead of logits.
    This metric is mostly useful when channels contain the same semantic class (e.g. affinities computed with different offsets).
    DO NOT USE this metric when training with DiceLoss, otherwise the results will be biased towards the loss.
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target):
        # Average across channels in order to get the final score
        return torch.mean(compute_per_channel_dice(input, target, epsilon=self.epsilon))

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
    Args:
         input (torch.Tensor): NxCxSpatial input tensor(DxHxW)
         target (torch.Tensor): NxCxSpatial target tensor(DxHxW)
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"
    
    input = (input > 0.5).float()
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(dim=(2,3,4))
    denominator = (input + target).sum(dim=(2,3,4))
    res = (2 * intersect + epsilon) / (denominator + epsilon)

    return res.mean(0)


class DiceRegion:
    """Computes Dice Coefficient of Region.
    label 1 : necrosis/nonenhancing tumor
    label 2 : edema
    label 3 : enhancing tumor(Originally 4)
    WT - label 1 + 2 + 3
    TC - label 1 + 3
    EC(ET) - label 3
    Args:
         input (torch.Tensor): NxCxSpatial input tensor(DxHxW), a result of Sigmoid or Softmax function.
         target (torch.Tensor): NxCxSpatial target tensor(DxHxW)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon

    def __call__(self, input, target, region='WT', mode='sigmoid', epsilon=1e-6):
        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        
        if mode == 'softmax':
            # softmax
            input = torch.argmax(input, dim=1)
            target = torch.argmax(target, dim=1)

            input_roi = input > 0
            target_roi = target > 0

            if region == 'TC':
                input_roi = input_roi*(input != 2)
                target_roi = target_roi*(target != 2)
            elif region == 'EC':
                input_roi = (input == 3)
                target_roi = (target == 3)
        elif mode == 'sigmoid':
            # sigmoid
            input = (input > 0.5)

            if region == 'WT':
                input_roi = input[:,0]
                target_roi = target[:,0]
            elif region == 'TC':
                input_roi = input[:,1]
                target_roi = target[:,1]
            elif region == 'EC':
                input_roi = input[:,2]
                target_roi = target[:,2]

        # common
        input_roi = input_roi.float()
        target_roi = target_roi.float()

        intersect = (input_roi * target_roi).sum(dim=(1,2,3))
        denominator = (input_roi + target_roi).sum(dim=(1,2,3))
        res = (2 * intersect + epsilon) / (denominator + epsilon)
        
        return res.mean(0)
    
def getHausdorff(input, target, mode='sigmoid'):
    """computes 95% Hausdorff Distance.
    """
    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False)
    distance_map3 = partial(sitk.SignedDanielssonDistanceMap, insideIsPositive=False, squaredDistance=False)
    
    if mode == 'softmax':
        input = torch.argmax(input, dim=1)
        target = torch.argmax(target, dim=1)
        
        wt_input = (input > 0).int()
        wt_target = (target > 0).int()        

        tc_input = wt_input*(input != 2)
        tc_target = wt_target*(target != 2)
        
        ec_input = (input == 3)
        ec_target = (target == 3)
        
        input = torch.stack([wt_input, tc_input, ec_input], 1)
        target = torch.stack([wt_target, tc_target, ec_target], 1)
            
    elif mode == 'sigmoid':
        input = (input > 0.5).int()
        
        
    target = target.int()
    result = []
    num_labels = len(input[0])
    for label in range(num_labels):
        if np.sum(input[0][label].numpy()) == 0 or np.sum(target[0][label].numpy()) == 0:
            result.append(0)
            continue
        seg = sitk.GetImageFromArray(np.transpose(input[0][label].numpy(), (2,1,0))) # from (W,H,D) to (D,H,W)
        gt = sitk.GetImageFromArray(np.transpose(target[0][label].numpy(), (2,1,0))) # from (W,H,D) to (D,H,W)
        seg_surface = sitk.LabelContour(seg, False)
        gt_surface = sitk.LabelContour(gt, False)

        ### Get distance map for contours (the distance map computes the minimum distances)
        seg_distance_map = sitk.Abs(distance_map(seg_surface))
        gt_distance_map = sitk.Abs(distance_map(gt_surface))

        ### Find the distances to surface points of the contour.  Calculate in both directions
        gt_to_seg = ArrayView(seg_distance_map)[ArrayView(gt_surface) == 1]
        seg_to_gt = ArrayView(gt_distance_map)[ArrayView(seg_surface) == 1]

        ### Find the 95% Distance for each direction
        result.append(max((np.percentile(seg_to_gt, 95), np.percentile(gt_to_seg, 95))))
        
    return result

    
class MeanIoU:
    """
    Computes IoU for each class separately and then averages over all classes.
    """

    def __init__(self, skip_channels=(), ignore_index=None, **kwargs):
        """
        :param skip_channels: list/tuple of channels to be ignored from the IoU computation
        :param ignore_index: id of the label to be ignored from IoU computation
        """
        self.ignore_index = ignore_index
        self.skip_channels = skip_channels

    def __call__(self, input, target):
        """
        :param input: 5D probability maps torch float tensor (NxCxDxHxW)
        :param target: 4D or 5D ground truth torch tensor. 4D (NxDxHxW) tensor will be expanded to 5D as one-hot
        :return: intersection over union averaged over all channels
        """
        assert input.dim() == 5

        n_classes = input.size()[1]

        if target.dim() == 4:
            target = expand_as_one_hot(target, C=n_classes, ignore_index=self.ignore_index)

        assert input.size() == target.size()

        per_batch_iou = []
        for _input, _target in zip(input, target):
            binary_prediction = self._binarize_predictions(_input, n_classes)

            if self.ignore_index is not None:
                # zero out ignore_index
                mask = _target == self.ignore_index
                binary_prediction[mask] = 0
                _target[mask] = 0

            # convert to uint8 just in case
            binary_prediction = binary_prediction.byte()
            _target = _target.byte()

            per_channel_iou = []
            for c in range(n_classes):
                if c in self.skip_channels:
                    continue

                per_channel_iou.append(self._jaccard_index(binary_prediction[c], _target[c]))

            assert per_channel_iou, "All channels were ignored from the computation"
            mean_iou = torch.mean(torch.tensor(per_channel_iou))
            per_batch_iou.append(mean_iou)

        return torch.mean(torch.tensor(per_batch_iou))
    
    def _binarize_predictions(self, input, n_classes):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if n_classes == 1:
            # for single channel input just threshold the probability map
            result = input > 0.5
            return result.long()

        _, max_index = torch.max(input, dim=0, keepdim=True)
        return torch.zeros_like(input, dtype=torch.uint8).scatter_(0, max_index, 1)

    def _jaccard_index(self, prediction, target):
        """
        Computes IoU for a given target and prediction tensors
        """
        return torch.sum(prediction & target).float() / torch.clamp(torch.sum(prediction | target).float(), min=1e-8)