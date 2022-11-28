import medpy.metric.binary as medpyMetrics
import numpy as np
import math
import torch
import torch.nn as nn


#
def toOrignalCategoryOneHot(labels, classes=5):
    shape = labels.shape
    out = torch.zeros([shape[0], shape[1], shape[2], shape[3], classes])
    for i in range(classes):
        out[:, :, :, :, i] = (labels == i)

    out = out.permute(0, 4, 1, 2, 3)
    return out

def softDice(pred, target, smoothing=1, nonSquared=False):
    intersection = (pred * target).sum(dim=(2, 3, 4))
    if nonSquared:
        union = (pred).sum() + (target).sum()
    else:
        union = (pred * pred).sum(dim=(2, 3, 4)) + (target * target).sum(dim=(2, 3, 4))
    dice = (2 * intersection + smoothing) / (union + smoothing)

    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])

    return dice.mean()

def dice(pred, target):
    predBin = (pred > 0.5).float()
    return softDice(predBin, target, 0, True).item()

def diceLoss(pred, target, nonSquared=False):
    return 1 - softDice(pred, target, nonSquared=nonSquared)

def bratsDiceLoss(outputs, labels, nonSquared=False):

    #bring outputs into correct shape
    wt, tc, et = outputs.chunk(3, dim=1)
    s = wt.shape
    wt = wt.view(s[0], s[2], s[3], s[4])
    tc = tc.view(s[0], s[2], s[3], s[4])
    et = et.view(s[0], s[2], s[3], s[4])

    # bring masks into correct shape
    wtMask, tcMask, etMask = labels.chunk(3, dim=1)
    s = wtMask.shape
    wtMask = wtMask.view(s[0], s[2], s[3], s[4])
    tcMask = tcMask.view(s[0], s[2], s[3], s[4])
    etMask = etMask.view(s[0], s[2], s[3], s[4])

    #calculate losses
    wtLoss = diceLoss(wt, wtMask, nonSquared=nonSquared)
    tcLoss = diceLoss(tc, tcMask, nonSquared=nonSquared)
    etLoss = diceLoss(et, etMask, nonSquared=nonSquared)
    return (wtLoss + tcLoss + etLoss) / 5

def bratsDiceLossOriginal5(outputs, labels, nonSquared=False):
    outputList = list(outputs.chunk(5, dim=1))
    labels = toOrignalCategoryOneHot(labels)
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred, target in zip(outputList, labelsList):
        pred = pred.cuda()
        target = target.cuda()
        # print("pred shape", pred.shape, print("target shape", target.shape))
        totalLoss = totalLoss + diceLoss(pred, target, nonSquared=nonSquared)
    return totalLoss.cuda()

def CrossEntropyDiceLoss(outputs, labels, nonSqured=False):
    CrossEntropyLoss = nn.CrossEntropyLoss(outputs, labels)
    DiceLoss = bratsDiceLossOriginal5(outputs, labels)
    totalLoss = CrossEntropyLoss + DiceLoss
    return totalLoss.cuda()

def ConsensusDiceLoss(pred1, pred2, target, smoothing=1, nonSquared=False):
    intersection = (pred1 * pred2 * target).sum(dim=(2, 3, 4))
    if nonSquared:
        union = (pred1).sum() + (pred2).sum() + (target).sum()
    else:
        union = (pred1 * pred1).sum(dim=(2, 3, 4)) + (pred2 * pred2).sum(dim=(2, 3, 4)) + (target * target).sum(dim=(2, 3, 4))
    consensusDice = (3 * intersection + smoothing) / (union + smoothing)

    # fix nans
    consensusDice[consensusDice != consensusDice] = consensusDice.new_tensor([1.0])

    return 1 - consensusDice.mean()


# This function is used for multual leraing for brain tumor segmentation
def bratsConsensusDiceLoss(outputs1, outputs2, labels, nonSquared=False):
    output1List = list(outputs1.chunk(5, dim=1))
    output2List = list(outputs2.chunk(5, dim=1))
    labels = toOrignalCategoryOneHot(labels)
    labelsList = list(labels.chunk(5, dim=1))
    totalLoss = 0
    for pred1, pred2, target in zip(output1List, output2List, labelsList):
        pred1 = pred1.cuda()
        pred2 = pred2.cuda()
        target = target.cuda()
        # print("pred shape", pred.shape, print("target shape", target.shape))
        totalLoss = totalLoss + ConsensusDiceLoss(pred1, pred2, target, nonSquared=nonSquared)
    return totalLoss.cuda()


def sensitivity(pred, target):
    predBin = (pred > 0.5).float()
    intersection = (predBin * target).sum()
    allPositive = target.sum()

    # special case for zero positives
    if allPositive == 0:
        return 1.0
    return (intersection / allPositive).item()

def specificity(pred, target):
    predBinInv = (pred <= 0.5).float()
    targetInv = (target == 0).float()
    intersection = (predBinInv * targetInv).sum()
    allNegative = targetInv.sum()
    return (intersection / allNegative).item()

def getHd95(pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    if np.count_nonzero(pred) > 0 and np.count_nonzero(target):
        surDist1 = medpyMetrics.__surface_distances(pred, target)
        surDist2 = medpyMetrics.__surface_distances(target, pred)
        hd95 = np.percentile(np.hstack((surDist1, surDist2)), 95)
        return hd95
    else:
        # Edge cases that medpy cannot handle
        return -1

def getWTMask(labels):
    return (labels != 0).float()

def getTCMask(labels):
    return ((labels != 0) * (labels != 2)).float() #We use multiplication as AND

def getETMask(labels):
    return (labels == 4).float()

def predictionOverlapLoss(pred1, pred2, smoothing=1, nonSquared=False):
    intersection = (pred1 * pred2).sum(dim=(2, 3, 4))
    if nonSquared:
        union = (pred1).sum() + (pred2).sum()
    else:
        union = (pred1 * pred1).sum(dim=(2, 3, 4)) + (pred2 * pred2).sum(dim=(2, 3, 4))
    predictionOverlap = (2 * intersection + smoothing) / (union + smoothing)

    # fix nans
    predictionOverlap[predictionOverlap != predictionOverlap] = predictionOverlap.new_tensor([1.0])

    return 1 - predictionOverlap.mean()


def bratsPredictionOverlap(outputs1, outputs2, nonSquared=False):
    output1List = list(outputs1.chunk(5, dim=1))
    output2List = list(outputs2.chunk(5, dim=1))
    totalLoss = 0
    for pred1, pred2 in zip(output1List, output2List):
        pred1 = pred1.cuda()
        pred2 = pred2.cuda()
        # print("pred shape", pred.shape, print("target shape", target.shape))
        totalLoss = totalLoss + predictionOverlapLoss(pred1, pred2, nonSquared=nonSquared)
    return totalLoss.cuda()


# Code was adapted and modified from https://github.com/wolny/pytorch-3dunet/blob/master/pytorch3dunet/unet3d/losses.py
def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, sigmoid_normalization=True):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.classes = None
        self.skip_index_after = None
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def skip_target_channels(self, target, index):
        """
        Assuming dim 1 is the classes dim , it skips all the indexes after the desired class
        """
        assert index >= 2
        return target[:, 0:index, ...]

    def forward(self, input, target):
        """
        Expand to one hot added extra for consistency reasons
        """
        target = toOrignalCategoryOneHot(target.long(), self.classes)

        assert input.dim() == target.dim() == 5 ,"'input' and 'target' have different number of dims"

        if self.skip_index_after is not None:
            target = self.skip_target_channels(target, self.skip_index_after)
        # print(input.size(),target.size())
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)
        loss = (1. - torch.mean(per_channel_dice))
        per_channel_dice = per_channel_dice.detach().cpu().numpy()

        # average Dice score across all channels/classes
        return loss, per_channel_dice

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, classes=4, sigmoid_normalization=True, skip_index_after=None, epsilon=1e-6,):
        super().__init__(weight=None, sigmoid_normalization=sigmoid_normalization)
        self.epsilon = epsilon
        self.classes = classes
        if skip_index_after is not None:
            self.skip_index_after = skip_index_after

    def dice(self, input, target, weight):
        assert input.size() == target.size()
        input = flatten(input).cuda()
        target = flatten(target)
        target = target.float().cuda()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon).cuda()
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())



if __name__ == '__main__':
    outputs1 = torch.rand((10, 5, 25, 25, 25)).cuda()
    outputs2 = torch.rand((10, 5, 25, 25, 25)).cuda()
    labels = torch.randint(0, 5, (10, 25, 25, 25)).cuda()
    print(labels.shape)
    # loss = bratsConsensusDiceLoss(outputs1, outputs2, labels)
    # print(loss)


    labels = toOrignalCategoryOneHot(labels)
    # output_list = list(outputs.chunk(5, dim=1))
    # print(len(output_list), output_list)
    # print(labels.shape)
    # loss = bratsDiceLossOriginal5(outputs1, labels)
    # o = outputs1.sum(dim=(2, 3, 4))
    # print(loss)
    # print(o.shape)

    criterion = GeneralizedDiceLoss(classes=5, sigmoid_normalization=False)

    print(criterion(outputs1, outputs2))

