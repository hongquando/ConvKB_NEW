import torch
import torch.nn as nn


class CustomTripletMarginLoss(nn.Module):
    """Triplet loss function.
    Based on: http://docs.chainer.org/en/stable/_modules/chainer/functions/loss/triplet.html
    """

    def __init__(self, margin):
        super(CustomTripletMarginLoss, self).__init__()
        self.margin = margin

    def forward(self, positive, negative):
        dist = torch.sum(positive ** 2 - negative ** 2, dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)  # maximum between 'dist' and 0.0
        loss = torch.mean(dist_hinge)
        return loss