import torch
from torch import nn
from torch.nn import functional as F
from .model import sobel


class CompositeLoss(nn.Module):
    def __init__(self, class_weights, weights=(1., 1., .1), ignore_index=255):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.weights, self.ignore_index = weights, ignore_index

    def forward(self, result, target):
        logits = result['logits']
        valid = target != self.ignore_index
        zero = logits.sum()*0
        if not valid.any():
            return zero, {'wce': zero, 'boundary': zero, 'physics': zero}
        ce = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index, reduction='none')
        # Eq. 30 sum, normalized by the number of valid pixels for batch-size independence.
        ce = ce[valid].mean()
        probabilities = logits.softmax(1)
        labels = F.one_hot(target.masked_fill(~valid, 0), logits.shape[1]).permute(0, 3, 1, 2).float()
        px, py = sobel(probabilities)
        tx, ty = sobel(labels)
        # Ignore entire Sobel neighborhoods touching ignored pixels, and image padding.
        interior = F.avg_pool2d(valid[:, None].float(), 3, 1, 1)[:, 0] == 1
        bp, bt = px.abs()+py.abs(), tx.abs()+ty.abs()
        error = (bp-bt).square().sum(1)
        boundary = error[interior].mean() if interior.any() else zero
        aligned = result['physics']
        if self.weights[2] and aligned is None:
            raise ValueError('Physics loss enabled but no aligned physical prior')
        physics = zero if aligned is None else (probabilities-aligned).square().sum(1)[valid].mean()
        terms = {'wce': ce, 'boundary': boundary, 'physics': physics}
        return sum(w*l for w, l in zip(self.weights, terms.values())), terms
