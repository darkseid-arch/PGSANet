import torch


class SegmentationMetrics:
    def __init__(self, classes):
        self.confusion = torch.zeros(classes, classes, dtype=torch.int64)

    def update(self, prediction, target):
        c = len(self.confusion)
        prediction, target = prediction.detach().cpu().long(), target.detach().cpu().long()
        valid = (target >= 0) & (target < c)
        if ((prediction[valid] < 0) | (prediction[valid] >= c)).any():
            raise ValueError('Invalid prediction class')
        self.confusion += torch.bincount(c*target[valid]+prediction[valid], minlength=c*c).reshape(c, c)

    def compute(self):
        m = self.confusion.double()
        tp, support, predicted = m.diag(), m.sum(1), m.sum(0)
        union = support+predicted-tp
        iou = torch.where(union > 0, tp/union.clamp_min(1), torch.full_like(tp, float('nan')))
        recall = torch.where(support > 0, tp/support.clamp_min(1), torch.full_like(tp, float('nan')))
        def values(x):
            return [float(v) if torch.isfinite(v) else None for v in x]
        def mean(x):
            good = x[torch.isfinite(x)]
            return float(good.mean()) if len(good) else None
        return {'mIoU': mean(iou), 'mPA': mean(recall), 'IoU': values(iou),
                'class_accuracy': values(recall), 'confusion_matrix': self.confusion.tolist()}
