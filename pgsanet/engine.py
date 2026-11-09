import json
import random
import numpy as np
import torch
from .metrics import SegmentationMetrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_config(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_checkpoint(path, device):
    # Only load checkpoints produced by you or a trusted source.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Local validation environment: torch 1.10.
        return torch.load(path, map_location=device)


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    metric = SegmentationMetrics(model.head[-1].out_channels)
    loss_total, samples = 0., 0
    with torch.set_grad_enabled(optimizer is not None):
        for image, target, prior in loader:
            image, target, prior = image.to(device), target.to(device), prior.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            result = model(image, prior)
            loss, _ = criterion(result, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
            metric.update(result['logits'].argmax(1), target)
            loss_total += float(loss.detach())*len(image)
            samples += len(image)
    result = metric.compute()
    result['loss'] = loss_total/max(samples, 1)
    return result
