"""Explicit manifests, strict label decoding, synchronized augmentation and priors."""
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

PALETTES = {
    'sss': [[0, 0, 128], [0, 128, 0], [128, 0, 0], [128, 128, 0], [0, 0, 0]],
    'sas': [[255, 0, 0], [55, 126, 184], [77, 175, 74], [152, 78, 168],
            [254, 254, 50], [125, 223, 215], [166, 86, 40], [247, 128, 191], [0, 0, 0]]}
NAMES = {'sss': ['Shadow', 'Water', 'Land', 'Mountain', 'Background'],
         'sas': ['Shadow', 'Sand (dark)', 'Sand (light)', 'Seagrass', 'Cobble/Rock',
                 'Unlabeled', 'Ripple (small)', 'Ripple (large)', 'Background']}


def decode_mask(path, dataset, label_format):
    with Image.open(path) as image:
        a = np.asarray(image.convert('RGB') if label_format == 'rgb' else image)
    classes = len(PALETTES[dataset])
    if label_format == 'index':
        if a.ndim != 2:
            raise ValueError('Indexed masks must have one channel')
        if not np.isin(a, list(range(classes))+[255]).all():
            raise ValueError('Unknown class index in ' + str(path))
        return torch.from_numpy(a.astype(np.int64))
    out = np.full(a.shape[:2], 255, dtype=np.int64)
    for index, color in enumerate(PALETTES[dataset]):
        out[(a == color).all(-1)] = index
    if (out == 255).any():
        raise ValueError('Unknown RGB label color in ' + str(path) + '; use exact Table I colors')
    return torch.from_numpy(out)


def load_image(path, channels):
    with Image.open(path) as image:
        image = image.convert('L' if channels == 1 else 'RGB')
        a = np.asarray(image, dtype=np.float32)/255.
    if a.ndim == 2:
        a = a[..., None]
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


def demo_prior(image):
    """Image-only proxies, NOT calibrated acoustic physics. Never uses labels."""
    gray = image.mean(0, keepdim=True)
    return torch.cat([1-gray, (gray < .2).float(), gray], 0)


def load_prior(path, image, channels):
    array = np.load(path, allow_pickle=False)
    if array.ndim != 3 or array.shape[0] != channels or tuple(array.shape[1:]) != tuple(image.shape[1:]):
        raise ValueError('Prior must be float CHW, spatially aligned with original image: ' + str(path))
    if not np.isfinite(array).all():
        raise ValueError('Non-finite prior: ' + str(path))
    return torch.from_numpy(array.astype(np.float32))


class SonarDataset(Dataset):
    def __init__(self, manifest, cfg, augment=False):
        self.path, self.cfg, self.augment = Path(manifest), cfg, augment
        self.rows = json.loads(self.path.read_text(encoding='utf-8'))
        if not isinstance(self.rows, list) or not self.rows:
            raise ValueError('Manifest must be a nonempty JSON list')
        ids = [str(self.resolve(r['image'])) for r in self.rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate images in manifest')

    def resolve(self, path):
        return (self.path.parent / path).resolve()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, cfg = self.rows[index], self.cfg
        image = load_image(self.resolve(row['image']), cfg['model']['in_channels'])
        target = decode_mask(self.resolve(row['mask']), cfg['dataset'], cfg['label_format'])
        if tuple(target.shape) != tuple(image.shape[1:]):
            raise ValueError('Image/mask size mismatch')
        if cfg['prior_mode'] == 'external':
            prior = load_prior(self.resolve(row['prior']), image, cfg['model']['prior_channels'])
        elif cfg['prior_mode'] == 'demo':
            if cfg['model']['prior_channels'] != 3:
                raise ValueError('Demo priors have exactly three channels')
            prior = demo_prior(image)
        else:
            raise ValueError('prior_mode must be external or demo')
        size = cfg['image_size']
        image = F.interpolate(image[None], (size, size), mode='bilinear', align_corners=False)[0]
        prior = F.interpolate(prior[None], (size, size), mode='bilinear', align_corners=False)[0]
        target = F.interpolate(target[None, None].float(), (size, size), mode='nearest')[0, 0].long()
        if self.augment:
            for axis in (1, 2):
                if random.random() < .5:
                    image, prior, target = image.flip(axis), prior.flip(axis), target.flip(axis-1)
            k = random.randrange(4)
            image, prior, target = image.rot90(k, (1, 2)), prior.rot90(k, (1, 2)), target.rot90(k, (0, 1))
            scale = random.uniform(.8, 1.2)
            theta = image.new_tensor([[scale, 0, 0], [0, scale, 0]])[None]
            grid = F.affine_grid(theta, (1, 1, size, size), align_corners=False)
            image = F.grid_sample(image[None], grid, align_corners=False, padding_mode='border')[0]
            prior = F.grid_sample(prior[None], grid, align_corners=False, padding_mode='border')[0]
            target = F.grid_sample(target[None, None].float(), grid, mode='nearest', align_corners=False, padding_mode='border')[0, 0].long()
        return image, target, prior


def check_disjoint(*datasets):
    seen = set()
    for dataset in datasets:
        current = {str(dataset.resolve(row['image'])) for row in dataset.rows}
        if seen & current:
            raise ValueError('Image leakage between split manifests')
        seen |= current


def class_weights(dataset):
    """Raw training labels only; never compute weights from validation/test statistics."""
    count = torch.zeros(len(PALETTES[dataset.cfg['dataset']]), dtype=torch.float64)
    for row in dataset.rows:
        mask = decode_mask(dataset.resolve(row['mask']), dataset.cfg['dataset'], dataset.cfg['label_format'])
        count += torch.bincount(mask[mask != 255], minlength=len(count))
    weight = torch.where(count > 0, 1/count.clamp_min(1), torch.zeros_like(count))
    if weight.sum() == 0:
        raise ValueError('No labeled training pixels')
    return (weight / weight.sum() * len(weight)).float(), count.tolist()
