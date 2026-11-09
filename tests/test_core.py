import numpy as np
import json
import pytest
import torch
from PIL import Image
from pgsanet import PGSANet
from pgsanet.model import SDCB, PAFE
from pgsanet.losses import CompositeLoss
from pgsanet.metrics import SegmentationMetrics
from pgsanet.data import decode_mask, PALETTES, SonarDataset, check_disjoint

torch.set_num_threads(2)


@pytest.mark.parametrize('classes,shape', [(5, (32, 32)), (9, (33, 35))])
def test_forward_backward(classes, shape):
    torch.manual_seed(7)
    model = PGSANet(classes, widths=(4, 8, 16), windows=(3, 5), attention_grid=4)
    result = model(torch.rand(2, 1, *shape), torch.rand(2, 3, *shape))
    assert result['logits'].shape == (2, classes, *shape)
    target = torch.randint(classes, (2, *shape))
    target[:, 0] = 255
    loss, _ = CompositeLoss(torch.ones(classes))(result, target)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.encoder[0][1].weight.grad.abs().sum() > 0
    assert model.pafe.route.weight.grad.abs().sum() > 0


def test_frequency_partition():
    block = SDCB(2)
    assert torch.equal(block.masks.sum(0), torch.ones(3, 3))
    spectrum = torch.fft.fft2(block.weight)
    restored = sum(torch.fft.ifft2(spectrum*m).real for m in block.masks)
    assert torch.allclose(restored, block.weight, atol=1e-6)


def test_cross_attention_chunking():
    torch.manual_seed(1)
    block = PAFE(4, attention_grid=0, chunk=5).eval()
    x, prior = torch.rand(1, 4, 4, 5), torch.rand(1, 3, 4, 5)
    with torch.no_grad():
        a = block(x, prior)
        block.chunk = 100
        b = block(x, prior)
    assert torch.allclose(a, b, atol=1e-6)


def test_metrics():
    metric = SegmentationMetrics(3)
    metric.update(torch.tensor([0, 1, 1, 1, 2]), torch.tensor([0, 0, 1, 1, 255]))
    result = metric.compute()
    assert result['IoU'][:2] == pytest.approx([.5, 2/3])
    assert result['IoU'][2] is None
    assert result['mIoU'] == pytest.approx(7/12)
    assert result['mPA'] == pytest.approx(.75)


@pytest.mark.parametrize('dataset', ['sss', 'sas'])
def test_palette(tmp_path, dataset):
    image = np.asarray(PALETTES[dataset], dtype=np.uint8)[None]
    path = tmp_path/'mask.png'
    Image.fromarray(image).save(path)
    assert decode_mask(path, dataset, 'rgb').tolist() == [list(range(len(image[0])))]
    Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError):
        decode_mask(path, dataset, 'rgb')


def test_ignored_loss():
    logits = torch.randn(1, 5, 8, 8, requires_grad=True)
    loss, _ = CompositeLoss(torch.ones(5))({'logits': logits, 'physics': None}, torch.full((1, 8, 8), 255))
    loss.backward()
    assert loss.item() == 0
    assert logits.grad.abs().sum() == 0


def test_external_prior_and_split_checks(tmp_path):
    Image.fromarray(np.zeros((17, 19), dtype=np.uint8)).save(tmp_path/'image.png')
    Image.fromarray(np.zeros((17, 19), dtype=np.uint8)).save(tmp_path/'mask.png')
    np.save(str(tmp_path/'prior.npy'), np.ones((3, 17, 19), dtype=np.float32))
    rows = [{'image': 'image.png', 'mask': 'mask.png', 'prior': 'prior.npy'}]
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(rows))
    cfg = {'dataset': 'sss', 'label_format': 'index', 'prior_mode': 'external',
           'image_size': 32, 'model': {'in_channels': 1, 'prior_channels': 3}}
    dataset = SonarDataset(path, cfg, augment=True)
    image, target, prior = dataset[0]
    assert image.shape == (1, 32, 32) and target.shape == (32, 32)
    assert torch.allclose(prior, torch.ones(3, 32, 32), atol=1e-6)
    with pytest.raises(ValueError):
        check_disjoint(dataset, dataset)
    np.save(str(tmp_path/'prior.npy'), np.zeros((17, 19, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        dataset[0]


def test_default_configuration_and_checkpoint(tmp_path):
    from pathlib import Path
    cfg = json.loads((Path(__file__).parents[1]/'configs/sss.json').read_text())
    model = PGSANet(**cfg['model']).eval()
    x, p = torch.rand(1, 1, 64, 64), torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(x, p)['logits']
    path = tmp_path/'weights.pt'
    torch.save(model.state_dict(), path)
    other = PGSANet(**cfg['model']).eval()
    other.load_state_dict(torch.load(path))
    with torch.no_grad():
        assert torch.equal(expected, other(x, p)['logits'])
