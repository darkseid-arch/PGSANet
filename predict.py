import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from pgsanet import PGSANet
from pgsanet.data import load_image, load_prior, demo_prior, PALETTES
from pgsanet.engine import load_checkpoint


def main():
    p = argparse.ArgumentParser(description='Predict class IDs and RGB mask at original image size')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--prior', help='External CHW .npy prior, required for external-mode checkpoints')
    p.add_argument('--output', default='predictions')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    saved = load_checkpoint(args.checkpoint, args.device)
    cfg = saved['config']
    model = PGSANet(**cfg['model']).to(args.device).eval()
    model.load_state_dict(saved['model'])
    image = load_image(args.image, cfg['model']['in_channels'])
    if cfg['prior_mode'] == 'external':
        if not args.prior:
            p.error('--prior is required by this checkpoint')
        prior = load_prior(args.prior, image, cfg['model']['prior_channels'])
    else:
        prior = demo_prior(image)
    size = (cfg['image_size'], cfg['image_size'])
    with torch.no_grad():
        result = model(F.interpolate(image[None], size, mode='bilinear', align_corners=False).to(args.device),
                       F.interpolate(prior[None], size, mode='bilinear', align_corners=False).to(args.device))
        pred = F.interpolate(result['logits'], image.shape[1:], mode='bilinear', align_corners=False).argmax(1)[0].cpu().numpy().astype('uint8')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    Image.fromarray(pred).save(out/(stem+'_ids.png'))
    Image.fromarray(np.asarray(PALETTES[cfg['dataset']], dtype=np.uint8)[pred]).save(out/(stem+'_color.png'))
    print('Saved predictions to ' + str(out))


if __name__ == '__main__':
    main()
