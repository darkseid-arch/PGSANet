import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from pgsanet import PGSANet
from pgsanet.data import SonarDataset, NAMES
from pgsanet.engine import load_checkpoint, run_epoch
from pgsanet.losses import CompositeLoss


def main():
    p = argparse.ArgumentParser(description='Evaluate a trusted checkpoint on a held-out manifest')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', default='metrics.json')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    saved = load_checkpoint(args.checkpoint, args.device)
    cfg = saved['config']
    model = PGSANet(**cfg['model']).to(args.device)
    model.load_state_dict(saved['model'])
    dataset = SonarDataset(args.manifest, cfg)
    result = run_epoch(model, DataLoader(dataset, batch_size=cfg['batch_size']),
                       CompositeLoss(saved['class_weights'], cfg['loss_weights']).to(args.device), args.device)
    result['classes'], result['prior_mode'] = NAMES[cfg['dataset']], cfg['prior_mode']
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
