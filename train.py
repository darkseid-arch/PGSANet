"""Train using train/validation manifests; the test split is never selected on."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pgsanet import PGSANet
from pgsanet.data import SonarDataset, check_disjoint, class_weights
from pgsanet.engine import seed_all, read_config, load_checkpoint, run_epoch
from pgsanet.losses import CompositeLoss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True)
    parser.add_argument('--output', default='runs/experiment')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume')
    args = parser.parse_args()
    cfg = read_config(args.config)
    seed_all(cfg['seed'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out/'last.pt').exists():
        raise FileExistsError('Run already exists; use --resume or a new output directory')
    train, val = SonarDataset(args.train, cfg, True), SonarDataset(args.val, cfg)
    check_disjoint(train, val)
    weights, counts = class_weights(train)
    loaders = [DataLoader(d, batch_size=cfg['batch_size'], shuffle=i == 0, num_workers=0)
               for i, d in enumerate([train, val])]
    model = PGSANet(**cfg['model']).to(args.device)
    criterion = CompositeLoss(weights, cfg['loss_weights']).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['restart_period'])
    start, best, stale, multiplier = 0, -1., 0, 1.
    if args.resume:
        saved = load_checkpoint(args.resume, args.device)
        if saved['config'] != cfg:
            raise ValueError('Resume config must match saved config exactly')
        if saved['train_rows'] != train.rows or saved['val_rows'] != val.rows:
            raise ValueError('Resume manifests must match saved manifests')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        start, best, stale, multiplier = saved['epoch']+1, saved['best'], saved['stale'], saved['lr_multiplier']
        random.setstate(saved['rng_python'])
        np.random.set_state(saved['rng_numpy'])
        torch.set_rng_state(saved['rng_torch'].cpu())
        if torch.cuda.is_available() and saved['rng_cuda'] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['rng_cuda']])
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    (out/'training_metadata.json').write_text(json.dumps({'class_counts': counts,
        'class_weights': weights.tolist(), 'parameters': sum(p.numel() for p in model.parameters()),
        'torch': torch.__version__, 'prior_mode': cfg['prior_mode']}, indent=2), encoding='utf-8')
    if cfg['prior_mode'] == 'demo':
        print('DEMO image-derived proxies enabled: this is not calibrated physics or paper reproduction.')
    for epoch in range(start, cfg['epochs']):
        training = run_epoch(model, loaders[0], criterion, args.device, optimizer)
        validation = run_epoch(model, loaders[1], criterion, args.device)
        score = validation[cfg['monitor']]
        if score is None:
            raise ValueError('Validation set has no valid labeled pixels')
        improved = score > best
        if improved:
            best, stale = score, 0
        else:
            stale += 1
        if stale and stale % cfg['lr_patience'] == 0:
            multiplier *= .5
        scheduler.step(epoch+1)
        for group in optimizer.param_groups:
            group['lr'] *= multiplier
        record = {'epoch': epoch, 'train': training, 'val': validation,
                  'next_lr': optimizer.param_groups[0]['lr']}
        with (out/'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        print('epoch={} train_loss={:.5f} val_mIoU={:.5f}'.format(epoch+1, training['loss'], validation['mIoU']), flush=True)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'epoch': epoch, 'best': best, 'stale': stale,
                 'lr_multiplier': multiplier, 'config': cfg, 'class_weights': weights,
                 'train_rows': train.rows, 'val_rows': val.rows,
                 'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                 'rng_torch': torch.get_rng_state(),
                 'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(state, out/'last.pt')
        if improved:
            torch.save(state, out/'best.pt')
        if cfg['early_stop'] and stale >= cfg['early_stop']:
            break


if __name__ == '__main__':
    main()
