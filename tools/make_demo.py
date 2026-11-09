"""Generate synthetic data only for software smoke tests; never research evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/demo')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(42)
    for split, count in [('train', 4), ('val', 2), ('test', 2)]:
        rows = []
        for i in range(count):
            stem = '{}_{}'.format(split, i)
            mask = rng.randint(0, 5, (32, 32)).astype('uint8')
            image = np.clip(mask*40+rng.randint(0, 30, mask.shape), 0, 255).astype('uint8')
            Image.fromarray(image).save(out/(stem+'.png'))
            Image.fromarray(mask).save(out/(stem+'_mask.png'))
            rows.append({'image': stem+'.png', 'mask': stem+'_mask.png'})
        (out/(split+'.json')).write_text(json.dumps(rows, indent=2), encoding='utf-8')
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'configs/sss.json').read_text(encoding='utf-8'))
    cfg.update(label_format='index', prior_mode='demo', image_size=32, epochs=2, batch_size=2)
    cfg['model'].update(widths=[4, 8, 16], windows=[3], attention_grid=4)
    (out/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    print('Synthetic smoke-test data written to ' + str(out))


if __name__ == '__main__':
    main()
