## 快速使用

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python tools/make_demo.py
python train.py --config data/demo/config.json --train data/demo/train.json --val data/demo/val.json --output runs/demo --device cpu
python evaluate.py --checkpoint runs/demo/best.pt --manifest data/demo/test.json --output runs/demo/test_metrics.json --device cpu
python predict.py --checkpoint runs/demo/best.pt --image data/demo/test_0.png --output predictions/demo --device cpu
```
每个划分建立 JSON 列表，所有路径相对于该 JSON 文件：

```json
[
  {"image": "images/001.png", "mask": "masks/001.png", "prior": "priors/001.npy"}
]
```

```bash
python train.py --config configs/sss.json --train data/sss/train.json --val data/sss/val.json --output runs/sss
python evaluate.py --checkpoint runs/sss/best.pt --manifest data/sss/test.json --output runs/sss/test_metrics.json
python predict.py --checkpoint runs/sss/best.pt --image data/sss/images/example.png --prior data/sss/priors/example.npy --output predictions
```
