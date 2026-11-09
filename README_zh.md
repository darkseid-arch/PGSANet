# PGSANet 中文说明

这是根据论文公式和算法编写的**可运行实现初稿**，包含网络、损失、训练、评估、推理和测试。它不是从作者原始实验工程恢复的代码，也尚未在真实 SSS/SAS 数据上复现论文结果。

## 快速使用

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python tools/make_demo.py
python train.py --config data/demo/config.json --train data/demo/train.json --val data/demo/val.json --output runs/demo --device cpu
python evaluate.py --checkpoint runs/demo/best.pt --manifest data/demo/test.json --output runs/demo/test_metrics.json --device cpu
python predict.py --checkpoint runs/demo/best.pt --image data/demo/test_0.png --output predictions/demo --device cpu
```

上述数据是随机合成数据，仅用于验证程序。真实训练请使用 `configs/sss.json` 或 `configs/sas.json`。

每个划分建立 JSON 列表，所有路径相对于该 JSON 文件：

```json
[
  {"image": "images/001.png", "mask": "masks/001.png", "prior": "priors/001.npy"}
]
```

- 图像默认使用灰度、除以 255；如果原始实验为 RGB，请将 `in_channels` 改为 3。
- 标注默认严格按论文表 I 的 RGB 颜色解码；索引标注需设 `label_format: "index"`，类别编号见英文 README 对应的数据模块。255 为忽略像素。
- SAS 的 Unlabeled 按论文结果表作为第 6 类参与评价，不自动忽略。
- `prior` 为与原图对齐的 `[3,H,W]` 浮点 `.npy` 数组。真实先验需要依据您的传感器、成像几何和传播模型准备，不能使用真实分割标签构造。
- 默认 `external` 模式必须提供先验；演示 `demo` 模式只有亮度代理量，不能作为真实物理模型或论文复现结果。

真实训练示例：

```bash
python train.py --config configs/sss.json --train data/sss/train.json --val data/sss/val.json --output runs/sss
python evaluate.py --checkpoint runs/sss/best.pt --manifest data/sss/test.json --output runs/sss/test_metrics.json
python predict.py --checkpoint runs/sss/best.pt --image data/sss/images/example.png --prior data/sss/priors/example.npy --output predictions
```

默认采用论文明确给出的 AdamW、学习率 0.0006、权重衰减 0.0001、batch size 8、256×256 输入、200 epochs、decoder dropout 0.5、梯度裁剪 5。其余未明确参数已记录在 [实现对照与待确认项](docs/IMPLEMENTATION.md)，包括损失权重、网络宽度、频带、专家结构、调度周期和早停规则。

权重按训练标签的实际频率计算，不使用测试集统计。输出包含 `best.pt`、`last.pt`、配置、日志和指标。mIoU/mPA 输出为 0～1，乘以 100 才是百分比。预测保存原图尺寸的类别图和 RGB 彩色图；评估按配置中的输入尺寸进行。

当前版本已提供 `.gitignore` 和 GitHub Actions 测试配置。上传前应确认实现对照表中的差异，选择许可证，并补充真实划分、先验定义和验证结果。源代码包不包含您的论文 PDF、真实数据或权重，也未上传到 GitHub。
