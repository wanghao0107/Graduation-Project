# lssegv2.1-master

医学/生物图像分割实验仓库，当前重点支持以下两类模型：

- 经典分割基线：`LSSeg`、`UNet`、`BDCN`、`TEED`、`ConDSeg`、`FSGNet`、`TransUNet`
- 级联模型：`SAMLoRA`、`MedSAM`、`LSSegSAMLoRA`、`LSSegMedSAM`

当前主训练入口为 [train_v3.py](train_v3.py)，代码中已包含：

- 5 折外层交叉验证
- 每折 Optuna 自动搜索
- `LSSeg + SAM/MedSAM + LoRA` 级联训练
- `DiceCE`、`tracing loss` 等损失函数
- `AP / DSC / mIoU / clDice / ASSD / 95HD / ODS` 评估指标

## 环境

推荐：

- Python 3.10+
- PyTorch 2.x
- CUDA 11.8+（可选，GPU 训练更实际）

安装依赖：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果 `segment-anything` 直接安装失败，可改为：

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## 预训练权重

级联模型依赖外部权重文件，请放在项目根目录或代码中指定的路径：

- `sam_vit_b_01ec64.pth`
- `medsam_vit_b.pth`
- 对应数据集/对应 fold 的 `LSSeg` checkpoint

对于 `LSSegSAMLoRA` / `LSSegMedSAM`，建议 `LSSeg` 初始化权重与当前外层 fold 对齐，避免交叉验证中的数据泄漏。

## 数据组织

数据索引通过 `data/idx_*.csv` 管理，每个 CSV 两列：

1. 图像路径
2. mask 路径

例如：

```csv
data/HRF/images/01.png,data/HRF/masks/01.png
data/HRF/images/02.png,data/HRF/masks/02.png
```

目前仓库内已包含多套数据目录与索引，如：

- `data/HRF`
- `data/RITE`
- `data/STARE`
- `data/CHASE_DB1`

## 运行训练

当前默认使用：

```bash
python train_v3.py
```

训练日志会输出到：

- `log/<exp_name> <month-day hour_minute>/fold_k/metrics.csv`
- `log/<exp_name> <month-day hour_minute>/fold_k/training_curve.jpg`
- `log/<exp_name> <month-day hour_minute>/fold_k/model_weights_k.pth`
- `log/<exp_name> <month-day hour_minute>/final_metrics.csv`

同时 Optuna 搜索结果会写入项目根目录的 `optuna.db`。

## 关键配置

当前实验配置主要在 [train_v3.py](train_v3.py) 中修改。

### 1. 模型选择

修改 `CURRENT_MODEL`：

```python
CURRENT_MODEL = 'LSSegSAMLoRA'
```

可选值包括：

- `LSSeg`
- `UNet`
- `BDCN`
- `TEED`
- `CATS`
- `ConDSeg`
- `FSGNet`
- `TransUNet`
- `SAMLoRA`
- `MedSAM`
- `LSSegSAMLoRA`
- `LSSegSAMLoRA_Simple`
- `LSSegMedSAM`
- `LSSegMedSAM_Simple`

### 2. 级联模型的 LSSeg 初始化权重

```python
LSSEG_CHECKPOINT_PATH = "log/test_HRF_LSSeg 02-14 21_39/fold_2/model_weights_2.pth"
```

建议按当前外层 fold 动态切换，而不是长期固定某一个 fold。

### 3. 数据集与训练参数

```python
config = {
    'exp_name': 'test_HRF_LSSegSAMLoRA',
    'outer_cv_num': 5,
    'random_state': 800,
    'index_csv': 'data/idx_HRF.csv',
    'image_resize': [512, 512],
    'num_epochs': 70,
    'n_trials': 40,
    'loss_function': dice_ce_loss(),
}
```

## 当前训练流程说明

`train_v3.py` 当前流程为：

1. 外层 5 折划分训练集/测试集
2. 每个外层 fold 内使用 Optuna 搜索超参数
3. Optuna 目标函数内部使用一次 `train_test_split` 划分训练/验证集
4. 采用最佳参数在该外层 fold 上重新训练，并在测试集上评估

因此，当前实现是“外层交叉验证 + 每折 Optuna 搜索”，而不是严格意义上的完整内层 K 折 nested CV。

## 评估指标

评估逻辑位于 [eval.py](eval.py)，当前输出指标包括：

- `AP`
- `MSE`
- `DSC`
- `mIoU`
- `clDice`
- `ASSD`
- `95HD`
- `ODS`
- `FLOPs`
- `Params`

说明：

- 对 `SAM` / `MedSAM` 及其级联模型，当前 `FLOPs` 和 `Params` 默认记为 `0`，因为 `thop` 对这类结构的统计未完整实现。

## 相关脚本

- [train_v3.py](train_v3.py)：当前主训练入口
- [train.py](train.py)：较早版本训练脚本
- [train_v2.py](train_v2.py)：中间实验版本
- [eval.py](eval.py)：评估与指标统计
- [loss.py](loss.py)：损失函数
- [dataset.py](dataset.py)：数据读取与增强
- `analyze_optuna*.py` / `analyze_db.py`：Optuna 结果分析脚本

## 备注

- 当前仓库包含大量历史实验日志，`log/` 目录体积可能较大。
- 代码仍以研究实验为主，默认配置会随着实验推进调整，建议在运行前先检查 `train_v3.py` 中的模型、checkpoint、数据集与损失配置。
