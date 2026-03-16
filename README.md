## 安装环境
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行
```bash
python train.py
```
使用嵌套交叉验证（Nested Cross-Validation）
- 外层：5折
- 内层：3折 + Optuna自动调参

每一折结果保存在`log/`文件夹下，同时写入`optuna.db`。

## 实验配置
```python
config = {
    'exp_name': 'LSSeg_STARE',
    'outer_cv_num': 5,
    'inner_cv_num': 3,
    'training_curve_path': 'curve.jpg',
    'random_state': 1001,
    'index_csv': 'data/idx_STARE.csv',
    'image_resize': [256, 256],
    'num_epochs': 50,
    'n_trials': 30
}
```

## 模型切换
```python
def build_model(**params):
    # model = UNet(
    #     n_channels=3,
    #     n_classes=1,
    # )

    model = LSSeg(in_channels=[3, 8, 8])

    return model
```