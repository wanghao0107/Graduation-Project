import torch
from torch.utils.data import DataLoader
from torchmetrics import Metric
from thop import profile
from skimage.morphology import skeletonize
import numpy as np
import warnings
from torchmetrics.classification import BinaryAveragePrecision
from torchmetrics.regression import MeanSquaredError
from torchmetrics.segmentation import DiceScore, MeanIoU
from monai.metrics import SurfaceDistanceMetric, HausdorffDistanceMetric

# 过滤 MONAI 相关警告
warnings.filterwarnings("ignore", message=".*always_return_as_numpy.*")
warnings.filterwarnings("ignore", message="the prediction of class 0 is all 0.*")

from dataset import read_index_csv, ImageSegDataset
from models.unet import UNet
from models.lsseg import LSSeg
from models.sam_lora import SAMLoRA
from models.medsam import MedSAM
from models.lsseg_sam_lora import LSSegSAMLoRA, LSSegSAMLoRA_Simple
from models.lsseg_medsam import LSSegMedSAM, LSSegMedSAM_Simple

class OIS(Metric):
    def __init__(self, thresholds=100, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = torch.linspace(0, 1, thresholds)

        self.add_state("total_f1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (N, 1, H, W) tensor with edge probability predictions
            target: (N, 1, H, W) tensor with binary edge annotations
        """
        # Move thresholds to same device as preds
        thresholds = self.thresholds.to(preds.device)

        target = target.bool()  # (N, 1, H, W)

        # Calculate binary masks for all thresholds
        binary_masks = preds > thresholds.view(1, -1, 1, 1)  # (N, T, H, W)

        # Calculate TP, FP, FN
        tp = (binary_masks & target).sum(dim=(2, 3))  # (N, T)
        fp = (binary_masks & ~target).sum(dim=(2, 3))
        fn = (~binary_masks & target).sum(dim=(2, 3))

        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)

        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        # Find best F1 for each image
        best_f1 = f1.max(dim=1).values  # (N,)

        # Update states
        self.total_f1 += best_f1.sum()
        self.total_samples += preds.shape[0]

    def compute(self):
        return self.total_f1 / self.total_samples


class ODS(Metric):
    def __init__(self, thresholds=100, **kwargs):
        super().__init__(**kwargs)
        self.thresholds = torch.linspace(0, 1, thresholds)

        self.add_state("tp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (N, 1, H, W) tensor with edge probability predictions
            target: (N, 1, H, W) tensor with binary edge annotations
        """
        thresholds = self.thresholds.to(preds.device)

        target = target.bool()  # (N, 1, H, W)

        # Calculate binary masks
        binary_masks = preds > thresholds.view(1, -1, 1, 1)  # (N, T, H, W)

        # Calculate TP, FP, FN
        tp = (binary_masks & target).sum(dim=(0, 2, 3))  # (T,)
        fp = (binary_masks & ~target).sum(dim=(0, 2, 3))
        fn = (~binary_masks & target).sum(dim=(0, 2, 3))

        # Update states
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-6)
        return f1_scores.max()


# def count(model, device, config):
#     input = torch.randn(1, 3, config['image_resize'][0], config['image_resize'][1]).to(
#         device)  # input shape (batch_size, channels, height, width)
#
#     # Calculate FLOPs and parameter count
#     flops, params = profile(model, inputs=(input,))
#
#     return {'FLOPs': flops, 'Params': params}

def count(model, device, config):
    """
    计算模型的 FLOPs 和参数量。
    对 SAM系列模型直接跳过，避免 thop 解析复杂 Transformer 结构时报错。
    """
    # 【新增】如果是 SAM系列模型，直接跳过计算
    if isinstance(model, (SAMLoRA, MedSAM, LSSegSAMLoRA, LSSegSAMLoRA_Simple, LSSegMedSAM, LSSegMedSAM_Simple)):
        return {'FLOPs': 0.0, 'Params': 0.0}

        # 其他标准模型，正常使用 thop 计算
    input_tensor = torch.randn(1, 3, config['image_resize'][0], config['image_resize'][1]).to(device)

    try:
        # 加上 verbose=False 可以防止 thop 在终端打印多余的层级信息
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        return {'FLOPs': flops, 'Params': params}
    except Exception as e:
        print(f"\n[Warning] thop.profile 无法计算当前模型的 FLOPs/Params。原因: {e}")
        return {'FLOPs': 0.0, 'Params': 0.0}


class clDiceMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state("total_cldice", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (N, 1, H, W) tensor with binary edge 0/1
            target: (N, 1, H, W) tensor with binary edge annotations
        """
        pred = preds.cpu().numpy().astype(bool)
        targ = target.cpu().numpy().astype(bool)
        for b in range(preds.shape[0]):
            self.total_cldice += self._clDice(pred[b, 0], targ[b, 0])
        self.total_samples += preds.shape[0]

    def compute(self):
        return self.total_cldice / self.total_samples

    # https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py
    def _cl_score(self, v, s):
        """[this function computes the skeleton volume overlap]

        Args:
            v ([bool]): [image]
            s ([bool]): [skeleton]

        Returns:
            [float]: [computed skeleton volume intersection]
        """
        if np.sum(s) == 0:
            return 0.0  # 防止除零
        return np.sum(v * s) / np.sum(s)

    def _clDice(self, v_p, v_l):
        """[this function computes the cldice metric]

        Args:
            v_p ([bool]): [predicted image]
            v_l ([bool]): [ground truth image]

        Returns:
            [float]: [cldice metric]
        """
        tprec = self._cl_score(v_p, skeletonize(v_l))
        tsens = self._cl_score(v_l, skeletonize(v_p))
        if tprec + tsens == 0:
            return 0.0  # 防止除零
        return 2 * tprec * tsens / (tprec + tsens)


def evaluate(model, data_iter, device, config):
    ap = BinaryAveragePrecision(thresholds=100).to(device)
    mse = MeanSquaredError().to(device)
    dsc = DiceScore(num_classes=2, average='macro').to(device)
    miou = MeanIoU(num_classes=2).to(device)
    cldice = clDiceMetric().to(device)
    assd = SurfaceDistanceMetric(symmetric=True, include_background=False, get_not_nans=True)
    hd95 = HausdorffDistanceMetric(percentile=95, include_background=False, get_not_nans=True)
    ods = ODS(thresholds=100).to(device)

    model.to(device)
    model.eval()

    is_sam_lora = isinstance(model, (SAMLoRA, MedSAM, LSSegSAMLoRA, LSSegSAMLoRA_Simple, LSSegMedSAM, LSSegMedSAM_Simple))

    with torch.no_grad():
        for images, masks in data_iter:
            images, masks = images.to(device), masks.to(device)
            if is_sam_lora:
                # SAMLoRA 内部自己做归一化，eval 时不传 masks（退化为全图 Box Prompt）
                preds = model(images).sigmoid()
                # GT Box 上限评估
                #preds = model(images, masks).sigmoid()

            else:
                images = images.float() / 255.0
                preds = model(images).sigmoid()

            # 累计指标
            ap.update(preds.flatten(), masks.long().flatten())
            mse.update(preds, masks.float())
            dsc.update((preds > 0.5).long(), masks.long())
            miou.update((preds > 0.5).long(), masks.long())
            cldice.update((preds > 0.5).long(), masks.long())
            ods.update(preds, masks.float())

            # MONAI 指标需要 one-hot 格式: (B, 2, H, W)
            pred_binary = (preds > 0.5).long()
            mask_long = masks.long()
            pred_onehot = torch.cat([1 - pred_binary, pred_binary], dim=1).float()
            mask_onehot = torch.cat([1 - mask_long, mask_long], dim=1).float()
            assd(pred_onehot, mask_onehot)
            hd95(pred_onehot, mask_onehot)

    # 计算ASSD和95HD, 过滤掉inf/nan
    dists, mask = assd.aggregate()  # dists 含 inf, mask 是 0/1
    assd_val = (dists * mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)

    dists, mask = hd95.aggregate()
    hd95_val = (dists * mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)

    metrics = {
        'AP': ap.compute().item(),
        'MSE': mse.compute().item(),
        'DSC': dsc.compute().item(),
        'mIoU': miou.compute().item(),
        'clDice': cldice.compute().item(),
        'ASSD': assd_val.item(),
        '95HD': hd95_val.item(),
        'ODS': ods.compute().item()
    }
    return metrics | count(model, device, config)


if __name__ == '__main__':
    config = {
        'exp_name': 'LSSeg_CREMI',
        'outer_cv_num': 5,
        'inner_cv_num': 3,
        'training_curve_path': 'curve.jpg',
        'random_state': 1001,
        'index_csv': 'data/idx_CREMI.csv',
        'image_resize': [256, 256],
        'num_epochs': 50,
        'n_trials': 30
    }

    model_path = r'log\LSSeg_Microtubule 01-29 03_57\fold_0\model_weights_0.pth'
    model = LSSeg(in_channels=[3, 8, 8])
    model.load_state_dict(torch.load(model_path))

    img_paths, msk_paths = read_index_csv('data/idx_Microtubule.csv')

    data_loader = DataLoader(
        ImageSegDataset(img_paths, msk_paths, resize=(256, 256), is_train=False),
        batch_size=2, shuffle=False
    )

    metrics = evaluate(model, data_loader, device=torch.device('cuda'), config=config)
    print(metrics)
    # pd.DataFrame([metrics]).to_csv('test.csv', index=False, encoding='utf-8')