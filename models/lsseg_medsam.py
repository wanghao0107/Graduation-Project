"""
LSSeg + MedSAM-LoRA 级联模型
LSSeg 生成粗分割 Mask，作为 Mask Prompt 输入给 MedSAM-LoRA 进行精细分割

与 LSSegSAMLoRA 的区别：
  - LSSegSAMLoRA: 使用 SAM 原始权重
  - LSSegMedSAM: 使用 MedSAM 权重（医学图像微调版）

用法（在 train.py 的 build_model 中）：
    from models.lsseg_medsam import LSSegMedSAM

    def build_model(**params):
        model = LSSegMedSAM(
            lsseg_checkpoint="log/xxx/model_weights_x.pth",
            sam_checkpoint="medsam_vit_b.pth",
            target_size=512,
            lora_r=4,
            lora_alpha=4,
            freeze_lsseg=True
        )
        return model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.lsseg import LSSeg
from models.medsam import load_medsam
import numpy as np


class LSSegMedSAM(nn.Module):
    """
    LSSeg + MedSAM-LoRA 级联模型

    流程：
        1. LSSeg 生成粗分割 mask
        2. 将 LSSeg 的 mask 作为 MedSAM 的 Mask Prompt
        3. MedSAM-LoRA 基于 Mask Prompt 生成精细分割结果

    参数：
        lsseg_checkpoint: LSSeg 预训练权重路径
        sam_checkpoint: MedSAM 权重路径（默认 medsam_vit_b.pth）
        target_size: 输入图像尺寸
        lora_r: LoRA 秩，0 表示不使用 LoRA
        lora_alpha: LoRA 缩放因子
        freeze_lsseg: 是否冻结 LSSeg
        use_box_prompt: 是否同时使用 box prompt
        prompt_bias: 控制 Mask Prompt 召回率的偏置
        box_bias: 控制 Box 生成阈值的偏置
        box_expand_ratio: Box 扩展比例（占图像尺寸的比例）
    """

    # SAM 标准归一化参数
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    def __init__(
        self,
        lsseg_checkpoint: Optional[str] = None,
        sam_checkpoint: str = "medsam_vit_b.pth",
        target_size: int = 512,
        lora_r: int = 4,
        lora_alpha: int = 4,
        freeze_lsseg: bool = True,
        lsseg_channels: list = [3, 8, 8],
        use_box_prompt: bool = True,
        prompt_bias: float = 0.0,
        box_bias: float = 0.0,
        box_expand_ratio: float = 0.02,
    ):
        super().__init__()
        self.target_size = target_size
        self.use_box_prompt = use_box_prompt
        self.prompt_bias = prompt_bias
        self.box_bias = box_bias
        self.box_expand_ratio = box_expand_ratio

        # ========== 1. 初始化 LSSeg ==========
        self.lsseg = LSSeg(in_channels=lsseg_channels)

        if lsseg_checkpoint is not None:
            print(f"Loading LSSeg checkpoint from {lsseg_checkpoint}...")
            state_dict = torch.load(lsseg_checkpoint, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.lsseg.load_state_dict(state_dict, strict=False)
            print("✅ LSSeg loaded successfully.")

        if freeze_lsseg:
            print("Freezing LSSeg parameters...")
            for param in self.lsseg.parameters():
                param.requires_grad = False
            self.lsseg.eval()

        # ========== 2. 初始化 MedSAM-LoRA ==========
        self.sam = load_medsam(
            model_type="vit_b",
            checkpoint_path=sam_checkpoint,
            target_size=target_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        self.mask_prompt_size = target_size // 4

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """将图像转换为 SAM 标准格式"""
        images_f = images.float()
        mean = self.PIXEL_MEAN.to(images.device)
        std = self.PIXEL_STD.to(images.device)
        images_sam = (images_f - mean) / std

        H, W = images_sam.shape[-2], images_sam.shape[-1]
        if H != self.target_size or W != self.target_size:
            images_sam = F.interpolate(
                images_sam, (self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            )
        return images_sam

    def _generate_boxes_from_mask(self, mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        从 logits mask 生成 bounding box
        
        Args:
            mask: [B, 1, h, w] logits（未经 sigmoid）
            H, W: 原始图像尺寸
            
        Returns:
            boxes: [B, 4] (x_min, y_min, x_max, y_max)
        """
        # 应用 Box 偏置后再计算概率
        adjusted_logits = mask + self.box_bias
        mask_prob = torch.sigmoid(adjusted_logits)
        mask_np = mask_prob.detach().cpu().numpy()
        B = mask_np.shape[0]
        boxes = []
        
        # 计算扩展像素数
        expand_pixels = int(min(H, W) * self.box_expand_ratio)

        for i in range(B):
            # 使用动态阈值（box_bias 已在 logits 上调整）
            y_indices, x_indices = np.where(mask_np[i, 0] > 0.5)
            if len(y_indices) > 0:
                x_min = int(x_indices.min())
                x_max = int(x_indices.max())
                y_min = int(y_indices.min())
                y_max = int(y_indices.max())
                # 使用可配置的扩展比例
                x_min = max(0, x_min - expand_pixels)
                x_max = min(W, x_max + expand_pixels)
                y_min = max(0, y_min - expand_pixels)
                y_max = min(H, y_max + expand_pixels)
            else:
                # 如果没有检测到前景，使用全图
                x_min, y_min, x_max, y_max = 0, 0, W, H

            boxes.append([x_min, y_min, x_max, y_max])

        return torch.tensor(boxes, dtype=torch.float32, device=mask.device)

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            images: [B, 3, H, W] uint8 0-255
            masks:  [B, H, W] 或 [B, 1, H, W] 0/1（可选）

        Returns:
            logits: [B, 1, H, W]
        """
        B, C, H, W = images.shape

        # ========== Step 1: LSSeg 生成粗分割 Mask ==========
        with torch.no_grad() if not any(p.requires_grad for p in self.lsseg.parameters()) else torch.enable_grad():
            lsseg_logits = self.lsseg(images.float() / 255.0)

        # ========== Step 2: 准备 SAM 输入 ==========
        images_sam = self._preprocess_images(images)

        # ========== Step 3: 准备 Mask Prompt（召回率优化）==========
        boosted_logits = lsseg_logits + self.prompt_bias

        mask_prompt = F.interpolate(
            boosted_logits,
            (self.mask_prompt_size, self.mask_prompt_size),
            mode='bilinear',
            align_corners=False
        )

        # ========== Step 4: 准备 Box Prompt（训练和推理统一使用 LSSeg logits）==========
        boxes = None
        if self.use_box_prompt:
            # 统一使用 LSSeg logits 生成 Box，保持训练-推理一致性
            boxes = self._generate_boxes_from_mask(lsseg_logits, H, W)

        # ========== Step 5: MedSAM 三步推理 ==========
        image_embeddings = self.sam.image_encoder(images_sam)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes.unsqueeze(1) if boxes is not None else None,
            masks=mask_prompt
        )

        low_res_masks, iou_pred = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # ========== Step 6: 上采样回原始分辨率 ==========
        logits = F.interpolate(low_res_masks, (H, W), mode='bilinear', align_corners=False)

        return logits


# ============================================================
# 简化版：只使用 LSSeg 的 mask 作为 prompt
# ============================================================

class LSSegMedSAM_Simple(nn.Module):
    """
    简化版：只用 LSSeg 的 mask 作为 prompt，不依赖 GT box
    """

    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    def __init__(
        self,
        lsseg_checkpoint: Optional[str] = None,
        sam_checkpoint: str = "medsam_vit_b.pth",
        target_size: int = 512,
        lora_r: int = 4,
        lora_alpha: int = 4,
        freeze_lsseg: bool = False,#【新增】不冻结，支持端到端
        lsseg_channels: list = [3, 8, 8],
        prompt_bias: float = 0.0,
    ):
        super().__init__()
        self.target_size = target_size
        self.prompt_bias = prompt_bias
        self.freeze_lsseg = freeze_lsseg

        # LSSeg
        self.lsseg = LSSeg(in_channels=lsseg_channels)
        if lsseg_checkpoint is not None:
            state_dict = torch.load(lsseg_checkpoint, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.lsseg.load_state_dict(state_dict, strict=False)

        if freeze_lsseg:
            for param in self.lsseg.parameters():
                param.requires_grad = False
            self.lsseg.eval()

        # MedSAM-LoRA
        self.sam = load_medsam(
            model_type="vit_b",
            checkpoint_path=sam_checkpoint,
            target_size=target_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        self.mask_prompt_size = target_size // 4

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        images_f = images.float()
        mean = self.PIXEL_MEAN.to(images.device)
        std = self.PIXEL_STD.to(images.device)
        images_sam = (images_f - mean) / std

        H, W = images_sam.shape[-2], images_sam.shape[-1]
        if H != self.target_size or W != self.target_size:
            images_sam = F.interpolate(
                images_sam, (self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            )
        return images_sam

    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, C, H, W = images.shape

        # LSSeg 生成粗 mask（条件梯度控制）
        use_grad = not self.freeze_lsseg
        with torch.no_grad() if not use_grad else torch.enable_grad():
            lsseg_logits = self.lsseg(images.float() / 255.0)

        # SAM 处理
        images_sam = self._preprocess_images(images)

        # 召回率优化
        boosted_logits = lsseg_logits + self.prompt_bias

        mask_prompt = F.interpolate(
            boosted_logits,
            (self.mask_prompt_size, self.mask_prompt_size),
            mode='bilinear',
            align_corners=False
        )

        # MedSAM 推理
        image_embeddings = self.sam.image_encoder(images_sam)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=mask_prompt
        )
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        logits = F.interpolate(low_res_masks, (H, W), mode='bilinear', align_corners=False)
        return logits