"""
MedSAM 封装模型适配器
将 MedSAM 封装为与 lssegv2.1 其他模型完全兼容的接口。

接口设计原则（零侵入）：
  - dataset.py 不需要任何改动
  - eval.py    仅需一行改动
  - train.py   仅需添加导入和一个分支

核心适配逻辑（全部封装在本文件内）：
  1. 图像格式转换：uint8 [0-255] → SAM 标准归一化
  2. GT-derived Box Prompt：从 GT mask 动态生成边界框
  3. 三步推理：image_encoder → prompt_encoder → mask_decoder
  4. 上采样：low_res_masks → 原始分辨率

用法（在 train.py 的 build_model 中）：
    from models.medsam import MedSAM

    def build_model(**params):
        ...
        elif CURRENT_MODEL == 'MedSAM':
            model = MedSAM(checkpoint_path="medsam_vit_b.pth", target_size=512)
        ...
    
    # 训练循环中：
    preds = model(images, masks)   # MedSAM 接收 images + masks（用于生成 box）

参考：
    - MedSAM: https://github.com/bowang-lab/MedSAM
    - 论文: "Segment Anything in Medical Images", Nature Communications 2024
"""

import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model


def resize_sam_pos_embed(sam_model, target_size=512):
    """
    调整位置编码分辨率 (1024 -> target_size)
    """
    pos_embed = sam_model.image_encoder.pos_embed
    orig_size = pos_embed.shape[1]
    new_grid_size = target_size // 16

    if orig_size != new_grid_size:
        print(f"⚠️ Resizing SAM pos_embed from {orig_size}x{orig_size} to {new_grid_size}x{new_grid_size}...")
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        sam_model.image_encoder.pos_embed = nn.Parameter(pos_embed)
        sam_model.image_encoder.img_size = target_size

    # 同步调整 Prompt Encoder
    sam_model.prompt_encoder.image_embedding_size = (new_grid_size, new_grid_size)
    sam_model.prompt_encoder.input_image_size = (target_size, target_size)


def predict_masks_batch_friendly(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    修复了 SAM 官方 predict_masks 在批量训练时的 repeat_interleave bug。
    """
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    # 修复点：只有 batch 维度不匹配时才复制
    if image_embeddings.shape[0] != tokens.shape[0]:
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
    else:
        src = image_embeddings

    src = src + dense_prompt_embeddings
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape

    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

    src = src.transpose(1, 2).view(b, c, h, w)
    upscaled_embedding = self.output_upscaling(src)
    hyper_in_list: list[torch.Tensor] = []
    for i in range(self.num_mask_tokens):
        hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

    iou_pred = self.iou_prediction_head(iou_token_out)
    return masks, iou_pred


def load_medsam(model_type="vit_b", checkpoint_path="medsam_vit_b.pth", target_size=512, lora_r=0, lora_alpha=4):
    """
    加载 MedSAM 模型，可选添加 LoRA 支持
    
    Args:
        model_type: SAM 模型类型，默认 "vit_b"
        checkpoint_path: MedSAM 权重路径
        target_size: 目标图像尺寸
        lora_r: LoRA 秩（rank），0 表示不使用 LoRA（原始 MedSAM 行为）
        lora_alpha: LoRA 缩放因子，实际缩放 = alpha/r
    
    训练策略：
        - lora_r=0: 冻结 Image Encoder，训练 Prompt Encoder 和 Mask Decoder（原始 MedSAM）
        - lora_r>0: 使用 LoRA 微调 Image Encoder，训练 Prompt Encoder 和 Mask Decoder
    """
    print(f"Loading MedSAM {model_type} (Target: {target_size}, LoRA r={lora_r})...")
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)

    resize_sam_pos_embed(sam_model, target_size=target_size)

    # 修复 batch 训练 bug
    sam_model.mask_decoder.predict_masks = types.MethodType(
        predict_masks_batch_friendly, sam_model.mask_decoder
    )
    print("✅ Patched MaskDecoder to support batch training.")

    # 冻结所有参数
    for param in sam_model.parameters():
        param.requires_grad = False

    # LoRA 配置
    if lora_r > 0:
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, target_modules=["qkv"],
            lora_dropout=0.1, bias="none", modules_to_save=[],
        )
        sam_model.image_encoder = get_peft_model(sam_model.image_encoder, lora_config)
        print(f"✅ LoRA enabled: r={lora_r}, alpha={lora_alpha}")
    else:
        # 原始 MedSAM：Image Encoder 保持冻结
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        print("✅ Image Encoder frozen (no LoRA)")

    # Prompt Encoder 和 Mask Decoder 可训练
    for param in sam_model.prompt_encoder.parameters():
        param.requires_grad = True
    for param in sam_model.mask_decoder.parameters():
        param.requires_grad = True

    print("✅ MedSAM loaded: Prompt Encoder & Mask Decoder trainable.")
    return sam_model


class MedSAM(nn.Module):
    """
    MedSAM 封装模型，与 lssegv2.1 其他模型接口兼容。

    forward(images, masks=None) -> logits [B, 1, H, W]

    参数：
        images : [B, 3, H, W]  torch.uint8，值域 0-255
                 （与 lssegv2.1 的 ImageSegDataset 输出格式完全一致）
        masks  : [B, H, W] 或 [B, 1, H, W]  torch.uint8，值域 0/1
                 训练时传入 GT mask，用于生成 GT-derived Box Prompt
                 eval 时传入 masks 使用 GT Box，不传则使用全图 Box

    返回：
        logits : [B, 1, H, W]  未经 sigmoid 的原始输出
    """

    # SAM 标准归一化参数（MedSAM 沿用 SAM 的 PIXEL_MEAN/STD）
    # 参考: https://github.com/bowang-lab/MedSAM/blob/main/segment_anything/modeling/sam.py
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    def __init__(self, checkpoint_path="medsam_vit_b.pth", target_size=512, lora_r=0, lora_alpha=4):
        super().__init__()
        self.target_size = target_size
        self.sam = load_medsam(
            model_type="vit_b",
            checkpoint_path=checkpoint_path,
            target_size=target_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

    # ----------------------------------------------------------
    # 内部工具方法
    # ----------------------------------------------------------

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        将 lssegv2.1 的图像格式转换为 MedSAM 标准格式。
        
        【重要】MedSAM 沿用 SAM 的 PIXEL_MEAN/STD 标准化方式
        参考: https://github.com/bowang-lab/MedSAM/blob/main/segment_anything/modeling/sam.py
        
        preprocess(x) = (x - pixel_mean) / pixel_std
        """
        images_f = images.float()
        mean = self.PIXEL_MEAN.to(images.device)
        std = self.PIXEL_STD.to(images.device)
        images_sam = (images_f - mean) / std

        # 如果尺寸不匹配，插值对齐到 target_size
        H, W = images_sam.shape[-2], images_sam.shape[-1]
        if H != self.target_size or W != self.target_size:
            images_sam = F.interpolate(
                images_sam, (self.target_size, self.target_size),
                mode='bilinear', align_corners=False
            )
        return images_sam

    def _generate_boxes(self, masks: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        从 GT mask 生成 GT-derived Box Prompt。

        masks : [B, H, W] 或 [B, 1, H, W]，值域 0/1
        返回  : [B, 4]  float，格式 [x1, y1, x2, y2]，坐标系为 H×W
        """
        # 统一 shape 为 [B, H, W]
        if masks.ndim == 4:
            msk = masks[:, 0, :, :]   # [B, 1, H, W] → [B, H, W]
        else:
            msk = masks               # [B, H, W]

        msk_np = msk.detach().cpu().numpy()  # [B, H, W]
        B = msk_np.shape[0]
        boxes = []

        for i in range(B):
            y_indices, x_indices = np.where(msk_np[i] > 0)
            if len(y_indices) > 0:
                x_min = int(x_indices.min())
                x_max = int(x_indices.max())
                y_min = int(y_indices.min())
                y_max = int(y_indices.max())
                # 添加小的边界扩展（与MedSAM一致）
                perturb = 5
                x_min = max(0, x_min - perturb)
                x_max = min(W,  x_max + perturb)
                y_min = max(0, y_min - perturb)
                y_max = min(H,  y_max + perturb)
            else:
                # 负样本（全黑 mask）→ 全图框
                x_min, y_min, x_max, y_max = 0, 0, W, H

            boxes.append([x_min, y_min, x_max, y_max])

        return torch.tensor(boxes, dtype=torch.float32, device=masks.device)

    def _get_full_image_boxes(self, B: int, device: torch.device) -> torch.Tensor:
        """
        生成全图 Box Prompt（eval 时无 GT mask 可用时的退化策略）。
        返回：[B, 4]
        """
        box = torch.tensor(
            [[0, 0, self.target_size, self.target_size]],
            dtype=torch.float32, device=device
        ).expand(B, -1)
        return box

    # ----------------------------------------------------------
    # 前向传播
    # ----------------------------------------------------------

    def forward(self, images: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        """
        images : [B, 3, H, W]  uint8 0-255
        masks  : [B, H, W] 或 [B, 1, H, W]  0/1（训练时传入；eval 时可不传）
        返回   : logits [B, 1, H, W]
        """
        B, C, H, W = images.shape

        # 1. 图像格式转换
        images_sam = self._preprocess_images(images)

        # 2. 生成 Box Prompt
        if masks is not None:
            boxes = self._generate_boxes(masks, H, W)   # GT-derived Box（训练时）
        else:
            boxes = self._get_full_image_boxes(B, images.device)  # 全图 Box（eval 时）

        # 3. SAM 三步推理
        image_embeddings = self.sam.image_encoder(images_sam)

        sparse, dense = self.sam.prompt_encoder(
            points=None,
            boxes=boxes.unsqueeze(1),   # [B, 4] → [B, 1, 4]
            masks=None
        )

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )

        # 4. 上采样回原始分辨率
        logits = F.interpolate(low_res_masks, (H, W), mode='bilinear', align_corners=False)
        return logits  # [B, 1, H, W]