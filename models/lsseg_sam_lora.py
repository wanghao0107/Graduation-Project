"""
LSSeg + SAM-LoRA 级联模型
LSSeg 生成粗分割 Mask，作为 Mask Prompt 输入给 SAM-LoRA 进行精细分割

用法（在 train.py 的 build_model 中）：
    from models.lsseg_sam_lora import LSSegSAMLoRA
    
    def build_model(**params):
        model = LSSegSAMLoRA(
            lsseg_checkpoint="log/xxx/model_weights_x.pth",  # 训练好的 LSSeg 权重
            sam_checkpoint="sam_vit_b_01ec64.pth",
            target_size=512,
            lora_r=4,
            lora_alpha=4,
            freeze_lsseg=True  # 是否冻结 LSSeg
        )
        return model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.lsseg import LSSeg
from models.sam_lora import SAMLoRA, get_lora_sam
import types
import numpy as np


class LSSegSAMLoRA(nn.Module):
    """
    LSSeg + SAM-LoRA 级联模型
    
    流程：
        1. LSSeg 生成粗分割 mask（低分辨率，快速）
        2. 将 LSSeg 的 mask 下采样为 SAM 的 Mask Prompt
        3. SAM-LoRA 基于 Mask Prompt 生成精细分割结果
    
    forward(images, masks=None) -> logits [B, 1, H, W]
    
    参数：
        images : [B, 3, H, W] torch.uint8，值域 0-255
        masks  : [B, H, W] 或 [B, 1, H, W] torch.uint8，值域 0/1
                 训练时传入 GT mask，用于生成 Box Prompt 作为补充
                 （可选，如果不传则只用 LSSeg 的 mask prompt）
    
    返回：
        logits : [B, 1, H, W] 未经 sigmoid 的原始输出
    """
    
    # SAM 标准归一化参数
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    
    def __init__(
        self,
        lsseg_checkpoint: Optional[str] = None,
        sam_checkpoint: str = "sam_vit_b_01ec64.pth",
        target_size: int = 512,
        lora_r: int = 4,
        lora_alpha: int = 4,
        freeze_lsseg: bool = True,
        lsseg_channels: list = [3, 8, 8],
        use_box_prompt: bool = True,  # 是否同时使用 box prompt
        prompt_bias: float = 0.0,  # 【新增】控制 LSSeg 召回率的偏置，值越大越宽松
    ):
        super().__init__()
        self.target_size = target_size
        self.use_box_prompt = use_box_prompt
        self.prompt_bias = prompt_bias  # 保存偏置参数
        
        # ========== 1. 初始化 LSSeg ==========
        self.lsseg = LSSeg(in_channels=lsseg_channels)
        
        if lsseg_checkpoint is not None:
            print(f"Loading LSSeg checkpoint from {lsseg_checkpoint}...")
            state_dict = torch.load(lsseg_checkpoint, map_location='cpu')
            # 处理可能的 'module.' 前缀
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.lsseg.load_state_dict(state_dict, strict=False)
            print("✅ LSSeg loaded successfully.")
        
        if freeze_lsseg:
            print("Freezing LSSeg parameters...")
            for param in self.lsseg.parameters():
                param.requires_grad = False
            self.lsseg.eval()
        
        # ========== 2. 初始化 SAM-LoRA ==========
        self.sam = get_lora_sam(
            model_type="vit_b",
            checkpoint_path=sam_checkpoint,
            target_size=target_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )
        
        # SAM Mask Prompt 的目标尺寸
        # SAM 的 prompt_encoder 内部有 mask_downscaling (4x 下采样)
        # 所以输入尺寸 = image_embedding_size * 4 = (target_size/16) * 4 = target_size/4
        # target_size=512 时，mask_prompt 应为 128x128
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
    
    def _preprocess_for_lsseg(self, images: torch.Tensor) -> torch.Tensor:
        """将 uint8 [0-255] 图像转换为 LSSeg 输入格式 [0-1]"""
        return images.float() / 255.0
    
    def _generate_boxes_from_mask(self, mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """从 logits mask 生成 bounding box（使用 sigmoid 阈值）"""
        # 将 logits 转为概率用于生成 box
        mask_prob = torch.sigmoid(mask)
        mask_np = mask_prob.detach().cpu().numpy()
        B = mask_np.shape[0]
        boxes = []
        
        for i in range(B):
            y_indices, x_indices = np.where(mask_np[i, 0] > 0.5)
            if len(y_indices) > 0:
                x_min = int(x_indices.min())
                x_max = int(x_indices.max())
                y_min = int(y_indices.min())
                y_max = int(y_indices.max())
                # 添加小扰动
                perturb = 5
                x_min = max(0, x_min - np.random.randint(0, perturb))
                x_max = min(W, x_max + np.random.randint(0, perturb))
                y_min = max(0, y_min - np.random.randint(0, perturb))
                y_max = min(H, y_max + np.random.randint(0, perturb))
            else:
                x_min, y_min, x_max, y_max = 0, 0, W, H
            
            boxes.append([x_min, y_min, x_max, y_max])
        
        return torch.tensor(boxes, dtype=torch.float32, device=mask.device)
    
    def forward(self, images: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: [B, 3, H, W] uint8 0-255
            masks:  [B, H, W] 或 [B, 1, H, W] 0/1（可选，用于训练时辅助）
        
        Returns:
            logits: [B, 1, H, W]
        """
        B, C, H, W = images.shape
        
        # ========== Step 1: LSSeg 生成粗分割 Mask ==========
        with torch.no_grad() if not any(p.requires_grad for p in self.lsseg.parameters()) else torch.enable_grad():
            images_lsseg = self._preprocess_for_lsseg(images)
            # LSSeg 输出 [B, 1, H, W]，是 Logits（关键：不要 sigmoid！）
            lsseg_logits = self.lsseg(images_lsseg)
        
        # ========== Step 2: 准备 SAM 输入 ==========
        images_sam = self._preprocess_images(images)
        
        # ========== Step 3: 准备 Mask Prompt（核心优化：提升召回率）==========
        # 将 LSSeg 的 logits 下采样到 SAM mask prompt 尺寸
        # 对于 target_size=512，mask_prompt 尺寸为 128x128
        # SAM 的 mask_downscaling 会将其 4x 下采样为 32x32，与 image_embeddings 尺寸匹配
        # 关键：SAM 要求传入 logits，不是概率！
        # 
        # 【prompt_bias 优化】：
        # 在 logits 上加偏置，等效于降低 sigmoid 后的阈值，提升召回率
        # 例如：原本 logits=-1.38 (概率≈0.2，被当成背景)
        #      加上 prompt_bias=1.5 后变成 0.12 (概率≈0.53，被传给 SAM)
        # 这样可以让更多"疑似血管"区域被传给 SAM 进行判断
        boosted_logits = lsseg_logits + self.prompt_bias
        
        mask_prompt = F.interpolate(
            boosted_logits, 
            (self.mask_prompt_size, self.mask_prompt_size),
            mode='bilinear', 
            align_corners=False
        )
        
        # ========== Step 4: 准备 Box Prompt（可选）==========
        boxes = None
        if self.use_box_prompt:
            if masks is not None:
                # 使用 GT mask 生成 box（训练时）
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                boxes = self._generate_boxes_from_mask(masks.float(), H, W)
            else:
                # 使用 LSSeg 的 mask 生成 box（推理时）
                # 传入 lsseg_logits，函数内部会 sigmoid 转为概率
                boxes = self._generate_boxes_from_mask(lsseg_logits, H, W)
        
        # ========== Step 5: SAM 三步推理 ==========
        # 1. Image Encoder
        image_embeddings = self.sam.image_encoder(images_sam)
        
        # 2. Prompt Encoder（传入 mask prompt 和 box prompt）
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=boxes.unsqueeze(1) if boxes is not None else None,
            masks=mask_prompt  # ← 关键：传入 LSSeg 生成的 mask prompt
        )
        
        # 3. Mask Decoder
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
# 简化版：只使用 LSSeg 的 mask 作为 prompt，不额外使用 box
# ============================================================

class LSSegSAMLoRA_Simple(nn.Module):
    """
    简化版：只用 LSSeg 的 mask 作为 prompt，不依赖 GT box
    更适合实际部署场景
    
    【新增 prompt_bias】：
    控制 LSSeg 召回率的偏置，值越大越宽松，召回率越高
    """
    
    PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
    
    def __init__(
        self,
        lsseg_checkpoint: Optional[str] = None,
        sam_checkpoint: str = "sam_vit_b_01ec64.pth",
        target_size: int = 512,
        lora_r: int = 4,
        lora_alpha: int = 4,
        freeze_lsseg: bool = True,
        lsseg_channels: list = [3, 8, 8],
        prompt_bias: float = 0.0,  # 【新增】控制 LSSeg 召回率的偏置
    ):
        super().__init__()
        self.target_size = target_size
        self.prompt_bias = prompt_bias  # 保存偏置参数
        
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
        
        # SAM-LoRA
        self.sam = get_lora_sam(
            model_type="vit_b",
            checkpoint_path=sam_checkpoint,
            target_size=target_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )
        
        # SAM prompt_encoder 有 4x 下采样，所以输入尺寸 = target_size/4
        # target_size=512 时，mask_prompt 应为 128x128
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
        
        # LSSeg 生成粗 mask（logits，不要 sigmoid）
        with torch.no_grad():
            lsseg_logits = self.lsseg(images.float() / 255.0)
        
        # SAM 处理
        images_sam = self._preprocess_images(images)
        
        # 【prompt_bias 优化】：在 logits 上加偏置，提升召回率
        boosted_logits = lsseg_logits + self.prompt_bias
        
        # Mask prompt（关键：传入 logits，不是概率）
        mask_prompt = F.interpolate(
            boosted_logits, 
            (self.mask_prompt_size, self.mask_prompt_size),
            mode='bilinear', 
            align_corners=False
        )
        
        # SAM 推理
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
