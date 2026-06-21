#!/usr/bin/env python3
"""
Generate quantitative comparison charts and qualitative segmentation panels.

Usage examples
--------------
1. Generate the default metric comparison figure:
   python scripts/generate_comparison_figures.py metrics

2. Generate one vessel comparison panel:
   python scripts/generate_comparison_figures.py vessel --dataset RITE --fold 0 --sample-index 0

3. Generate one vessel panel for every vessel dataset in the selected experiment set:
   python scripts/generate_comparison_figures.py batch-vessels --fold 0 --sample-index 0

Notes
-----
- Run this script inside the project environment that already has torch/cv2/matplotlib.
- The script uses the selected 10 logs that were chosen for the thesis comparison.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import KFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lsseg import LSSeg
from models.lsseg_sam_lora import LSSegSAMLoRA


SELECTED_EXPERIMENTS: Dict[str, Dict[str, str]] = {
    "STARE": {
        "baseline": "log/test_STARE_LSSeg 02-13 11_34",
        "cascade": "log/test_STARE_LSSegSAMLoRA_DiceCE 04-10 23_07",
    },
    "RITE": {
        "baseline": "log/test_RITE_LSSeg 02-14 08_30",
        "cascade": "log/test_RITE_LSSegSAMLoRA 04-03 14_33",
    },
    "CHASE_DB1": {
        "baseline": "log/test_CHASE_DB1_LSSeg 02-17 12_48",
        "cascade": "log/test_CHASE_DB1_LSSegSAMLoRA_DiceCE 04-12 00_09",
    },
    "HRF": {
        "baseline": "log/test_HRF_LSSeg 02-14 21_39",
        "cascade": "log/test_HRF_LSSegSAMLoRA_DiceCE 04-05 21_24",
    },
    "AxonDeepSeg_SEM": {
        "baseline": "log/test_AxonDeepSeg_SEM_LSSeg 03-02 15_55",
        "cascade": "log/test_AxonDeepSeg_SEM_LSSegSAMLoRA_DiceCE 04-18 09_22",
    },
}

VESSEL_DATASETS = ("STARE", "RITE", "CHASE_DB1", "HRF")
DEFAULT_METRICS = ("DSC", "mIoU", "clDice", "95HD")


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_final_metrics(csv_path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2 or not row[0]:
                continue
            try:
                metrics[row[0]] = float(row[1])
            except ValueError:
                continue
    return metrics


def read_fold_metrics(csv_path: Path) -> Dict[str, float]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return {k: float(v) for k, v in row.items() if k and v is not None and v != ""}


def read_index_csv(csv_path: Path) -> Tuple[List[str], List[str]]:
    images: List[str] = []
    masks: List[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            images.append(row[0])
            masks.append(row[1])
    return images, masks


def get_test_sample(dataset: str, fold: int, sample_index: int, random_state: int = 800, n_splits: int = 5) -> Tuple[Path, Path]:
    idx_csv = PROJECT_ROOT / "data" / f"idx_{dataset}.csv"
    image_paths, mask_paths = read_index_csv(idx_csv)
    indices = np.arange(len(image_paths))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(splitter.split(indices))
    _, test_idx = splits[fold]
    local_index = test_idx[sample_index % len(test_idx)]
    return resolve_path(image_paths[local_index]), resolve_path(mask_paths[local_index])


def select_extreme_folds(dataset: str, metric: str = "DSC") -> Tuple[int, float, int, float]:
    pair = SELECTED_EXPERIMENTS[dataset]
    baseline_dir = resolve_path(pair["baseline"])
    cascade_dir = resolve_path(pair["cascade"])

    baseline_scores: List[Tuple[int, float]] = []
    cascade_scores: List[Tuple[int, float]] = []
    for fold in range(5):
        baseline_metric_path = baseline_dir / f"fold_{fold}" / "metrics.csv"
        cascade_metric_path = cascade_dir / f"fold_{fold}" / "metrics.csv"
        baseline_scores.append((fold, read_fold_metrics(baseline_metric_path)[metric]))
        cascade_scores.append((fold, read_fold_metrics(cascade_metric_path)[metric]))

    baseline_worst_fold, baseline_worst_value = min(baseline_scores, key=lambda x: x[1])
    cascade_best_fold, cascade_best_value = max(cascade_scores, key=lambda x: x[1])
    return baseline_worst_fold, baseline_worst_value, cascade_best_fold, cascade_best_value


def select_max_positive_delta_fold(dataset: str, metric: str = "DSC") -> Tuple[int, float, float, float]:
    pair = SELECTED_EXPERIMENTS[dataset]
    baseline_dir = resolve_path(pair["baseline"])
    cascade_dir = resolve_path(pair["cascade"])

    candidates: List[Tuple[int, float, float, float]] = []
    for fold in range(5):
        baseline_metric_path = baseline_dir / f"fold_{fold}" / "metrics.csv"
        cascade_metric_path = cascade_dir / f"fold_{fold}" / "metrics.csv"
        baseline_value = read_fold_metrics(baseline_metric_path)[metric]
        cascade_value = read_fold_metrics(cascade_metric_path)[metric]
        delta = cascade_value - baseline_value
        if delta > 0:
            candidates.append((fold, baseline_value, cascade_value, delta))

    if not candidates:
        raise ValueError(f"No positive {metric} delta found for {dataset}")

    return max(candidates, key=lambda x: x[3])


def get_best_optuna_params(exp_name: str, fold: int, db_path: Path) -> Dict[str, float]:
    params: Dict[str, float] = {}
    if not db_path.exists():
        return params

    study_name = f"{exp_name} fold_{fold}"
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT t.trial_id, v.value
            FROM studies s
            JOIN trials t ON s.study_id = t.study_id
            JOIN trial_values v ON t.trial_id = v.trial_id
            WHERE s.study_name = ? AND t.state = 'COMPLETE'
            ORDER BY v.value DESC
            LIMIT 1
            """,
            (study_name,),
        )
        best = cur.fetchone()
        if not best:
            return params

        trial_id = int(best[0])
        cur.execute(
            "SELECT param_name, param_value FROM trial_params WHERE trial_id = ?",
            (trial_id,),
        )
        for name, value in cur.fetchall():
            if isinstance(value, (int, float)):
                params[name] = value
    finally:
        conn.close()

    if "lora_r" in params and "lora_alpha" not in params:
        lora_alpha_ratio = params.get("lora_alpha_ratio", 2.0)
        params["lora_alpha"] = int(round(params["lora_r"] * lora_alpha_ratio))
    return params


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def build_model(exp_dir: Path, fold: int, device: torch.device) -> torch.nn.Module:
    exp_name = exp_dir.name
    checkpoint = exp_dir / f"fold_{fold}" / f"model_weights_{fold}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    exp_name_lower = exp_name.lower()
    if "lssegsamlora" in exp_name_lower:
        db_params = get_best_optuna_params(exp_name, fold, PROJECT_ROOT / "optuna.db")
        params = {
            "lora_r": int(round(db_params.get("lora_r", 4))),
            "lora_alpha": int(round(db_params.get("lora_alpha", 8))),
            "freeze_lsseg": bool(db_params.get("freeze_lsseg", False)),
            "prompt_bias": float(db_params.get("prompt_bias", 0.0)),
            "box_bias": float(db_params.get("box_bias", 0.0)),
            "box_expand_ratio": float(db_params.get("box_expand_ratio", 0.02)),
            "residual_init_alpha": float(db_params.get("residual_init_alpha", 0.3)),
        }
        model = LSSegSAMLoRA(
            lsseg_checkpoint=None,
            sam_checkpoint=str(PROJECT_ROOT / "sam_vit_b_01ec64.pth"),
            target_size=512,
            lora_r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            freeze_lsseg=params["freeze_lsseg"],
            use_box_prompt=True,
            prompt_bias=params["prompt_bias"],
            box_bias=params["box_bias"],
            box_expand_ratio=params["box_expand_ratio"],
            residual_init_alpha=params["residual_init_alpha"],
        )
    elif "lsseg" in exp_name_lower:
        model = LSSeg(in_channels=[3, 8, 8])
    else:
        raise ValueError(f"Unsupported experiment type: {exp_name}")

    state_dict = torch.load(checkpoint, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected checkpoint format in {checkpoint}")
    model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model


def load_rgb_and_mask(image_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError(f"Failed to load {image_path} or {mask_path}")
    mask = (mask > 0).astype(np.uint8)
    return image, mask


def resize_sample(image_rgb: np.ndarray, mask: np.ndarray, size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    resized_image = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    resized_mask = (resized_mask > 0).astype(np.uint8)
    return resized_image, resized_mask


def run_inference(model: torch.nn.Module, image_rgb: np.ndarray, device: torch.device) -> np.ndarray:
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    if isinstance(model, LSSeg):
        image_tensor = image_tensor.to(device=device, dtype=torch.float32) / 255.0
    else:
        image_tensor = image_tensor.to(device=device, dtype=torch.uint8)
    np.random.seed(0)
    torch.manual_seed(0)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    return (probs >= 0.5).astype(np.uint8)


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, target).sum() / denom


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    image = image_rgb.astype(np.float32).copy()
    color_arr = np.asarray(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    image[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * color_arr
    return np.clip(image, 0, 255).astype(np.uint8)


def build_delta_map(gt: np.ndarray, baseline: np.ndarray, cascade: np.ndarray) -> np.ndarray:
    canvas = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    recovered = (gt == 1) & (baseline == 0) & (cascade == 1)
    lost = (gt == 1) & (baseline == 1) & (cascade == 0)
    canvas[recovered] = (0, 220, 0)
    canvas[lost] = (230, 60, 60)
    return canvas


def create_metric_figure(metric_names: Iterable[str], output_path: Path) -> None:
    datasets = list(SELECTED_EXPERIMENTS.keys())
    baseline_values = {metric: [] for metric in metric_names}
    cascade_values = {metric: [] for metric in metric_names}

    for dataset in datasets:
        pair = SELECTED_EXPERIMENTS[dataset]
        baseline_metrics = read_final_metrics(resolve_path(pair["baseline"]) / "final_metrics.csv")
        cascade_metrics = read_final_metrics(resolve_path(pair["cascade"]) / "final_metrics.csv")
        for metric in metric_names:
            baseline_values[metric].append(baseline_metrics[metric])
            cascade_values[metric].append(cascade_metrics[metric])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=180)
    axes = axes.flatten()
    x = np.arange(len(datasets))
    width = 0.34
    colors = {"baseline": "#4C72B0", "cascade": "#DD8452"}

    for ax, metric in zip(axes, metric_names):
        bvals = np.array(baseline_values[metric], dtype=float)
        cvals = np.array(cascade_values[metric], dtype=float)
        ax.bar(x - width / 2, bvals, width=width, label="LSSeg", color=colors["baseline"])
        ax.bar(x + width / 2, cvals, width=width, label="LSSegSAMLoRA", color=colors["cascade"])
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.25)

        for idx, (bv, cv) in enumerate(zip(bvals, cvals)):
            delta = cv - bv
            if metric in {"95HD", "ASSD", "MSE"}:
                delta_text = f"{delta:.3f}"
                y = max(bv, cv)
                va = "bottom"
            else:
                delta_text = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
                y = max(bv, cv)
                va = "bottom"
            ax.text(idx, y, delta_text, ha="center", va=va, fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("LSSeg vs LSSegSAMLoRA on Selected Experiments", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_vessel_figure(dataset: str, fold: int, sample_index: int, output_path: Path, device: torch.device) -> None:
    if dataset not in SELECTED_EXPERIMENTS:
        raise KeyError(f"Unknown dataset: {dataset}")

    pair = SELECTED_EXPERIMENTS[dataset]
    image_path, mask_path = get_test_sample(dataset, fold, sample_index)
    image_rgb, gt_mask = load_rgb_and_mask(image_path, mask_path)
    image_rgb, gt_mask = resize_sample(image_rgb, gt_mask, size=512)

    baseline_model = build_model(resolve_path(pair["baseline"]), fold, device)
    cascade_model = build_model(resolve_path(pair["cascade"]), fold, device)

    baseline_pred = run_inference(baseline_model, image_rgb, device)
    cascade_pred = run_inference(cascade_model, image_rgb, device)

    baseline_dice = dice_score(baseline_pred, gt_mask)
    cascade_dice = dice_score(cascade_pred, gt_mask)

    delta_map = build_delta_map(gt_mask, baseline_pred, cascade_pred)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180)
    axes = axes.ravel()

    panels = [
        (image_rgb, "Original Image"),
        (overlay_mask(image_rgb, gt_mask, (0, 220, 0)), "Ground Truth Overlay"),
        (overlay_mask(image_rgb, baseline_pred, (255, 170, 0)), f"LSSeg Overlay\nDice={baseline_dice:.3f}"),
        (overlay_mask(image_rgb, cascade_pred, (0, 180, 255)), f"LSSegSAMLoRA Overlay\nDice={cascade_dice:.3f}"),
        (gt_mask, "Ground Truth Mask"),
        (baseline_pred, "LSSeg Mask"),
        (cascade_pred, "LSSegSAMLoRA Mask"),
        (delta_map, "Recovered vs Lost\nGreen=Recovered, Red=Lost"),
    ]

    for ax, (panel, title) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    sample_name = image_path.name
    fig.suptitle(f"{dataset} | fold {fold} | sample {sample_index} | {sample_name}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_best_worst_showcase_figure(
    dataset: str,
    sample_index: int,
    output_path: Path,
    device: torch.device,
    metric: str = "DSC",
) -> None:
    if dataset not in SELECTED_EXPERIMENTS:
        raise KeyError(f"Unknown dataset: {dataset}")

    pair = SELECTED_EXPERIMENTS[dataset]
    baseline_worst_fold, baseline_worst_value, cascade_best_fold, cascade_best_value = select_extreme_folds(dataset, metric=metric)

    baseline_img_path, baseline_mask_path = get_test_sample(dataset, baseline_worst_fold, sample_index)
    cascade_img_path, cascade_mask_path = get_test_sample(dataset, cascade_best_fold, sample_index)

    baseline_image, baseline_gt = load_rgb_and_mask(baseline_img_path, baseline_mask_path)
    cascade_image, cascade_gt = load_rgb_and_mask(cascade_img_path, cascade_mask_path)
    baseline_image, baseline_gt = resize_sample(baseline_image, baseline_gt, size=512)
    cascade_image, cascade_gt = resize_sample(cascade_image, cascade_gt, size=512)

    baseline_model = build_model(resolve_path(pair["baseline"]), baseline_worst_fold, device)
    cascade_model = build_model(resolve_path(pair["cascade"]), cascade_best_fold, device)

    baseline_pred = run_inference(baseline_model, baseline_image, device)
    cascade_pred = run_inference(cascade_model, cascade_image, device)

    baseline_dice = dice_score(baseline_pred, baseline_gt)
    cascade_dice = dice_score(cascade_pred, cascade_gt)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180)
    axes = axes.reshape(2, 4)

    top_row = [
        (baseline_image, f"LSSeg worst fold image\nfold={baseline_worst_fold}, {metric}={baseline_worst_value:.3f}"),
        (overlay_mask(baseline_image, baseline_gt, (0, 220, 0)), "Ground Truth Overlay"),
        (overlay_mask(baseline_image, baseline_pred, (255, 170, 0)), f"LSSeg Overlay\nsample Dice={baseline_dice:.3f}"),
        (baseline_pred, "LSSeg Binary Mask"),
    ]
    bottom_row = [
        (cascade_image, f"LSSegSAMLoRA best fold image\nfold={cascade_best_fold}, {metric}={cascade_best_value:.3f}"),
        (overlay_mask(cascade_image, cascade_gt, (0, 220, 0)), "Ground Truth Overlay"),
        (overlay_mask(cascade_image, cascade_pred, (0, 180, 255)), f"LSSegSAMLoRA Overlay\nsample Dice={cascade_dice:.3f}"),
        (cascade_pred, "LSSegSAMLoRA Binary Mask"),
    ]

    for row_axes, row_panels in zip(axes, [top_row, bottom_row]):
        for ax, (panel, title) in zip(row_axes, row_panels):
            if panel.ndim == 2:
                ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(panel)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

    fig.suptitle(
        f"{dataset} qualitative showcase | worst LSSeg fold vs best LSSegSAMLoRA fold | sample-index={sample_index}",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_max_delta_paired_figure(
    dataset: str,
    fold: int,
    sample_index: int,
    output_path: Path,
    device: torch.device,
    metric: str = "DSC",
    show_suptitle: bool = False,
) -> None:
    pair = SELECTED_EXPERIMENTS[dataset]
    image_path, mask_path = get_test_sample(dataset, fold, sample_index)
    image_rgb, gt_mask = load_rgb_and_mask(image_path, mask_path)
    image_rgb, gt_mask = resize_sample(image_rgb, gt_mask, size=512)

    baseline_model = build_model(resolve_path(pair["baseline"]), fold, device)
    cascade_model = build_model(resolve_path(pair["cascade"]), fold, device)

    baseline_pred = run_inference(baseline_model, image_rgb, device)
    cascade_pred = run_inference(cascade_model, image_rgb, device)

    baseline_dice = dice_score(baseline_pred, gt_mask)
    cascade_dice = dice_score(cascade_pred, gt_mask)
    delta_map = build_delta_map(gt_mask, baseline_pred, cascade_pred)

    baseline_fold_metric = read_fold_metrics(resolve_path(pair["baseline"]) / f"fold_{fold}" / "metrics.csv")[metric]
    cascade_fold_metric = read_fold_metrics(resolve_path(pair["cascade"]) / f"fold_{fold}" / "metrics.csv")[metric]
    fold_delta = cascade_fold_metric - baseline_fold_metric

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=180)
    axes = axes.ravel()
    panels = [
        (image_rgb, "Original Image"),
        (overlay_mask(image_rgb, gt_mask, (0, 220, 0)), "Ground Truth Overlay"),
        (
            overlay_mask(image_rgb, baseline_pred, (255, 170, 0)),
            f"LSSeg Overlay\nsample Dice={baseline_dice:.3f}\nfold {metric}={baseline_fold_metric:.3f}",
        ),
        (
            overlay_mask(image_rgb, cascade_pred, (0, 180, 255)),
            f"LSSegSAMLoRA Overlay\nsample Dice={cascade_dice:.3f}\nfold {metric}={cascade_fold_metric:.3f}",
        ),
        (gt_mask, "Ground Truth Mask"),
        (baseline_pred, "LSSeg Binary Mask"),
        (cascade_pred, "LSSegSAMLoRA Binary Mask"),
        (delta_map, "Recovered vs Lost\nGreen=Recovered, Red=Lost"),
    ]

    for ax, (panel, title) in zip(axes, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    if show_suptitle:
        fig.suptitle(
            f"{dataset} | fold {fold} | max positive {metric} delta = {fold_delta:.3f} | sample-index={sample_index}",
            fontsize=14,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison charts for LSSeg vs LSSegSAMLoRA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metrics_parser = subparsers.add_parser("metrics", help="Generate the default quantitative comparison chart.")
    metrics_parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "figures" / "selected_log_metrics_comparison.png"),
        help="Output figure path.",
    )

    vessel_parser = subparsers.add_parser("vessel", help="Generate one qualitative vessel comparison panel.")
    vessel_parser.add_argument("--dataset", required=True, choices=list(SELECTED_EXPERIMENTS.keys()))
    vessel_parser.add_argument("--fold", type=int, default=0)
    vessel_parser.add_argument("--sample-index", type=int, default=0)
    vessel_parser.add_argument(
        "--output",
        default=None,
        help="Output figure path. Defaults to figures/vessel_comparisons/{dataset}_fold{fold}_sample{sample}.png",
    )
    vessel_parser.add_argument("--device", default=None, help="cuda, cuda:0 or cpu")

    batch_parser = subparsers.add_parser("batch-vessels", help="Generate one qualitative figure per vessel dataset.")
    batch_parser.add_argument("--fold", type=int, default=0)
    batch_parser.add_argument("--sample-index", type=int, default=0)
    batch_parser.add_argument("--datasets", nargs="*", default=list(VESSEL_DATASETS), choices=list(VESSEL_DATASETS))
    batch_parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "figures" / "vessel_comparisons"),
        help="Directory for output figures.",
    )
    batch_parser.add_argument("--device", default=None, help="cuda, cuda:0 or cpu")

    showcase_parser = subparsers.add_parser(
        "best-worst-showcase",
        help="Generate 5 qualitative figures using the worst LSSeg fold and best LSSegSAMLoRA fold per dataset.",
    )
    showcase_parser.add_argument("--sample-index", type=int, default=0)
    showcase_parser.add_argument("--datasets", nargs="*", default=list(SELECTED_EXPERIMENTS.keys()), choices=list(SELECTED_EXPERIMENTS.keys()))
    showcase_parser.add_argument("--metric", default="DSC", choices=["AP", "DSC", "mIoU", "clDice", "ODS"])
    showcase_parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "figures" / "best_worst_showcase"),
        help="Directory for output showcase figures.",
    )
    showcase_parser.add_argument("--device", default=None, help="cuda, cuda:0 or cpu")

    max_delta_parser = subparsers.add_parser(
        "max-delta-paired",
        help="Generate paired figures using the same fold where LSSegSAMLoRA has the largest positive DSC delta over LSSeg.",
    )
    max_delta_parser.add_argument("--sample-index", type=int, default=0)
    max_delta_parser.add_argument("--datasets", nargs="*", default=list(SELECTED_EXPERIMENTS.keys()), choices=list(SELECTED_EXPERIMENTS.keys()))
    max_delta_parser.add_argument("--metric", default="DSC", choices=["AP", "DSC", "mIoU", "clDice", "ODS"])
    max_delta_parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "figures" / "max_delta_paired"),
        help="Directory for output paired figures.",
    )
    max_delta_parser.add_argument("--device", default=None, help="cuda, cuda:0 or cpu")
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    if args.command == "metrics":
        create_metric_figure(DEFAULT_METRICS, resolve_path(args.output))
        print(f"Saved metric comparison figure to {resolve_path(args.output)}")
        return

    device = choose_device(getattr(args, "device", None))
    print(f"Using device: {device}")

    if args.command == "vessel":
        output = (
            resolve_path(args.output)
            if args.output
            else PROJECT_ROOT / "figures" / "vessel_comparisons" / f"{args.dataset}_fold{args.fold}_sample{args.sample_index}.png"
        )
        create_vessel_figure(args.dataset, args.fold, args.sample_index, output, device)
        print(f"Saved vessel comparison figure to {output}")
        return

    if args.command == "batch-vessels":
        output_dir = resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            output = output_dir / f"{dataset}_fold{args.fold}_sample{args.sample_index}.png"
            create_vessel_figure(dataset, args.fold, args.sample_index, output, device)
            print(f"Saved {dataset} vessel comparison to {output}")
        return

    if args.command == "best-worst-showcase":
        output_dir = resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            output = output_dir / f"{dataset}_best_worst_showcase.png"
            bwf, bwv, cbf, cbv = select_extreme_folds(dataset, metric=args.metric)
            print(
                f"{dataset}: LSSeg worst fold={bwf} ({args.metric}={bwv:.3f}), "
                f"LSSegSAMLoRA best fold={cbf} ({args.metric}={cbv:.3f})"
            )
            create_best_worst_showcase_figure(
                dataset=dataset,
                sample_index=args.sample_index,
                output_path=output,
                device=device,
                metric=args.metric,
            )
            print(f"Saved showcase figure to {output}")
        return

    if args.command == "max-delta-paired":
        output_dir = resolve_path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for dataset in args.datasets:
            fold, baseline_value, cascade_value, delta = select_max_positive_delta_fold(dataset, metric=args.metric)
            print(
                f"{dataset}: fold={fold}, "
                f"LSSeg {args.metric}={baseline_value:.3f}, "
                f"LSSegSAMLoRA {args.metric}={cascade_value:.3f}, "
                f"delta={delta:.3f}"
            )
            output = output_dir / f"{dataset}_fold{fold}_max_delta_paired.png"
            create_max_delta_paired_figure(
                dataset=dataset,
                fold=fold,
                sample_index=args.sample_index,
                output_path=output,
                device=device,
                metric=args.metric,
            )
            print(f"Saved paired figure to {output}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
