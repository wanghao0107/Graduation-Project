



from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_LAUNCH_FLAG = "DATASET_DASHBOARD_STREAMLIT_LAUNCHED"


def should_relaunch_with_streamlit() -> bool:
    if __name__ != "__main__" or os.environ.get(STREAMLIT_LAUNCH_FLAG) == "1":
        return False
    return Path(sys.argv[0]).resolve() == Path(__file__).resolve()


if should_relaunch_with_streamlit():
    env = os.environ.copy()
    env[STREAMLIT_LAUNCH_FLAG] = "1"
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, env=env))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.morphology import skeletonize
from sklearn.metrics import average_precision_score


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.lsseg import LSSeg
from models.lsseg_sam_lora import LSSegSAMLoRA


IMAGE_SIZE = 512
METRIC_COLUMNS = ["AP", "DSC", "mIoU", "clDice", "ASSD", "95HD", "ODS"]
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


st.set_page_config(
    page_title="生物医学线状结构分割分析",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17212b;
        --muted: #607080;
        --line: #d9e0e7;
        --panel: #f8fafb;
        --accent: #0f766e;
        --accent-soft: #d9f2ee;
        --warn: #b45309;
    }
    .main .block-container {
        padding-top: 1.6rem;
        max-width: 1380px;
    }
    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: 0;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, var(--panel) 100%);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }
    .section-note {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.55;
        margin: -0.35rem 0 0.8rem;
    }
    .result-strip {
        border-left: 4px solid var(--accent);
        padding: 0.6rem 0.8rem;
        background: var(--accent-soft);
        border-radius: 0 8px 8px 0;
        color: var(--ink);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(str(path_like).strip())
    return path if path.is_absolute() else PROJECT_ROOT / path


def dataset_name_from_csv(csv_path: Path) -> str:
    return csv_path.stem.replace("idx_", "")


@st.cache_data(show_spinner=False)
def list_dataset_indices() -> Dict[str, str]:
    indices: Dict[str, str] = {}
    for dataset_name in SELECTED_EXPERIMENTS:
        csv_path = PROJECT_ROOT / "data" / f"idx_{dataset_name}.csv"
        if csv_path.exists():
            indices[dataset_name] = str(csv_path)
    return indices


@st.cache_data(show_spinner=False)
def read_index_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=["image_path", "mask_path"])
    df["image_abs"] = df["image_path"].map(lambda p: str(resolve_path(p)))
    df["mask_abs"] = df["mask_path"].map(lambda p: str(resolve_path(p)))
    df["image_name"] = df["image_path"].map(lambda p: Path(str(p)).name)
    df["mask_name"] = df["mask_path"].map(lambda p: Path(str(p)).name)
    return df


def load_image_rgb(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取标注: {path}")
    return (mask > 0).astype(np.uint8)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: int = IMAGE_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    image_resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return image_resized, (mask_resized > 0).astype(np.uint8)


@st.cache_data(show_spinner=False)
def summarize_dataset(csv_path: str, scan_limit: int) -> Dict[str, object]:
    df = read_index_csv(csv_path)
    sizes: List[Tuple[int, int]] = []
    fg_ratios: List[float] = []
    missing = 0

    for _, row in df.head(scan_limit).iterrows():
        image = cv2.imread(row["image_abs"], cv2.IMREAD_COLOR)
        mask = cv2.imread(row["mask_abs"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            missing += 1
            continue
        h, w = image.shape[:2]
        sizes.append((w, h))
        fg_ratios.append(float((mask > 0).mean()))

    unique_sizes = sorted(set(sizes))
    return {
        "count": int(len(df)),
        "scanned": int(min(scan_limit, len(df))),
        "missing": int(missing),
        "unique_size_count": int(len(unique_sizes)),
        "sizes_preview": unique_sizes[:8],
        "fg_mean": float(np.mean(fg_ratios)) if fg_ratios else 0.0,
        "fg_min": float(np.min(fg_ratios)) if fg_ratios else 0.0,
        "fg_max": float(np.max(fg_ratios)) if fg_ratios else 0.0,
    }


@st.cache_data(show_spinner=False)
def list_metric_files() -> pd.DataFrame:
    records = []
    for dataset_name, pair in SELECTED_EXPERIMENTS.items():
        for role, exp_path in pair.items():
            exp_dir = resolve_path(exp_path)
            final_metrics = exp_dir / "final_metrics.csv"
            if not final_metrics.exists():
                continue
            records.append(
                {
                    "experiment": exp_dir.name,
                    "role": "LSSeg" if role == "baseline" else "LSSegSAMLoRA",
                    "path": str(final_metrics),
                    "dataset_hint": dataset_name,
                }
            )
    return pd.DataFrame(records)


def infer_dataset_from_experiment(name: str) -> str:
    known = [
        "AxonDeepSeg_SEM",
        "CHASE_DB1",
        "STARE",
        "RITE",
        "HRF",
        "FIVES",
        "ARCADE",
        "GlaS",
        "LUMINOUS",
        "CREMI",
        "Microtubule",
        "FALLMUD",
        "Synthetic_SMG",
    ]
    lowered = name.lower()
    for dataset in known:
        if dataset.lower() in lowered:
            return dataset
    return "未识别"


@st.cache_data(show_spinner=False)
def read_final_metric_file(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if {"Mean", "Std Dev"}.issubset(raw.columns):
        metric_col = raw.columns[0]
        return raw.rename(columns={metric_col: "Metric"})
    return raw


@st.cache_data(show_spinner=False)
def list_checkpoint_files() -> pd.DataFrame:
    records = []
    for dataset_name, pair in SELECTED_EXPERIMENTS.items():
        for role, exp_path in pair.items():
            exp_dir = resolve_path(exp_path)
            for ckpt in sorted(exp_dir.glob("fold_*/model_weights_*.pth")):
                fold_match = re.search(r"fold_(\d+)", str(ckpt.parent))
                model_hint = "LSSeg" if role == "baseline" else "LSSegSAMLoRA"
                records.append(
                    {
                        "label": f"{dataset_name} / {model_hint} / {ckpt.parent.name}",
                        "experiment": exp_dir.name,
                        "fold": int(fold_match.group(1)) if fold_match else -1,
                        "path": str(ckpt),
                        "dataset_hint": dataset_name,
                        "model_hint": model_hint,
                    }
                )
    return pd.DataFrame(records)


def infer_model_from_experiment(name: str) -> str:
    lowered = name.lower()
    if "lssegsamlora" in lowered or "lsseg+samlora" in lowered:
        return "LSSegSAMLoRA"
    if "lsseg" in lowered:
        return "LSSeg"
    return "未知"


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


@st.cache_resource(show_spinner=False)
def load_model(
    model_type: str,
    checkpoint_path: str,
    sam_checkpoint: str,
    device_name: str,
    lora_r: int,
    lora_alpha: int,
    prompt_bias: float,
    box_bias: float,
    box_expand_ratio: float,
    residual_init_alpha: float,
) -> torch.nn.Module:
    device = torch.device(device_name)
    if model_type == "LSSeg":
        model = LSSeg(in_channels=[3, 8, 8])
    elif model_type == "LSSegSAMLoRA":
        model = LSSegSAMLoRA(
            lsseg_checkpoint=None,
            sam_checkpoint=sam_checkpoint,
            target_size=IMAGE_SIZE,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            freeze_lsseg=False,
            use_box_prompt=True,
            prompt_bias=prompt_bias,
            box_bias=box_bias,
            box_expand_ratio=box_expand_ratio,
            residual_init_alpha=residual_init_alpha,
        )
    else:
        raise ValueError(f"当前界面暂不支持模型类型: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint 格式不是 state_dict 字典")
    model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference(model: torch.nn.Module, model_type: str, image_rgb: np.ndarray, device_name: str) -> np.ndarray:
    device = torch.device(device_name)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    np.random.seed(0)
    torch.manual_seed(0)

    with torch.no_grad():
        if model_type == "LSSeg":
            image_tensor = image_tensor.to(device=device, dtype=torch.float32) / 255.0
        else:
            image_tensor = image_tensor.to(device=device, dtype=torch.uint8)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    return probs.astype(np.float32)


def binary_surface(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return mask_bool
    return mask_bool ^ binary_erosion(mask_bool)


def surface_distances(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_surface = binary_surface(pred)
    target_surface = binary_surface(target)
    if not pred_surface.any() or not target_surface.any():
        return np.array([], dtype=np.float32)
    target_distance = distance_transform_edt(~target_surface)
    pred_distance = distance_transform_edt(~pred_surface)
    distances = np.concatenate([target_distance[pred_surface], pred_distance[target_surface]])
    return distances.astype(np.float32)


def cldice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    pred_skeleton = skeletonize(pred_bool)
    target_skeleton = skeletonize(target_bool)
    if not pred_skeleton.any() or not target_skeleton.any():
        return 1.0 if pred_bool.sum() == target_bool.sum() == 0 else 0.0
    tprec = np.logical_and(pred_skeleton, target_bool).sum() / (pred_skeleton.sum() + 1e-8)
    tsens = np.logical_and(target_skeleton, pred_bool).sum() / (target_skeleton.sum() + 1e-8)
    return float(2.0 * tprec * tsens / (tprec + tsens + 1e-8))


def ods_score(probs: np.ndarray, target: np.ndarray, thresholds: int = 100) -> float:
    best = 0.0
    target_bool = target.astype(bool)
    for threshold in np.linspace(0.0, 1.0, thresholds):
        pred_bool = probs >= threshold
        tp = np.logical_and(pred_bool, target_bool).sum()
        fp = np.logical_and(pred_bool, ~target_bool).sum()
        fn = np.logical_and(~pred_bool, target_bool).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        best = max(best, float(f1))
    return best


def compute_metrics(probs: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    target = target.astype(np.uint8)
    pred = (probs >= threshold).astype(np.uint8)
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)

    tp = np.logical_and(pred_bool, target_bool).sum()
    fp = np.logical_and(pred_bool, ~target_bool).sum()
    fn = np.logical_and(~pred_bool, target_bool).sum()
    denom_dice = pred_bool.sum() + target_bool.sum()
    union = np.logical_or(pred_bool, target_bool).sum()
    distances = surface_distances(pred, target)

    try:
        ap = float(average_precision_score(target.reshape(-1), probs.reshape(-1)))
    except ValueError:
        ap = 0.0

    return {
        "AP": ap,
        "DSC": float(2.0 * tp / (denom_dice + 1e-8)),
        "mIoU": float(tp / (union + 1e-8)),
        "clDice": cldice_score(pred, target),
        "ASSD": float(distances.mean()) if distances.size else 0.0,
        "95HD": float(np.percentile(distances, 95)) if distances.size else 0.0,
        "ODS": ods_score(probs, target),
    }


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.48) -> np.ndarray:
    result = image.astype(np.float32).copy()
    mask_bool = mask.astype(bool)
    color_arr = np.asarray(color, dtype=np.float32)
    result[mask_bool] = (1.0 - alpha) * result[mask_bool] + alpha * color_arr
    return np.clip(result, 0, 255).astype(np.uint8)


def error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    canvas = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.uint8)
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    canvas[np.logical_and(pred_bool, target_bool)] = (45, 180, 90)
    canvas[np.logical_and(pred_bool, ~target_bool)] = (230, 80, 70)
    canvas[np.logical_and(~pred_bool, target_bool)] = (235, 185, 45)
    return canvas


def render_metric_cards(metrics: Dict[str, float]) -> None:
    cols = st.columns(len(METRIC_COLUMNS))
    for col, name in zip(cols, METRIC_COLUMNS):
        value = metrics.get(name, 0.0)
        col.metric(name, f"{value:.4f}")


def render_dataset_overview(dataset_name: str, df: pd.DataFrame, summary: Dict[str, object]) -> None:
    st.subheader("数据集概览")
    st.markdown(
        f"<div class='section-note'>当前选择 `{dataset_name}`。统计信息来自索引文件和前若干样本扫描，可用于快速检查数据规模、尺寸和前景占比。</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    cols[0].metric("样本数", f"{summary['count']}")
    cols[1].metric("扫描样本", f"{summary['scanned']}")
    cols[2].metric("缺失文件", f"{summary['missing']}")
    cols[3].metric("尺寸种类", f"{summary['unique_size_count']}")
    cols[4].metric("平均前景占比", f"{summary['fg_mean'] * 100:.2f}%")

    size_text = ", ".join([f"{w}x{h}" for w, h in summary["sizes_preview"]]) or "暂无"
    st.caption(f"尺寸预览: {size_text}")
    st.dataframe(df[["image_path", "mask_path"]].head(30), use_container_width=True, hide_index=True)


def render_sample_view(df: pd.DataFrame, sample_index: int) -> Tuple[np.ndarray, np.ndarray]:
    row = df.iloc[sample_index]
    image = load_image_rgb(row["image_abs"])
    mask = load_mask(row["mask_abs"])
    image_resized, mask_resized = resize_pair(image, mask)

    st.subheader("样本预览")
    st.caption(f"{row['image_name']} | 原始尺寸 {image.shape[1]}x{image.shape[0]} | 前景占比 {mask.mean() * 100:.2f}%")
    cols = st.columns(3)
    cols[0].image(image, caption="原始图像", use_container_width=True)
    cols[1].image(mask * 255, caption="真实标注", use_container_width=True, clamp=True)
    cols[2].image(overlay_mask(image, mask, (20, 180, 120)), caption="标注叠加", use_container_width=True)
    return image_resized, mask_resized


def render_history_metrics(selected_dataset: str) -> None:
    st.subheader("历史实验指标")
    metric_files = list_metric_files()
    if metric_files.empty:
        st.info("未发现 log/*/final_metrics.csv。")
        return

    visible = metric_files[metric_files["dataset_hint"] == selected_dataset]
    if visible.empty:
        st.info("当前 5 个指定实验中没有找到该数据集的 final_metrics.csv。")
        return

    visible = visible.copy()
    visible["label"] = visible["role"] + " | " + visible["experiment"]
    exp_label = st.selectbox("选择历史实验", visible["label"].tolist(), key="history_exp")
    path = visible.loc[visible["label"] == exp_label, "path"].iloc[0]
    metrics_df = read_final_metric_file(path)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)


def render_inference_panel(selected_dataset: str, df: pd.DataFrame, image: np.ndarray, mask: np.ndarray) -> None:
    st.subheader("分割结果与评价指标")
    st.markdown(
        "<div class='section-note'>选择训练日志中的权重后，可以对当前样本推理，也可以对当前数据集前 N 个样本做快速批量评估。</div>",
        unsafe_allow_html=True,
    )

    ckpts = list_checkpoint_files()
    ckpts = ckpts[ckpts["dataset_hint"] == selected_dataset]

    if ckpts.empty:
        st.info("当前 5 个指定实验中没有找到该数据集的模型权重。")
        return

    left, right = st.columns([1.2, 0.8])
    with left:
        label = st.selectbox("选择模型权重", ckpts["label"].tolist())
        record = ckpts.loc[ckpts["label"] == label].iloc[0]
        model_type = st.selectbox("模型类型", ["自动识别", "LSSeg", "LSSegSAMLoRA"], index=0)
        if model_type == "自动识别":
            model_type = record["model_hint"]

    with right:
        threshold = st.slider("二值化阈值", 0.05, 0.95, 0.50, 0.05)
        device_default = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = st.selectbox("推理设备", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"], index=0 if device_default == "cpu" else 1)

    sam_checkpoint = str(PROJECT_ROOT / "sam_vit_b_01ec64.pth")
    if model_type == "LSSegSAMLoRA":
        with st.expander("LSSegSAMLoRA 参数", expanded=False):
            sam_checkpoint = st.text_input("SAM checkpoint", sam_checkpoint)
            c1, c2, c3 = st.columns(3)
            lora_r = c1.number_input("lora_r", min_value=1, max_value=64, value=4, step=1)
            lora_alpha = c2.number_input("lora_alpha", min_value=1, max_value=128, value=8, step=1)
            residual_init_alpha = c3.number_input("residual_init_alpha", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
            c4, c5, c6 = st.columns(3)
            prompt_bias = c4.number_input("prompt_bias", min_value=-2.0, max_value=4.0, value=0.0, step=0.25)
            box_bias = c5.number_input("box_bias", min_value=-2.0, max_value=4.0, value=0.0, step=0.25)
            box_expand_ratio = c6.number_input("box_expand_ratio", min_value=0.0, max_value=0.5, value=0.02, step=0.01)
    else:
        lora_r, lora_alpha = 4, 8
        prompt_bias, box_bias, box_expand_ratio, residual_init_alpha = 0.0, 0.0, 0.02, 0.3

    run_single = st.button("运行当前样本推理", type="primary", use_container_width=True)
    if run_single:
        try:
            with st.spinner("正在加载模型并推理..."):
                model = load_model(
                    model_type=model_type,
                    checkpoint_path=record["path"],
                    sam_checkpoint=sam_checkpoint,
                    device_name=device_name,
                    lora_r=int(lora_r),
                    lora_alpha=int(lora_alpha),
                    prompt_bias=float(prompt_bias),
                    box_bias=float(box_bias),
                    box_expand_ratio=float(box_expand_ratio),
                    residual_init_alpha=float(residual_init_alpha),
                )
                probs = run_inference(model, model_type, image, device_name)
                pred = (probs >= threshold).astype(np.uint8)
                metrics = compute_metrics(probs, mask, threshold=threshold)

            render_metric_cards(metrics)
            cols = st.columns(4)
            cols[0].image(image, caption="输入图像", use_container_width=True)
            cols[1].image(probs, caption="预测概率图", use_container_width=True, clamp=True)
            cols[2].image(overlay_mask(image, pred, (0, 155, 220)), caption="预测叠加", use_container_width=True)
            cols[3].image(error_map(pred, mask), caption="误差图 绿=TP 红=FP 黄=FN", use_container_width=True)
        except Exception as exc:
            st.error(f"推理失败: {exc}")

    with st.expander("批量快速评估", expanded=False):
        limit = st.slider("评估样本数", 1, min(50, len(df)), min(10, len(df)), 1)
        if st.button("评估前 N 个样本", use_container_width=True):
            rows = []
            try:
                model = load_model(
                    model_type=model_type,
                    checkpoint_path=record["path"],
                    sam_checkpoint=sam_checkpoint,
                    device_name=device_name,
                    lora_r=int(lora_r),
                    lora_alpha=int(lora_alpha),
                    prompt_bias=float(prompt_bias),
                    box_bias=float(box_bias),
                    box_expand_ratio=float(box_expand_ratio),
                    residual_init_alpha=float(residual_init_alpha),
                )
                progress = st.progress(0.0)
                for idx, row in df.head(limit).iterrows():
                    img = load_image_rgb(row["image_abs"])
                    gt = load_mask(row["mask_abs"])
                    img, gt = resize_pair(img, gt)
                    probs = run_inference(model, model_type, img, device_name)
                    item_metrics = compute_metrics(probs, gt, threshold=threshold)
                    item_metrics["sample"] = row["image_name"]
                    rows.append(item_metrics)
                    progress.progress((len(rows)) / limit)
                result = pd.DataFrame(rows)
                mean_row = result[METRIC_COLUMNS].mean().to_frame("Mean").T
                st.markdown("<div class='result-strip'>批量评估完成，下面给出均值和逐样本结果。</div>", unsafe_allow_html=True)
                st.dataframe(mean_row, use_container_width=True)
                st.dataframe(result[["sample"] + METRIC_COLUMNS], use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"批量评估失败: {exc}")


def main() -> None:
    st.title("生物医学线状结构分割分析面板")
    st.caption("数据集检查、历史指标浏览、模型推理和评价指标计算")

    indices = list_dataset_indices()
    if not indices:
        st.error("未找到 data/idx_*.csv。")
        return

    with st.sidebar:
        st.header("数据选择")
        dataset_options = [name for name in SELECTED_EXPERIMENTS if name in indices]
        dataset_name = st.selectbox("数据集", dataset_options, index=dataset_options.index("STARE") if "STARE" in dataset_options else 0)
        scan_limit = st.slider("统计扫描样本数", 5, 200, 50, 5)
        df = read_index_csv(indices[dataset_name])
        sample_index = st.slider("样本序号", 0, max(0, len(df) - 1), 0, 1)
        st.divider()
        page = st.radio("视图", ["数据集与样本", "历史实验指标", "分割推理评估"], label_visibility="collapsed")

    summary = summarize_dataset(indices[dataset_name], scan_limit=scan_limit)

    if page == "数据集与样本":
        render_dataset_overview(dataset_name, df, summary)
        render_sample_view(df, sample_index)
    elif page == "历史实验指标":
        render_dataset_overview(dataset_name, df, summary)
        render_history_metrics(dataset_name)
    else:
        image, mask = render_sample_view(df, sample_index)
        render_inference_panel(dataset_name, df, image, mask)


if __name__ == "__main__":
    main()
