import os
import optuna
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold, train_test_split
from monai.metrics import DiceMetric
from datetime import datetime
import gc
from contextlib import contextmanager
from tqdm import tqdm

# 假设这些是你本地的文件，保持不变
from models.unet import UNet
from models.lsseg import LSSeg
from models.bdcn import BDCN
from models.cats import CATS
from models.teed import TEED
from models.condseg import ConDSeg
from models.fsgnet import FSGNet
from models.transunet import TransUNet
from models.sam_lora import SAMLoRA
from models.medsam import MedSAM
from models.lsseg_sam_lora import LSSegSAMLoRA, LSSegSAMLoRA_Simple
from models.lsseg_medsam import LSSegMedSAM, LSSegMedSAM_Simple  # 【新增】
from loss import tracing_loss, dice_ce_loss
from dataset import ImageSegDataset, read_index_csv
from eval import evaluate
from utils import aggregate_metrics, plot_curve


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 模型选择开关：切换模型时只需修改这一个变量
# 可选值：'SAMLoRA' | 'MedSAM' | 'LSSeg' | 'UNet' | 'TEED' | 'CATS' |
#         'BDCN' | 'ConDSeg' | 'FSGNet' | 'TransUNet' |
#         'LSSegSAMLoRA' | 'LSSegSAMLoRA_Simple' |
#         'LSSegMedSAM' | 'LSSegMedSAM_Simple'
# ============================================================
CURRENT_MODEL = 'FSGNet'

# ============================================================
# LSSegSAMLoRA 专用配置：手动指定 LSSeg 权重路径
# 设为 None 则自动查找 log/ 目录下最新的权重
# ============================================================
LSSEG_CHECKPOINT_PATH = "log/test_RITE_LSSeg 02-14 08_30/fold_4/model_weights_4.pth"  # 例如: "log/test_STARE_LSSeg 02-13 11_34/fold_0/model_weights_0.pth"


def suggest_params(trial):
    if CURRENT_MODEL in ['SAMLoRA', 'LSSegSAMLoRA', 'LSSegSAMLoRA_Simple', 'LSSegMedSAM', 'LSSegMedSAM_Simple']:
        lora_r = trial.suggest_categorical("lora_r", [2, 4, 8, 16])
        lora_alpha_ratio = trial.suggest_categorical("lora_alpha_ratio", [1, 2])  # alpha = r * ratio

        # 基础参数
        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "lr_factor": trial.suggest_float("lr_factor", 0.5, 0.9, step=0.1),
            "step_size": trial.suggest_int("step_size", 10, 20, step=5),
            "lora_r": lora_r,
            "lora_alpha": lora_r * lora_alpha_ratio,
        }

        # 【新增】LSSegSAMLoRA / LSSegMedSAM 专用参数：prompt_bias 控制召回率
        if CURRENT_MODEL in ['LSSegSAMLoRA', 'LSSegSAMLoRA_Simple', 'LSSegMedSAM', 'LSSegMedSAM_Simple']:
            # prompt_bias: 在 logits 上加偏置，等效于降低阈值，提升召回率
            # 0.0 = 原始行为，值越大越宽松（召回率越高）
            params["prompt_bias"] = trial.suggest_float("prompt_bias", 0.0, 3.0, step=0.5)
            # 【新增】端到端联合训练：是否冻结 LSSeg
            # True = 仅训练 SAM-LoRA（默认，稳定）
            # False = 端到端联合训练（梯度回传到 LSSeg，潜在提升更大但需要调参）
            # 当前设置：强制使用端到端训练
            params["freeze_lsseg"] = trial.suggest_categorical("freeze_lsseg", [False])  # 强制端到端
            # 【新增】端到端训练时的 LSSeg 学习率比例
            # 仅当 freeze_lsseg=False 时有效，LSSeg 学习率 = lr * lsseg_lr_ratio
            params["lsseg_lr_ratio"] = trial.suggest_categorical("lsseg_lr_ratio", [0.05, 0.1, 0.2])
        else:
            params["prompt_bias"] = 0.0  # SAMLoRA 不使用此参数
            params["freeze_lsseg"] = True
            params["lsseg_lr_ratio"] = 0.1

        # 【新增】LSSegMedSAM 专用参数：Box Prompt 优化
        if CURRENT_MODEL == 'LSSegMedSAM':
            # box_bias: Box 生成时的 logits 偏置，控制 Box 召回率
            params["box_bias"] = trial.suggest_float("box_bias", 0.0, 2.0, step=0.5)
            # box_expand_ratio: Box 扩展比例（占图像尺寸的比例）
            params["box_expand_ratio"] = trial.suggest_float("box_expand_ratio", 0.0, 0.1, step=0.02)
        else:
            params["box_bias"] = 0.0
            params["box_expand_ratio"] = 0.02

        return params
    elif CURRENT_MODEL == 'MedSAM':
        # MedSAM 可选 LoRA：lora_r=0 表示不使用 LoRA（原始 MedSAM 行为）
        lora_r = trial.suggest_categorical("lora_r", [0, 2, 4, 8])
        lora_alpha_ratio = trial.suggest_categorical("lora_alpha_ratio", [1, 2])  # alpha = r * ratio
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "lr_factor": trial.suggest_float("lr_factor", 0.5, 0.9, step=0.1),
            "step_size": trial.suggest_int("step_size", 10, 20, step=5),
            "lora_r": lora_r,
            "lora_alpha": lora_r * lora_alpha_ratio if lora_r > 0 else 4,  # lora_r=0 时不使用 LoRA
        }
    else:
        return {
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            "lr_factor": trial.suggest_float("lr_factor", 0.5, 0.8, step=0.1),
            "step_size": trial.suggest_int("step_size", 5, 11, step=2),
        }


def build_model(**params):
    if CURRENT_MODEL == 'LSSegSAMLoRA':
        # 确定 LSSeg 权重路径
        if LSSEG_CHECKPOINT_PATH is not None:
            # 手动指定路径
            lsseg_checkpoint = LSSEG_CHECKPOINT_PATH
            print(f"Using specified LSSeg checkpoint: {lsseg_checkpoint}")
        else:
            # 自动查找最新的 LSSeg 权重
            import glob
            lsseg_checkpoints = glob.glob("log/test_STARE_LSSeg*/model_weights_*.pth")
            if lsseg_checkpoints:
                lsseg_checkpoint = max(lsseg_checkpoints, key=os.path.getmtime)
                print(f"Using latest LSSeg checkpoint: {lsseg_checkpoint}")
            else:
                lsseg_checkpoint = None
                print("Warning: No LSSeg checkpoint found, will use random initialization.")

        model = LSSegSAMLoRA(
            lsseg_checkpoint=lsseg_checkpoint,
            sam_checkpoint="sam_vit_b_01ec64.pth",
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 4),
            freeze_lsseg=params.get("freeze_lsseg", True),  # 【修改】支持端到端训练
            use_box_prompt=True,
            prompt_bias=params.get("prompt_bias", 0.0),
        )
    elif CURRENT_MODEL == 'LSSegSAMLoRA_Simple':
        # 确定 LSSeg 权重路径
        if LSSEG_CHECKPOINT_PATH is not None:
            lsseg_checkpoint = LSSEG_CHECKPOINT_PATH
            print(f"Using specified LSSeg checkpoint: {lsseg_checkpoint}")
        else:
            import glob
            lsseg_checkpoints = glob.glob("log/test_STARE_LSSeg*/model_weights_*.pth")
            if lsseg_checkpoints:
                lsseg_checkpoint = max(lsseg_checkpoints, key=os.path.getmtime)
                print(f"Using latest LSSeg checkpoint: {lsseg_checkpoint}")
            else:
                lsseg_checkpoint = None
                print("Warning: No LSSeg checkpoint found, will use random initialization.")

        model = LSSegSAMLoRA_Simple(
            lsseg_checkpoint=lsseg_checkpoint,
            sam_checkpoint="sam_vit_b_01ec64.pth",
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 4),
            freeze_lsseg=params.get("freeze_lsseg", True),  # 【修改】支持端到端训练
            prompt_bias=params.get("prompt_bias", 0.0),
        )
    elif CURRENT_MODEL == 'LSSegMedSAM':
        # 确定 LSSeg 权重路径
        if LSSEG_CHECKPOINT_PATH is not None:
            lsseg_checkpoint = LSSEG_CHECKPOINT_PATH
            print(f"Using specified LSSeg checkpoint: {lsseg_checkpoint}")
        else:
            import glob
            lsseg_checkpoints = glob.glob("log/test_STARE_LSSeg*/model_weights_*.pth")
            if lsseg_checkpoints:
                lsseg_checkpoint = max(lsseg_checkpoints, key=os.path.getmtime)
                print(f"Using latest LSSeg checkpoint: {lsseg_checkpoint}")
            else:
                lsseg_checkpoint = None
                print("Warning: No LSSeg checkpoint found, will use random initialization.")

        model = LSSegMedSAM(
            lsseg_checkpoint=lsseg_checkpoint,
            sam_checkpoint="medsam_vit_b.pth",  # 【关键】使用 MedSAM 权重
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 4),
            freeze_lsseg=True,
            use_box_prompt=True,
            prompt_bias=params.get("prompt_bias", 0.0),
            box_bias=params.get("box_bias", 0.0),  # 【新增】Box 生成偏置
            box_expand_ratio=params.get("box_expand_ratio", 0.02),  # 【新增】Box 扩展比例
        )
    elif CURRENT_MODEL == 'LSSegMedSAM_Simple':
        # 确定 LSSeg 权重路径
        if LSSEG_CHECKPOINT_PATH is not None:
            lsseg_checkpoint = LSSEG_CHECKPOINT_PATH
            print(f"Using specified LSSeg checkpoint: {lsseg_checkpoint}")
        else:
            import glob
            lsseg_checkpoints = glob.glob("log/test_STARE_LSSeg*/model_weights_*.pth")
            if lsseg_checkpoints:
                lsseg_checkpoint = max(lsseg_checkpoints, key=os.path.getmtime)
                print(f"Using latest LSSeg checkpoint: {lsseg_checkpoint}")
            else:
                lsseg_checkpoint = None
                print("Warning: No LSSeg checkpoint found, will use random initialization.")

        model = LSSegMedSAM_Simple(
            lsseg_checkpoint=lsseg_checkpoint,
            sam_checkpoint="medsam_vit_b.pth",
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 4),
            freeze_lsseg=True,
            prompt_bias=params.get("prompt_bias", 0.0),
        )
    elif CURRENT_MODEL == 'SAMLoRA':
        model = SAMLoRA(
            checkpoint_path="sam_vit_b_01ec64.pth",
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 4)
        )
    elif CURRENT_MODEL == 'MedSAM':
        model = MedSAM(
            checkpoint_path="medsam_vit_b.pth",
            target_size=512,
            lora_r=params.get("lora_r", 0),
            lora_alpha=params.get("lora_alpha", 4)
        )
    elif CURRENT_MODEL == 'LSSeg':
        model = LSSeg(in_channels=[3, 8, 8])
    elif CURRENT_MODEL == 'UNet':
        model = UNet(n_channels=3, n_classes=1)
    elif CURRENT_MODEL == 'TEED':
        model = TEED()
    elif CURRENT_MODEL == 'CATS':
        model = CATS()
    elif CURRENT_MODEL == 'BDCN':
        model = BDCN()
    elif CURRENT_MODEL == 'ConDSeg':
        model = ConDSeg()
    elif CURRENT_MODEL == 'FSGNet':
        model = FSGNet()
    elif CURRENT_MODEL == 'TransUNet':
        model = TransUNet()
    else:
        raise ValueError(f"未知模型: {CURRENT_MODEL}，请检查 CURRENT_MODEL 变量")
    return model


def train_and_validate(params, model, train_iter, val_iter, config, device, jd_desc, plot=False, trial=None,
                       verbose=False):
    # 【修改】端到端训练时使用分层学习率，扩展到所有级联模型
    # 当 freeze_lsseg=False 时，LSSeg 使用较小的学习率
    is_cascade_model = CURRENT_MODEL in ['LSSegSAMLoRA', 'LSSegSAMLoRA_Simple', 'LSSegMedSAM', 'LSSegMedSAM_Simple']
    if is_cascade_model and hasattr(model, 'lsseg') and hasattr(model, 'freeze_lsseg') and not model.freeze_lsseg:
        lsseg_lr_ratio = params.get("lsseg_lr_ratio", 0.1)
        print(f"Using layered learning rates: LSSeg lr = {params['lr'] * lsseg_lr_ratio:.6f}, SAM lr = {params['lr']:.6f}")
        optimizer = torch.optim.AdamW([
            {'params': model.lsseg.parameters(), 'lr': params["lr"] * lsseg_lr_ratio},
            {'params': model.sam.prompt_encoder.parameters(), 'lr': params["lr"]},
            {'params': model.sam.mask_decoder.parameters(), 'lr': params["lr"]},
        ], weight_decay=params["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params["step_size"], gamma=params["lr_factor"])
    loss_fn = config['loss_function']

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    best_dice = 0.0
    train_loss_lst, val_dice_lst = [], []

    model.to(device)

    # 【优化 1】初始化 AMP GradScaler
    is_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=is_cuda)

    if verbose:
        iterator = tqdm(range(config['num_epochs']), desc=jd_desc, dynamic_ncols=True)
    else:
        iterator = range(config['num_epochs'])

    is_sam_lora = isinstance(model,
                             (SAMLoRA, MedSAM, LSSegSAMLoRA, LSSegSAMLoRA_Simple, LSSegMedSAM, LSSegMedSAM_Simple))

    for epoch in iterator:
        # 训练
        model.train()
        train_l = []

        for images, masks in train_iter:
            masks = masks.float()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            # 【优化 1】开启混合精度上下文
            with torch.autocast(device_type=device.type, enabled=is_cuda):
                if is_sam_lora:
                    preds = model(images, masks)
                else:
                    images = images.float() / 255.0
                    preds = model(images)
                l = loss_fn(preds, masks)

            # 【优化 1】Scaler 反向传播
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            train_l.append(l.item())

        # 【修复】在 optimizer.step() 之后调用 scheduler.step()
        scheduler.step()

        # 验证
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_iter:
                images, masks = images.to(device), masks.long().to(device)

                # 【优化 1】验证推理也使用混合精度
                with torch.autocast(device_type=device.type, enabled=is_cuda):
                    if is_sam_lora:
                        preds = model(images).sigmoid()
                    else:
                        images = images.float() / 255.0
                        preds = model(images).sigmoid()

                dice_metric(y_pred=(preds > 0.5).long(), y=masks)
            val_dice = dice_metric.aggregate().item()
            dice_metric.reset()

        if val_dice > best_dice:
            best_dice = val_dice

        train_loss = np.mean(train_l)

        if verbose:
            iterator.set_postfix(loss=f"{train_loss:.4f}", dice=f"{val_dice:.4f}", best=f"{best_dice:.4f}")

        # Optuna 剪枝
        if trial:
            trial.report(val_dice, step=epoch)
            if trial.should_prune():
                if verbose: iterator.close()
                raise optuna.TrialPruned()

        train_loss_lst.append(train_loss)
        val_dice_lst.append(val_dice)

    # 显示训练曲线
    if plot:
        fig, axes = plot_curve(list(range(config['num_epochs'])), [train_loss_lst, val_dice_lst],
                               xlabel='Epochs', ylabel='Loss/Dice Value',
                               legend=['Train Loss', 'Val Dice'],
                               xlim=[0, config['num_epochs'] - 1], ylim=[0, 2])
        return best_dice, fig
    else:
        return best_dice


@contextmanager
def cuda_empty_cache():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def train_one_fold(params, train_idx, val_idx, X, y, config, device, fold_info, trial=None, verbose=False):
    with cuda_empty_cache():
        # 【优化 2】DataLoader 加速参数
        loader_kwargs = config.get('loader_kwargs', {})

        train_iter = DataLoader(
            ImageSegDataset(image_paths=X[train_idx], mask_paths=y[train_idx],
                            resize=config['image_resize'], is_train=True),
            batch_size=params['batch_size'], shuffle=True, **loader_kwargs
        )
        val_iter = DataLoader(
            ImageSegDataset(image_paths=X[val_idx], mask_paths=y[val_idx],
                            resize=config['image_resize'], is_train=False),
            batch_size=params['batch_size'], shuffle=False, **loader_kwargs
        )

        model = build_model(**params)
        score = train_and_validate(params, model, train_iter, val_iter, config, device,
                                   fold_info, trial=trial, verbose=verbose)
        del model, train_iter, val_iter
    return score


def nested_cross_validation(config, device):
    now = datetime.now()
    log_path = f'log/{config["exp_name"]} {now.strftime("%m-%d %H_%M")}'
    os.makedirs(log_path, exist_ok=True)

    img_paths, msk_paths = read_index_csv(config['index_csv'])
    outer_cv = KFold(n_splits=config['outer_cv_num'], shuffle=True, random_state=config['random_state'])
    outer_scores = []
    loader_kwargs = config.get('loader_kwargs', {})

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(img_paths, msk_paths)):
        print(f"\n{'=' * 20} Outer Fold {outer_fold} Start {'=' * 20}")
        X_outer_train, X_test = img_paths[outer_train_idx], img_paths[outer_test_idx]
        y_outer_train, y_test = msk_paths[outer_train_idx], msk_paths[outer_test_idx]

        study = optuna.create_study(
            storage=f"sqlite:///optuna.db",
            study_name=f'{os.path.basename(log_path)} outer{outer_fold}',
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config['random_state']),
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=config['inner_cv_num'],
                                                  reduction_factor=3)
        )

        def inner_objective(trial):
            params = suggest_params(trial)
            inner_cv = KFold(n_splits=config['inner_cv_num'], shuffle=True, random_state=config['random_state'])
            inner_scores = []

            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train)):
                score = train_one_fold(params, inner_train_idx, inner_val_idx,
                                       X_outer_train, y_outer_train, config, device,
                                       f'trial{trial.number}-inner{inner_fold}', verbose=False)
                inner_scores.append(score)
                trial.report(score, step=inner_fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(inner_scores)

        print(f"Running Optuna search for Outer Fold {outer_fold}...")
        study.optimize(inner_objective, n_trials=config['n_trials'], timeout=7200)

        best_params = study.best_params
        print(f"Best params found: {best_params}")

        with cuda_empty_cache():
            # 【优化 2】此处也应用 DataLoader 加速参数
            outer_train_iter = DataLoader(
                ImageSegDataset(X_outer_train, y_outer_train, resize=config['image_resize'], is_train=True),
                batch_size=best_params['batch_size'], shuffle=True, **loader_kwargs
            )
            outer_test_iter = DataLoader(
                ImageSegDataset(X_test, y_test, resize=config['image_resize'], is_train=False),
                batch_size=best_params['batch_size'], shuffle=False, **loader_kwargs
            )
            final_model = build_model(**best_params)

            final_score, fig = train_and_validate(best_params, final_model,
                                                  outer_train_iter, outer_test_iter, config, device,
                                                  f'Outer Fold {outer_fold} Final Train',
                                                  plot=True, verbose=True)
            outer_scores.append(final_score)
            metrics = evaluate(final_model, outer_test_iter, device=device, config=config)

            folder_save_path = f'{log_path}/fold_{outer_fold}'
            os.makedirs(folder_save_path, exist_ok=True)
            fig.savefig(f'{folder_save_path}/training_curve.jpg', dpi=300, bbox_inches='tight')
            pd.DataFrame([metrics]).to_csv(os.path.join(folder_save_path, 'metrics.csv'), index=False, encoding='utf-8')
            state_dict = {k: v for k, v in final_model.state_dict().items()
                          if 'total_ops' not in k and 'total_params' not in k}
            torch.save(state_dict, f'{folder_save_path}/model_weights_{outer_fold}.pth')

            del final_model, outer_train_iter, outer_test_iter

    aggregate_metrics(log_path, f'{log_path}/final_metrics.csv')
    return outer_scores


def cross_validation(config, device):
    now = datetime.now()
    log_path = f'log/{config["exp_name"]} {now.strftime("%m-%d %H_%M")}'
    os.makedirs(log_path, exist_ok=True)

    img_paths, msk_paths = read_index_csv(config['index_csv'])
    cv = KFold(n_splits=config['outer_cv_num'], shuffle=True, random_state=config['random_state'])
    fold_scores = []
    loader_kwargs = config.get('loader_kwargs', {})

    for fold, (train_idx, test_idx) in enumerate(cv.split(img_paths, msk_paths)):
        print(f"\n{'=' * 20} Fold {fold} Start {'=' * 20}")
        img_train, img_test = img_paths[train_idx], img_paths[test_idx]
        msk_train, msk_test = msk_paths[train_idx], msk_paths[test_idx]

        study = optuna.create_study(
            storage=f"sqlite:///optuna.db",
            study_name=f'{os.path.basename(log_path)} fold_{fold}',
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config['random_state']),
            pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=config['num_epochs'],
                                                  reduction_factor=3),
            load_if_exists=True
        )
        # 【修复】已经删除了这里覆盖 INFO 级别的日志设置代码

        print(f"  > Start Optuna Search...")

        def objective(trial):
            params = suggest_params(trial)
            sub_train_idx, val_idx = train_test_split(np.arange(len(img_train)),
                                                      test_size=0.25, random_state=config['random_state'],
                                                      shuffle=True)
            score = train_one_fold(params, sub_train_idx, val_idx,
                                   img_train, msk_train, config, device,
                                   f'fold {fold}, trial {trial.number}', trial, verbose=False)
            return score

        print(f"Running Optuna search for Fold {fold}...")
        study.optimize(objective, n_trials=config['n_trials'], timeout=7200)

        best_params = study.best_params
        print(f"Best params found: {best_params}")

        with cuda_empty_cache():
            # 【优化 2】此处也应用 DataLoader 加速参数
            outer_train_iter = DataLoader(
                ImageSegDataset(img_train, msk_train, resize=config['image_resize'], is_train=True),
                batch_size=best_params['batch_size'], shuffle=True, **loader_kwargs
            )
            outer_test_iter = DataLoader(
                ImageSegDataset(img_test, msk_test, resize=config['image_resize'], is_train=False),
                batch_size=best_params['batch_size'], shuffle=False, **loader_kwargs
            )
            final_model = build_model(**best_params)

            final_score, fig = train_and_validate(best_params, final_model,
                                                  outer_train_iter, outer_test_iter, config, device,
                                                  f'Fold {fold} Final Train', plot=True, verbose=True)
            fold_scores.append(final_score)
            metrics = evaluate(final_model, outer_test_iter, device=device, config=config)

            folder_save_path = f'{log_path}/fold_{fold}'
            os.makedirs(folder_save_path, exist_ok=True)
            fig.savefig(f'{folder_save_path}/training_curve.jpg', dpi=300, bbox_inches='tight')
            pd.DataFrame([metrics]).to_csv(os.path.join(folder_save_path, 'metrics.csv'), index=False, encoding='utf-8')
            state_dict = {k: v for k, v in final_model.state_dict().items()
                          if 'total_ops' not in k and 'total_params' not in k}
            torch.save(state_dict, f'{folder_save_path}/model_weights_{fold}.pth')

            del final_model, outer_train_iter, outer_test_iter

    aggregate_metrics(log_path, f'{log_path}/final_metrics.csv')
    return fold_scores


if __name__ == '__main__':
    # 保持 Optuna 日志整洁
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 【优化 3】开启 CuDNN Benchmark 加速 CNN 运算
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = {
        'exp_name': 'test_STARE_FSGNet',
        'outer_cv_num': 5,
        'inner_cv_num': 3,
        'random_state': 800,
        'index_csv': 'data/idx_STARE.csv',
        'image_resize': [512, 512],
        # num_epochs:sam类模型设置为70，原参数为100
        'num_epochs': 100,
        # n_trials:端到端训练参数更多,55
        'n_trials': 30,
        'loss_function': dice_ce_loss(),

        # 【优化 2】集中配置 DataLoader 参数
        # 提示：如果 Windows 报 DataLoader worker 错误，请把 num_workers 改为 0
        'loader_kwargs': {
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': False,
            'prefetch_factor': 2
        }
    }

    set_seed(config['random_state'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}.')

    # 嵌套交叉验证
    # outer_scores = nested_cross_validation(config, device)

    # 交叉验证
    outer_scores = cross_validation(config, device)

    print("\nCV results:")
    print(outer_scores)
    print(f"Mean Dice: {np.mean(outer_scores):.4f}")
    print(f"Std Dice:  {np.std(outer_scores):.4f}")