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
from models.lsseg_medsam import LSSegMedSAM, LSSegMedSAM_Simple
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
# ============================================================
CURRENT_MODEL = 'LSSegSAMLoRA'

# ============================================================
# LSSegSAMLoRA 专用配置：手动指定 LSSeg 权重路径
# ============================================================
LSSEG_CHECKPOINT_PATH = "log/test_AxonDeepSeg_SEM_LSSeg 03-02 15_55/fold_3/model_weights_3.pth"


def get_fixed_params():
    """
    获取固定的参数（不参与搜索）
    这些参数不会出现在 optuna.best_params 中，需要手动补充
    """
    if CURRENT_MODEL in ['SAMLoRA', 'LSSegSAMLoRA', 'LSSegSAMLoRA_Simple', 'LSSegMedSAM', 'LSSegMedSAM_Simple']:
        return {
            "batch_size": 4,
            "weight_decay": 2e-4,
            "lr_factor": 0.7,
            "step_size": 15,
            "lora_alpha_ratio": 2,
            "freeze_lsseg": False,
            "box_bias": 0.0,
            "box_expand_ratio": 0.02,
        }
    elif CURRENT_MODEL == 'MedSAM':
        return {}
    else:
        return {}


def suggest_params(trial):
    """
    【优化】基于 optuna 历史最佳参数，固定次要参数，减少搜索空间
    """
    if CURRENT_MODEL in ['SAMLoRA', 'LSSegSAMLoRA', 'LSSegSAMLoRA_Simple', 'LSSegMedSAM', 'LSSegMedSAM_Simple']:
        # 获取固定参数
        fixed = get_fixed_params()

        # 搜索核心参数
        lora_r = trial.suggest_categorical("lora_r", [4, 8])

        params = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "lora_r": lora_r,
            "lora_alpha": lora_r * fixed["lora_alpha_ratio"],
            **fixed  # 合并固定参数
        }

        # LSSegSAMLoRA / LSSegMedSAM 专用参数
        if CURRENT_MODEL in ['LSSegSAMLoRA', 'LSSegSAMLoRA_Simple', 'LSSegMedSAM', 'LSSegMedSAM_Simple']:
            params["prompt_bias"] = trial.suggest_float("prompt_bias", 0.0, 2.0, step=0.5)
            params["lsseg_lr_ratio"] = trial.suggest_categorical("lsseg_lr_ratio", [0.1, 0.2])

        return params
    elif CURRENT_MODEL == 'MedSAM':
        lora_r = trial.suggest_categorical("lora_r", [0, 2, 4, 8])
        lora_alpha_ratio = trial.suggest_categorical("lora_alpha_ratio", [1, 2])
        return {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [2, 4]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "lr_factor": trial.suggest_float("lr_factor", 0.5, 0.9, step=0.1),
            "step_size": trial.suggest_int("step_size", 10, 20, step=5),
            "lora_r": lora_r,
            "lora_alpha": lora_r * lora_alpha_ratio if lora_r > 0 else 4,
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

        model = LSSegSAMLoRA(
            lsseg_checkpoint=lsseg_checkpoint,
            sam_checkpoint="sam_vit_b_01ec64.pth",
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 8),
            freeze_lsseg=params.get("freeze_lsseg", False),
            use_box_prompt=True,
            prompt_bias=params.get("prompt_bias", 0.0),
        )
    elif CURRENT_MODEL == 'LSSegSAMLoRA_Simple':
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
            lora_alpha=params.get("lora_alpha", 8),
            freeze_lsseg=params.get("freeze_lsseg", False),
            prompt_bias=params.get("prompt_bias", 0.0),
        )
    elif CURRENT_MODEL == 'LSSegMedSAM':
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
            sam_checkpoint="medsam_vit_b.pth",
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 8),
            freeze_lsseg=params.get("freeze_lsseg", False),
            use_box_prompt=True,
            prompt_bias=params.get("prompt_bias", 0.0),
            box_bias=params.get("box_bias", 0.0),
            box_expand_ratio=params.get("box_expand_ratio", 0.02),
        )
    elif CURRENT_MODEL == 'LSSegMedSAM_Simple':
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
            lora_alpha=params.get("lora_alpha", 8),
            freeze_lsseg=params.get("freeze_lsseg", False),
            prompt_bias=params.get("prompt_bias", 0.0),
        )
    elif CURRENT_MODEL == 'SAMLoRA':
        model = SAMLoRA(
            checkpoint_path="sam_vit_b_01ec64.pth",
            target_size=512,
            lora_r=params.get("lora_r", 4),
            lora_alpha=params.get("lora_alpha", 8)
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
    is_cascade_model = CURRENT_MODEL in ['LSSegSAMLoRA', 'LSSegSAMLoRA_Simple', 'LSSegMedSAM', 'LSSegMedSAM_Simple']
    if is_cascade_model and hasattr(model, 'lsseg') and hasattr(model, 'freeze_lsseg') and not model.freeze_lsseg:
        lsseg_lr_ratio = params.get("lsseg_lr_ratio", 0.1)
        print(f"Using layered learning rates: LSSeg lr = {params['lr'] * lsseg_lr_ratio:.6f}, SAM lr = {params['lr']:.6f}")
        optimizer = torch.optim.AdamW([
            {'params': model.lsseg.parameters(), 'lr': params["lr"] * lsseg_lr_ratio},
            {'params': model.sam.prompt_encoder.parameters(), 'lr': params["lr"]},
            {'params': model.sam.mask_decoder.parameters(), 'lr': params["lr"]},
        ], weight_decay=params.get("weight_decay", 2e-4))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params.get("weight_decay", 2e-4))
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.get("step_size", 15), gamma=params.get("lr_factor", 0.7))
    loss_fn = config['loss_function']

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    best_dice = 0.0
    train_loss_lst, val_dice_lst = [], []

    model.to(device)

    is_cuda = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=is_cuda)

    if verbose:
        iterator = tqdm(range(config['num_epochs']), desc=jd_desc, dynamic_ncols=True)
    else:
        iterator = range(config['num_epochs'])

    is_sam_lora = isinstance(model,
                             (SAMLoRA, MedSAM, LSSegSAMLoRA, LSSegSAMLoRA_Simple, LSSegMedSAM, LSSegMedSAM_Simple))

    for epoch in iterator:
        model.train()
        train_l = []

        for images, masks in train_iter:
            masks = masks.float()
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=is_cuda):
                if is_sam_lora:
                    preds = model(images, masks)
                else:
                    images = images.float() / 255.0
                    preds = model(images)
                l = loss_fn(preds, masks)

            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            train_l.append(l.item())

        scheduler.step()

        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_iter:
                images, masks = images.to(device), masks.long().to(device)

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

        if trial:
            trial.report(val_dice, step=epoch)
            if trial.should_prune():
                if verbose: iterator.close()
                raise optuna.TrialPruned()

        train_loss_lst.append(train_loss)
        val_dice_lst.append(val_dice)

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

        # 【修复】合并搜索参数和固定参数
        best_params = study.best_params.copy()
        fixed_params = get_fixed_params()
        best_params.update(fixed_params)
        
        # 计算依赖参数
        if 'lora_r' in best_params:
            best_params['lora_alpha'] = best_params['lora_r'] * fixed_params.get('lora_alpha_ratio', 2)
        
        print(f"Best params found: {best_params}")

        with cuda_empty_cache():
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    config = {
        'exp_name': 'test_AxonDeepSeg_SEM_LSSegSAMLoRA_reduced_params',
        'outer_cv_num': 5,
        'inner_cv_num': 3,
        'random_state': 800,
        'index_csv': 'data/idx_AxonDeepSeg_SEM.csv',
        'image_resize': [512, 512],
        'num_epochs': 70,
        'n_trials': 55,
        'loss_function': dice_ce_loss(),

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

    outer_scores = cross_validation(config, device)

    print("\nCV results:")
    print(outer_scores)
    print(f"Mean Dice: {np.mean(outer_scores):.4f}")
    print(f"Std Dice:  {np.std(outer_scores):.4f}")