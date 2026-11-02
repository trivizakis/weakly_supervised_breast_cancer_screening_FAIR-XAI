import os
import gc
import sys
import json
import time
import shutil
import random
import argparse
import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np
import pandas as pd
import pydicom
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import timm

# Path resolution
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
for p in [curr_dir, parent_dir, os.path.join(curr_dir, "transfer_learning")]:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

from preprocessing_RSNA_MAMO import MammogramPreprocessor, read_and_window_dicom
from senAnals import run_threshold_sensitivity_analysis
from MammographyXAI import MammographyXAI
from ViTXAIEnsembler import ViTXAIEnsembler
from BIRADSShapeDescriptorExtractor import BIRADSShapeDescriptorExtractor
from export_contours_to_geojson import export_contours_to_geojson
from visualize_clf_xai import visualize_classification_xai


# =========================================================================
# GLOBAL HYPERPARAMETERS & PIPELINE CONFIGURATION
# =========================================================================

RANDOM_SEED = 86
TARGET_IMG_SIZE = (512, 512)   # High-resolution input matching ImageNet transfer learning backbones
AUGMENTATION_FACTOR = 40        # 40-fold random data augmentation for training set only
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_EVAL = 32
NUM_WORKERS = 8
PROCESS_POOL_WORKERS = 24
PIN_MEMORY = True

# Optimization Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5     # Early stopping with patience 5 on validation set
CLINICAL_BETA = 1.5

# XAI Hyperparameters
XAI_CORE_MULTIPLIER = 0.80
XAI_MARGIN_MULTIPLIER = 0.50
XAI_MIN_CONTOUR_AREA = 25
XAI_IG_STEPS = 20
XAI_PIXEL_SPACING_MM = 0.10
XAI_EVAL_MAX_SAMPLES = 50       # Representative XAI reports per model

# All 18 ImageNet-based architectures from deep_transfer_learning.py
TRANSFER_ARCH_MAP = {
    "resnet50":     {"timm_name": "resnet50"},
    "resnet101":    {"timm_name": "resnet101"},
    "resnet152":    {"timm_name": "resnet152"},
    "resnetv250":   {"timm_name": "resnet50.a1_in1k"},
    "resnetv2101":  {"timm_name": "resnet101.a1_in1k"},
    "resnetv2152":  {"timm_name": "resnet152.a1_in1k"},
    "densenet121":  {"timm_name": "densenet121"},
    "densenet169":  {"timm_name": "densenet169"},
    "densenet201":  {"timm_name": "densenet201"},
    "vgg16":        {"timm_name": "vgg16"},
    "vgg19":        {"timm_name": "vgg19"},
    "inc3":         {"timm_name": "inception_v3"},
    "incr2":        {"timm_name": "inception_resnet_v2"},
    "xception":     {"timm_name": "legacy_xception"},
    "mobile":       {"timm_name": "mobilenetv1_100"},
    "mobilev2":     {"timm_name": "mobilenetv2_100"},
    "nasnetm":      {"timm_name": "mnasnet_100"},
    "nasnetl":      {"timm_name": "nasnetalarge"}
}

ALL_TRANSFER_ARCHITECTURES = list(TRANSFER_ARCH_MAP.keys())


def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assign_age_group(age) -> str:
    if pd.isna(age):
        return "Unknown"
    age = float(age)
    if age < 20:
        return "before 20"
    elif age < 40:
        return "20-40"
    elif age < 55:
        return "40-55"
    else:
        return "55+"


def compute_metrics(y_true, y_probs, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_probs = np.nan_to_num(np.asarray(y_probs, dtype=float), nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (y_probs >= threshold).astype(int)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos > 0 and n_neg > 0:
        try:
            auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            auc = 0.5
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sp = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sn = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        acc = float((tp + tn) / len(y_true))
    elif n_pos > 0:
        auc = 0.5
        bacc = float(np.mean(y_pred == 1))
        tp = int(np.sum(y_pred == 1))
        fn = int(np.sum(y_pred == 0))
        tn, fp = 0, 0
        sp = 0.0
        sn = float(tp / n_pos)
        prec = 1.0 if tp > 0 else 0.0
        acc = sn
    else:
        auc = 0.5
        bacc = float(np.mean(y_pred == 0))
        tn = int(np.sum(y_pred == 0))
        fp = int(np.sum(y_pred == 1))
        tp, fn = 0, 0
        sp = float(tn / n_neg)
        sn = 0.0
        prec = 0.0
        acc = sp

    beta = CLINICAL_BETA
    f1 = 2 * (prec * sn) / (prec + sn + 1e-8)
    f_beta = (1 + beta**2) * (prec * sn) / ((beta**2 * prec) + sn + 1e-8)

    return {
        "balanced_ACC": round(bacc, 4),
        "AUC": round(auc, 4),
        "SP": round(sp, 4),
        "SN": round(sn, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "f_beta": round(f_beta, 4),
        "decision_threshold": round(threshold, 4),
        "total_samples": int(len(y_true)),
        "true_label_cancer_cases": n_pos,
        "true_label_benign_cases": n_neg,
        "cancer_cases": n_pos,
        "non_cancer_cases": n_neg,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


# =========================================================================
# OFFLINE 40-FOLD DATA AUGMENTATION WORKER & PIPELINE
# =========================================================================

def _patient_aug_worker(task):
    """
    Applies 40 distinct random data augmentations to an existing preprocessed mammogram array
    by loading the base image once and saving 40 float16 .npy files.
    """
    src_path, dest_base_dir, base_key, num_aug, aug_seed_base, target_size = task
    if not os.path.exists(src_path):
        return False

    # Fast-check if all files already exist
    all_exist = True
    for aug_i in range(num_aug):
        if not os.path.exists(os.path.join(dest_base_dir, f"{base_key}_aug{aug_i}.npy")):
            all_exist = False
            break
    if all_exist:
        return True

    try:
        if src_path.endswith(".dcm"):
            arr = read_and_window_dicom(src_path)
        else:
            arr = np.load(src_path)

        if arr.shape != target_size:
            arr = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

        for aug_i in range(num_aug):
            aug_dest = os.path.join(dest_base_dir, f"{base_key}_aug{aug_i}.npy")
            if os.path.exists(aug_dest):
                continue

            gen = torch.Generator().manual_seed(int(aug_seed_base + aug_i * 100))
            tensor = torch.from_numpy(arr).unsqueeze(0).to(torch.float32)

            # 1. Random Flips
            if torch.rand(1, generator=gen).item() > 0.5:
                tensor = torch.flip(tensor, dims=[-1])
            if torch.rand(1, generator=gen).item() > 0.5:
                tensor = torch.flip(tensor, dims=[-2])

            # 2. Random Rotation
            if torch.rand(1, generator=gen).item() > 0.25:
                angle = (torch.rand(1, generator=gen).item() - 0.5) * 20.0
                tensor = TF.rotate(tensor, angle)

            # 3. Random Affine Translation & Scale
            if torch.rand(1, generator=gen).item() > 0.25:
                scale = 0.94 + torch.rand(1, generator=gen).item() * 0.12
                shift_x = int((torch.rand(1, generator=gen).item() - 0.5) * 24)
                shift_y = int((torch.rand(1, generator=gen).item() - 0.5) * 24)
                tensor = TF.affine(tensor, angle=0, translate=[shift_x, shift_y], scale=scale, shear=0)

            # 4. Photometric Contrast & Brightness Jitter
            if torch.rand(1, generator=gen).item() > 0.25:
                alpha = 0.92 + torch.rand(1, generator=gen).item() * 0.16
                beta = (torch.rand(1, generator=gen).item() - 0.5) * 0.08
                tissue_mask = (tensor != 0.0)
                tensor[tissue_mask] = tensor[tissue_mask] * alpha + beta

            # 5. Gaussian Noise Injection
            if torch.rand(1, generator=gen).item() > 0.5:
                noise = torch.randn(tensor.shape, generator=gen) * 0.012
                tissue_mask = (tensor != 0.0)
                tensor[tissue_mask] = tensor[tissue_mask] + noise[tissue_mask]

            aug_arr = tensor.squeeze(0).numpy().astype(np.float16)
            tmp_dest = f"{aug_dest[:-4]}.tmp_{os.getpid()}_{time.time_ns()}.npy"
            np.save(tmp_dest, aug_arr)
            os.replace(tmp_dest, aug_dest)
        return True
    except Exception:
        return False


def build_offline_augmented_dataset(df_train, search_cache_dirs, aug_cache_dir, data_dir, aug_factor=AUGMENTATION_FACTOR, target_size=TARGET_IMG_SIZE):
    """
    Expands the training set 40-fold with offline augmented arrays saved on disk.
    Dispatches 1 task per base training exam to generate all 40 augmentations with 1 disk read.
    """
    os.makedirs(aug_cache_dir, exist_ok=True)
    total_augmented_samples = len(df_train) * aug_factor
    print(f"\n[*] Generating 40-Fold Offline Augmented Dataset ({len(df_train)} base -> {total_augmented_samples} augmented samples)...")

    aug_records = []
    patient_tasks = []

    for idx, row in df_train.reset_index(drop=True).iterrows():
        pid = str(row["patient_id"])
        imid = str(row["image_id"])
        base_key = f"{pid}_{imid}"

        src_path = None
        for d in search_cache_dirs:
            cand = os.path.join(d, f"{base_key}.npy")
            if os.path.exists(cand):
                src_path = cand
                break

        if src_path is None:
            dcm_cand = os.path.join(data_dir, "train_images", pid, f"{imid}.dcm")
            if os.path.exists(dcm_cand):
                src_path = dcm_cand

        for aug_idx in range(aug_factor):
            aug_fname = f"{base_key}_aug{aug_idx}.npy"
            aug_dest = os.path.join(aug_cache_dir, aug_fname)

            rec = row.to_dict()
            rec["aug_key"] = f"{base_key}_aug{aug_idx}"
            rec["aug_file_path"] = aug_dest
            aug_records.append(rec)

        if src_path is not None:
            seed = RANDOM_SEED * 100000 + idx * 1000
            patient_tasks.append((src_path, aug_cache_dir, base_key, aug_factor, seed, target_size))

    print(f"[*] Dispatching {len(patient_tasks)} patient augmentation jobs across {PROCESS_POOL_WORKERS} workers...")
    t0_aug = time.time()
    with ProcessPoolExecutor(max_workers=PROCESS_POOL_WORKERS) as ex:
        list(ex.map(_patient_aug_worker, patient_tasks))
    print(f"[✓] Offline augmentation completed in {time.time()-t0_aug:.1f}s for {total_augmented_samples} files.")

    df_aug = pd.DataFrame(aug_records)
    return df_aug


# =========================================================================
# DATASET IMPLEMENTATION
# =========================================================================

class MammoTransferDataset(Dataset):
    """
    PyTorch Dataset loading preprocessed or augmented numpy arrays and
    formatting 3-channel tensors [3, H, W] for ImageNet transfer learning.
    """
    def __init__(self, df: pd.DataFrame, search_cache_dirs: list, data_dir: str = None, target_size=TARGET_IMG_SIZE, is_train=False):
        self.df = df.reset_index(drop=True)
        self.search_cache_dirs = search_cache_dirs
        self.data_dir = data_dir
        self.target_size = target_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = float(row.get("cancer", 0))
        pid = str(row["patient_id"])
        imid = str(row["image_id"])

        img = None
        # Check if direct augmented file path is provided
        if "aug_file_path" in row and os.path.exists(str(row["aug_file_path"])):
            try:
                img = np.load(row["aug_file_path"]).astype(np.float32)
            except Exception:
                pass

        if img is None:
            base_key = f"{pid}_{imid}"
            for cdir in self.search_cache_dirs:
                cand = os.path.join(cdir, f"{base_key}.npy")
                if os.path.exists(cand):
                    try:
                        img = np.load(cand).astype(np.float32)
                        break
                    except Exception:
                        pass

        if img is None and self.data_dir is not None:
            dcm_cand = os.path.join(self.data_dir, "train_images", pid, f"{imid}.dcm")
            if os.path.exists(dcm_cand):
                try:
                    img = read_and_window_dicom(dcm_cand)
                except Exception:
                    pass

        if img is None:
            img = np.zeros(self.target_size, dtype=np.float32)

        if img.shape != self.target_size:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)

        # Standardize into 3-channel tensor [3, H, W] for ImageNet transfer models
        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).repeat(3, 1, 1).float()
        return img_tensor, torch.tensor(target, dtype=torch.float32), pid, imid


# =========================================================================
# MODEL ARCHITECTURE FACTORY
# =========================================================================

class ImageNetTransferClassifier(nn.Module):
    """
    Wraps any pretrained ImageNet architecture from deep_transfer_learning.py
    with a single binary classification logit output.
    Hooks the final spatial feature map for Grad-CAM / XAI dynamically and robustly.
    """
    def __init__(self, model_key: str, pretrained: bool = True):
        super().__init__()
        self.model_key = model_key
        cfg = TRANSFER_ARCH_MAP[model_key]
        self.timm_name = cfg["timm_name"]

        # Instantiate pretrained model from timm
        self.backbone = timm.create_model(self.timm_name, pretrained=pretrained, num_classes=1)

        # Gradient hooks for XAI
        self.activations = None
        self.gradients = None
        self._register_target_hooks()

    def _register_target_hooks(self):
        target_mod = None
        for mod in self.backbone.modules():
            if isinstance(mod, (nn.Conv2d, nn.BatchNorm2d)):
                target_mod = mod

        if target_mod is not None:
            def fwd_hook(m, inp, out):
                self.activations = out
                if out.requires_grad:
                    out.register_hook(lambda g: setattr(self, "gradients", g))

            target_mod.register_forward_hook(fwd_hook)

    def forward(self, x):
        return self.backbone(x)


# =========================================================================
# LEARNING CURVES PLOTTING
# =========================================================================

def plot_and_save_learning_curves(history: dict, save_png_path: str, save_csv_path: str, best_epoch: int, model_name: str):
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(save_csv_path, index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#1e1e1e")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#252526")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.grid(True, linestyle="--", alpha=0.3)

    epochs = history["epoch"]
    # Panel 1: Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#ff7f0e", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#1f77b4", lw=2, ls="--")
    ax1.axvline(best_epoch, color="#2ca02c", ls=":", label=f"Best Epoch ({best_epoch})")
    ax1.set_title(f"{model_name.upper()} - Training & Validation Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(facecolor="#252526", edgecolor="white", labelcolor="white")

    # Panel 2: Balanced Accuracy & AUC
    ax2.plot(epochs, history["val_bacc"], label="Val BACC", color="#2ca02c", lw=2.2)
    ax2.plot(epochs, history["val_auc"], label="Val AUC", color="#d62728", lw=2.0, ls="-.")
    ax2.plot(epochs, history["train_bacc"], label="Train BACC", color="#9467bd", lw=1.5, alpha=0.7)
    ax2.axvline(best_epoch, color="#2ca02c", ls=":", label=f"Best Epoch ({best_epoch})")
    ax2.set_title(f"{model_name.upper()} - Validation Balanced Accuracy & AUC", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric Value")
    ax2.legend(facecolor="#252526", edgecolor="white", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_png_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


# =========================================================================
# MAIN PIPELINE EXECUTION
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="ImageNet Transfer Learning Fine-Tuning Sweep")
    parser.add_argument("--models", nargs="+", default=ALL_TRANSFER_ARCHITECTURES, help="List of architectures to train")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Maximum epochs per architecture")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_TRAIN, help="Training batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--run-dir", type=str, default=None, help="Existing run directory to resume or append to")
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing Transfer Learning Fine-Tuning Sweep on device: {device}")
    print(f"[*] Architectures to execute ({len(args.models)}): {args.models}")

    base_dir = "/mnt/fourT/projects4TB/MAMO"
    main_ws_dir = os.path.join(base_dir, "main_ws")
    data_dir = os.path.join(base_dir, "rsna-breast-cancer-detection")
    cache_root_dir = os.path.join(base_dir, "preprocessed_cache")
    aug_cache_dir = os.path.join(main_ws_dir, "augmented_cache")

    # ---------------------------------------------------------
    # Setup Dedicated Output Directory or Resume
    # ---------------------------------------------------------
    if args.run_dir and os.path.exists(args.run_dir):
        run_dir = args.run_dir
    else:
        # Check if the latest run_ws_* folder has splits and summary to resume
        latest_candidate = None
        for item in sorted(os.listdir(main_ws_dir), reverse=True):
            if item.startswith("run_ws_"):
                cand_p = os.path.join(main_ws_dir, item)
                if os.path.exists(os.path.join(cand_p, "splits", "convergence_set.csv")):
                    latest_candidate = cand_p
                    break
        if latest_candidate is not None:
            run_dir = latest_candidate
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(main_ws_dir, f"run_ws_{timestamp}")

    run_results_dir = os.path.join(run_dir, "results")
    run_checkpoints_dir = os.path.join(run_dir, "checkpoints")
    run_splits_dir = os.path.join(run_dir, "splits")
    run_xai_dir = os.path.join(run_dir, "rsna_xai_outputs")

    for d in [run_dir, run_results_dir, run_checkpoints_dir, run_splits_dir, run_xai_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 85)
    print(f"[*] Dedicated Run Directory:  {run_dir}")
    print(f"[*] Checkpoints Directory:    {run_checkpoints_dir}")
    print(f"[*] Results Directory:        {run_results_dir}")
    print(f"[*] Splits Directory:         {run_splits_dir}")
    print(f"[*] XAI Outputs Directory:    {run_xai_dir}")
    print("=" * 85)

    # ---------------------------------------------------------
    # Step 1: Discover or Deterministically Construct Splits
    # ---------------------------------------------------------
    print("\n--- [Step 1: Dataset Cohort Partitioning] ---")
    calib_csv_path = os.path.join(run_splits_dir, "testing_set.csv")
    conv_csv_path = os.path.join(run_splits_dir, "convergence_set.csv")
    train_csv_path = os.path.join(run_splits_dir, "training_set.csv")
    val_csv_path = os.path.join(run_splits_dir, "validation_set.csv")

    if os.path.exists(train_csv_path) and os.path.exists(val_csv_path) and os.path.exists(calib_csv_path):
        print(f"[*] Loaded existing verified splits directly from run directory: {run_splits_dir}")
        df_train = pd.read_csv(train_csv_path).reset_index(drop=True)
        df_val = pd.read_csv(val_csv_path).reset_index(drop=True)
        df_test = pd.read_csv(calib_csv_path).reset_index(drop=True)
        df_convergence = pd.read_csv(conv_csv_path).reset_index(drop=True)
    else:
        df_raw = pd.read_csv(os.path.join(data_dir, "train.csv"))
        df_raw["age_group"] = df_raw["age"].apply(assign_age_group)
        df_raw["density_clean"] = df_raw["density"].fillna("Unknown")

        print("[*] Deterministically reconstructing identical splits from train.csv (Seed = 86)...")
        df_all_mal = df_raw[df_raw["cancer"] == 1].reset_index(drop=True)
        df_all_ben = df_raw[df_raw["cancer"] == 0].reset_index(drop=True)

        df_mal_calib, df_mal_test = train_test_split(df_all_mal, test_size=0.80, random_state=RANDOM_SEED)
        df_mal_calib = df_mal_calib.reset_index(drop=True)
        df_mal_test = df_mal_test.reset_index(drop=True)

        df_ben_shuffled = df_all_ben.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        df_ben_calib = df_ben_shuffled.iloc[:len(df_mal_calib)].reset_index(drop=True)
        df_ben_test = df_ben_shuffled.iloc[len(df_mal_calib):len(df_mal_calib) + len(df_mal_test)].reset_index(drop=True)

        df_test = pd.concat([df_mal_calib, df_ben_calib]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        df_convergence = pd.concat([df_mal_test, df_ben_test]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

        df_test["split_role"] = "testing_set"
        df_convergence["split_role"] = "convergence_set"

        df_train, df_val = train_test_split(
            df_convergence,
            test_size=0.10,
            stratify=df_convergence["cancer"],
            random_state=RANDOM_SEED
        )
        df_train = df_train.reset_index(drop=True)
        df_train["split_role"] = "training_set"
        df_val = df_val.reset_index(drop=True)
        df_val["split_role"] = "validation_set"

        # Save split CSVs
        df_convergence.to_csv(os.path.join(run_splits_dir, "convergence_set.csv"), index=False)
        df_train.to_csv(os.path.join(run_splits_dir, "training_set.csv"), index=False)
        df_val.to_csv(os.path.join(run_splits_dir, "validation_set.csv"), index=False)
        df_test.to_csv(os.path.join(run_splits_dir, "testing_set.csv"), index=False)

        df_convergence.to_csv(os.path.join(run_dir, "convergence_set.csv"), index=False)
        df_train.to_csv(os.path.join(run_dir, "training_set.csv"), index=False)
        df_val.to_csv(os.path.join(run_dir, "validation_set.csv"), index=False)
        df_test.to_csv(os.path.join(run_dir, "testing_set.csv"), index=False)

        summary_df = pd.DataFrame([
            {"set_name": "convergence_set", "total_samples": len(df_convergence), "malignant_count": int(df_convergence['cancer'].sum()), "benign_count": int(len(df_convergence)-df_convergence['cancer'].sum()), "malignant_pct": 50.0},
            {"set_name": "training_set", "total_samples": len(df_train), "malignant_count": int(df_train['cancer'].sum()), "benign_count": int(len(df_train)-df_train['cancer'].sum()), "malignant_pct": 50.0},
            {"set_name": "validation_set", "total_samples": len(df_val), "malignant_count": int(df_val['cancer'].sum()), "benign_count": int(len(df_val)-df_val['cancer'].sum()), "malignant_pct": 50.0},
            {"set_name": "testing_set", "total_samples": len(df_test), "malignant_count": int(df_test['cancer'].sum()), "benign_count": int(len(df_test)-df_test['cancer'].sum()), "malignant_pct": 50.0},
        ])
        summary_df.to_csv(os.path.join(run_dir, "dataset_splits_summary.csv"), index=False)
        summary_df.to_csv(os.path.join(run_splits_dir, "dataset_splits_summary.csv"), index=False)
        print(f"[✓] Split CSV files successfully preserved.")

    print("\n--- Cohort Summary ---")
    print(f"  • Convergence Set (Old Test Set):        {len(df_convergence)} exams ({df_convergence['cancer'].sum()} Malignant / {len(df_convergence)-df_convergence['cancer'].sum()} Benign)")
    print(f"    -> Training Set (90% Convergence):     {len(df_train)} exams ({df_train['cancer'].sum()} Malignant / {len(df_train)-df_train['cancer'].sum()} Benign)")
    print(f"    -> Validation Set (10% Convergence):   {len(df_val)} exams ({df_val['cancer'].sum()} Malignant / {len(df_val)-df_val['cancer'].sum()} Benign)")
    print(f"  • Testing Set (Old Calibration Set):     {len(df_test)} exams ({df_test['cancer'].sum()} Malignant / {len(df_test)-df_test['cancer'].sum()} Benign)")

    # ---------------------------------------------------------
    # Step 2: 40-Fold Offline Data Augmentation on Training Set Only
    # ---------------------------------------------------------
    search_cache_dirs = [
        os.path.join(cache_root_dir, "train"),
        os.path.join(cache_root_dir, "val"),
        os.path.join(cache_root_dir, "test"),
        aug_cache_dir
    ]

    aug_train_csv = os.path.join(run_splits_dir, "training_set_augmented_40fold.csv")
    if os.path.exists(aug_train_csv):
        df_aug_train = pd.read_csv(aug_train_csv)
        print(f"[✓] Loaded existing 40-fold augmented dataframe: {len(df_aug_train)} records.")
    else:
        df_aug_train = build_offline_augmented_dataset(
            df_train=df_train,
            search_cache_dirs=search_cache_dirs,
            aug_cache_dir=aug_cache_dir,
            data_dir=data_dir,
            aug_factor=AUGMENTATION_FACTOR,
            target_size=TARGET_IMG_SIZE
        )
        df_aug_train.to_csv(aug_train_csv, index=False)
        print(f"[✓] 40-Fold augmented training dataframe constructed: {len(df_aug_train)} records.")

    # Prepare DataLoaders
    # Training set: 40-fold augmented
    train_dataset = MammoTransferDataset(df_aug_train, search_cache_dirs, data_dir=data_dir, target_size=TARGET_IMG_SIZE, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Validation set: deterministic (NO augmentation)
    val_dataset = MammoTransferDataset(df_val, search_cache_dirs, data_dir=data_dir, target_size=TARGET_IMG_SIZE, is_train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Testing set: deterministic (NO augmentation)
    test_dataset = MammoTransferDataset(df_test, search_cache_dirs, data_dir=data_dir, target_size=TARGET_IMG_SIZE, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # ---------------------------------------------------------
    # Step 3: Sequential Fine-Tuning Sweep Across All Architectures
    # ---------------------------------------------------------
    benchmark_records = []
    completed_models = set()
    bench_json_path = os.path.join(run_results_dir, "all_models_benchmark_summary.json")
    if os.path.exists(bench_json_path):
        try:
            with open(bench_json_path, "r") as f:
                benchmark_records = json.load(f)
            completed_models = {r["architecture"] for r in benchmark_records}
            print(f"[*] Resuming from existing benchmark records ({len(completed_models)} completed): {list(completed_models)}")
        except Exception as e:
            print(f"[!] Warning reading benchmark summary: {e}")

    for arch_idx, arch_key in enumerate(args.models, 1):
        if arch_key in completed_models:
            print("\n" + "=" * 90)
            print(f"[✓] [{arch_idx}/{len(args.models)}] ARCHITECTURE: {arch_key.upper()} ALREADY COMPLETED - SKIPPING!")
            print("=" * 90)
            continue

        print("\n" + "=" * 90)
        print(f"[*] [{arch_idx}/{len(args.models)}] FINE-TUNING ARCHITECTURE: {arch_key.upper()} ({TRANSFER_ARCH_MAP[arch_key]['timm_name']})")
        print("=" * 90)

        torch.cuda.empty_cache()
        gc.collect()

        # Adaptive batch size & gradient accumulation for massive architectures (e.g. NASNetLarge)
        arch_batch_size = args.batch_size
        accum_steps = 1
        if arch_key == "nasnetl":
            arch_batch_size = 8
            accum_steps = 4
            eval_batch_size = 8
        else:
            eval_batch_size = BATCH_SIZE_EVAL

        current_train_loader = DataLoader(
            train_dataset,
            batch_size=arch_batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        current_val_loader = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        current_test_loader = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

        model = ImageNetTransferClassifier(arch_key, pretrained=True).to(device)

        model_lr = args.lr
        if arch_key in ["vgg16", "vgg19"]:
            model_lr = min(args.lr, 5e-5)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_lr, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.amp.GradScaler("cuda")

        patience = args.patience
        patience_counter = 0
        best_val_bacc = -1.0
        best_epoch = 1

        arch_ckpt_path = os.path.join(run_checkpoints_dir, f"best_{arch_key}.pth")

        history = {
            "epoch": [], "train_loss": [], "train_bacc": [], "train_auc": [],
            "train_sn": [], "train_sp": [], "val_loss": [], "val_bacc": [],
            "val_auc": [], "val_sn": [], "val_sp": [], "lr": []
        }

        print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train BACC':^12} | {'Val Loss':^10} | {'Val BACC':^10} | {'Val SN':^10} | {'Val SP':^10} | {'Val AUC':^10} | {'Status':^22}")
        print("-" * 115)

        for ep in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            train_preds, train_targets = [], []
            optimizer.zero_grad()

            for step, (imgs, targets, _, _) in enumerate(current_train_loader):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    logits = model(imgs).squeeze(-1)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=15.0, neginf=-15.0)
                    loss = criterion(logits, targets)
                    loss_scaled = loss / accum_steps

                scaler.scale(loss_scaled).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(current_train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * imgs.size(0)
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    probs = torch.nan_to_num(probs, nan=0.5)
                    train_preds.extend(probs.cpu().numpy().tolist())
                    train_targets.extend(targets.cpu().numpy().tolist())

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            ep_train_loss = running_loss / len(train_dataset)
            train_m = compute_metrics(train_targets, train_preds, threshold=0.5)

            # Validation evaluation
            model.eval()
            val_running_loss = 0.0
            val_preds, val_targets = [], []

            with torch.no_grad():
                for v_imgs, v_targets, _, _ in current_val_loader:
                    v_imgs = v_imgs.to(device, non_blocking=True)
                    v_targets = v_targets.to(device, non_blocking=True)

                    with torch.amp.autocast("cuda"):
                        v_logits = model(v_imgs).squeeze(-1)
                        v_logits = torch.nan_to_num(v_logits, nan=0.0, posinf=15.0, neginf=-15.0)
                        v_loss = criterion(v_logits, v_targets)
                        v_probs = torch.sigmoid(v_logits)
                        v_probs = torch.nan_to_num(v_probs, nan=0.5)

                    val_running_loss += v_loss.item() * v_imgs.size(0)
                    val_preds.extend(v_probs.cpu().numpy().tolist())
                    val_targets.extend(v_targets.cpu().numpy().tolist())

            ep_val_loss = val_running_loss / len(val_dataset)
            val_m = compute_metrics(val_targets, val_preds, threshold=0.5)

            history["epoch"].append(ep)
            history["train_loss"].append(round(ep_train_loss, 4))
            history["train_bacc"].append(train_m["balanced_ACC"])
            history["train_auc"].append(train_m["AUC"])
            history["train_sn"].append(train_m["SN"])
            history["train_sp"].append(train_m["SP"])
            history["val_loss"].append(round(ep_val_loss, 4))
            history["val_bacc"].append(val_m["balanced_ACC"])
            history["val_auc"].append(val_m["AUC"])
            history["val_sn"].append(val_m["SN"])
            history["val_sp"].append(val_m["SP"])
            history["lr"].append(current_lr)

            val_bacc = val_m["balanced_ACC"]
            if val_bacc > best_val_bacc + 1e-4:
                best_val_bacc = val_bacc
                best_epoch = ep
                patience_counter = 0
                torch.save(model.state_dict(), arch_ckpt_path)
                status_msg = f"[★] Best BACC ({val_bacc:.4f})"
            else:
                patience_counter += 1
                status_msg = f"Patience {patience_counter}/{patience}"

            print(f"{ep:^8} | {ep_train_loss:^12.4f} | {train_m['balanced_ACC']:^12.4f} | {ep_val_loss:^10.4f} | {val_bacc:^10.4f} | {val_m['SN']:^10.4f} | {val_m['SP']:^10.4f} | {val_m['AUC']:^10.4f} | {status_msg:^22}")

            # Early stopping check
            if patience_counter >= patience:
                print("-" * 115)
                print(f"[!] Early stopping triggered for {arch_key.upper()} at epoch {ep}! No validation BACC gain for {patience} epochs.")
                break

        # Save learning curve plots & CSV
        lc_png = os.path.join(run_results_dir, f"learning_curves_{arch_key}.png")
        lc_csv = os.path.join(run_results_dir, f"learning_curves_{arch_key}.csv")
        plot_and_save_learning_curves(history, lc_png, lc_csv, best_epoch=best_epoch, model_name=arch_key)

        # Restore best model checkpoint
        if os.path.exists(arch_ckpt_path):
            model.load_state_dict(torch.load(arch_ckpt_path, map_location=device))
        model.eval()

        # ---------------------------------------------------------
        # Threshold Sensitivity Analysis Calibration on Validation Set
        # ---------------------------------------------------------
        val_preds_calib, val_targets_calib = [], []
        with torch.no_grad():
            for v_imgs, v_targets, _, _ in current_val_loader:
                v_imgs = v_imgs.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    probs = torch.sigmoid(model(v_imgs).squeeze(-1))
                    probs = torch.nan_to_num(probs, nan=0.5)
                val_preds_calib.extend(probs.cpu().numpy().tolist())
                val_targets_calib.extend(v_targets.numpy().tolist())

        val_preds_calib = np.nan_to_num(np.array(val_preds_calib), nan=0.5)
        val_targets_calib = np.array(val_targets_calib)

        sens_curve_path = os.path.join(run_results_dir, f"sensitivity_curve_{arch_key}.png")
        sens_res = run_threshold_sensitivity_analysis(
            y_true=val_targets_calib,
            y_probs=val_preds_calib,
            beta=CLINICAL_BETA,
            save_plot_path=sens_curve_path
        )
        opt_thresh = float(sens_res["calibrated_optimal_threshold"])
        print(f"[*] Calibrated Optimal Threshold ({arch_key.upper()}): {opt_thresh:.4f}")

        # ---------------------------------------------------------
        # Evaluation on Held-Out Testing Set (N = 462, Old Calibration Set)
        # ---------------------------------------------------------
        test_scores, test_targets, test_pids, test_imids = [], [], [], []
        with torch.no_grad():
            for t_imgs, targets, pids, imids in current_test_loader:
                t_imgs = t_imgs.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    probs = torch.sigmoid(model(t_imgs).squeeze(-1))
                    probs = torch.nan_to_num(probs, nan=0.5)
                test_scores.extend(probs.cpu().numpy().tolist())
                test_targets.extend(targets.numpy().tolist())
                test_pids.extend(pids)
                test_imids.extend(imids)

        test_scores = np.nan_to_num(np.array(test_scores), nan=0.5)
        test_targets = np.array(test_targets)

        test_m = compute_metrics(test_targets, test_scores, threshold=opt_thresh)

        print("\n" + "=" * 80)
        print(f"   HELD-OUT TESTING METRICS FOR {arch_key.upper()} (N = {len(df_test)}, Threshold = {opt_thresh:.4f})")
        print("=" * 80)
        print(f"  Balanced Accuracy:     {test_m['balanced_ACC']:.4f}")
        print(f"  Area Under Curve (AUC):{test_m['AUC']:.4f}")
        print(f"  Sensitivity (Recall):  {test_m['SN']:.4f} ({test_m['TP']} / {test_m['true_label_cancer_cases']})")
        print(f"  Specificity:           {test_m['SP']:.4f} ({test_m['TN']} / {test_m['true_label_benign_cases']})")
        print(f"  Precision:             {test_m['precision']:.4f}")
        print(f"  Clinical F-1.5:        {test_m['f_beta']:.4f}")
        print(f"  Confusion Matrix:      TP={test_m['TP']}, TN={test_m['TN']}, FP={test_m['FP']}, FN={test_m['FN']}")
        print("=" * 80)

        # Save test predictions
        df_test_res = df_test.copy()
        df_test_res["pred_prob"] = test_scores
        df_test_res["pred_label"] = (test_scores >= opt_thresh).astype(int)
        df_test_res["pred_decision"] = np.where(df_test_res["pred_label"] == 1, "MALIGNANT", "BENIGN")
        df_test_res.to_csv(os.path.join(run_results_dir, f"test_predictions_{arch_key}.csv"), index=False)

        # Record benchmark
        rec = {
            "architecture": arch_key,
            "timm_name": TRANSFER_ARCH_MAP[arch_key]["timm_name"],
            "best_epoch": best_epoch,
            "calibrated_threshold": opt_thresh,
            **test_m
        }
        benchmark_records.append(rec)
        completed_models.add(arch_key)

        # Update master summary tables after each architecture
        df_bench = pd.DataFrame(benchmark_records)
        df_bench.to_csv(os.path.join(run_results_dir, "all_models_benchmark_summary.csv"), index=False)
        df_bench.to_csv(os.path.join(run_dir, "all_models_benchmark_summary.csv"), index=False)
        with open(os.path.join(run_results_dir, "all_models_benchmark_summary.json"), "w") as f:
            json.dump(benchmark_records, f, indent=2)

        # ---------------------------------------------------------
        # XAI Suite for Representative Samples (Primary Architecture or Top Performers)
        # ---------------------------------------------------------
        if arch_idx == 1 or arch_key in ["resnet50", "densenet121", "vgg16", "mobilev2"]:
            print(f"\n[*] Rendering 6-Panel Traditional XAI Suite for {arch_key.upper()}...")
            arch_xai_dir = os.path.join(run_xai_dir, arch_key)
            mal_xai_dir = os.path.join(arch_xai_dir, "malignant")
            ben_xai_dir = os.path.join(arch_xai_dir, "benign")
            os.makedirs(mal_xai_dir, exist_ok=True)
            os.makedirs(ben_xai_dir, exist_ok=True)

            xai_engine = MammographyXAI(model=model, device=device)
            ensembler = ViTXAIEnsembler(xai_engine=None)
            shape_extractor = BIRADSShapeDescriptorExtractor(pixel_spacing_mm=XAI_PIXEL_SPACING_MM)

            pos_samples = df_test_res[df_test_res["cancer"] == 1].head(XAI_EVAL_MAX_SAMPLES // 2)
            neg_samples = df_test_res[df_test_res["cancer"] == 0].sample(n=len(pos_samples), random_state=RANDOM_SEED)
            xai_cohort = pd.concat([pos_samples, neg_samples]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

            xai_recs = []
            for _, r in xai_cohort.iterrows():
                pid = str(r["patient_id"])
                imid = str(r["image_id"])
                key = f"{pid}_{imid}"
                c_true = int(r["cancer"])

                base_img = None
                for cdir in search_cache_dirs:
                    cand = os.path.join(cdir, f"{key}.npy")
                    if os.path.exists(cand):
                        try:
                            base_img = np.load(cand).astype(np.float32)
                            break
                        except Exception:
                            pass

                if base_img is None and data_dir is not None:
                    dcm_cand = os.path.join(data_dir, "train_images", pid, f"{imid}.dcm")
                    if os.path.exists(dcm_cand):
                        try:
                            base_img = read_and_window_dicom(dcm_cand)
                        except Exception:
                            pass

                if base_img is None:
                    base_img = np.zeros(TARGET_IMG_SIZE, dtype=np.float32)
                if base_img.shape != TARGET_IMG_SIZE:
                    base_img = cv2.resize(base_img, (TARGET_IMG_SIZE[1], TARGET_IMG_SIZE[0]))

                inp_tensor = torch.from_numpy(np.ascontiguousarray(base_img)).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

                gcam = xai_engine.grad_cam(inp_tensor)
                gcam_pp = xai_engine.grad_cam_plus_plus(inp_tensor)
                ig = xai_engine.integrated_gradients(inp_tensor, steps=XAI_IG_STEPS)

                ens_map = ensembler.compute_ensembled_map(gcam_pp, ig, alpha=0.5, fusion_mode="harmonic")
                contours = ensembler.extract_dual_confidence_contours(
                    base_img=base_img,
                    saliency_map=ens_map,
                    core_multiplier=XAI_CORE_MULTIPLIER,
                    margin_multiplier=XAI_MARGIN_MULTIPLIER,
                    min_contour_area=XAI_MIN_CONTOUR_AREA
                )

                overlay_rgb = cv2.cvtColor(contours["overlay_image"], cv2.COLOR_BGR2RGB)
                subfolder = "malignant" if c_true == 1 else "benign"
                save_f = os.path.join(arch_xai_dir, subfolder, f"{pid}_{imid}_clf_xai_report.png")
                save_geo = os.path.join(arch_xai_dir, subfolder, f"{pid}_{imid}_contours.geojson")

                export_contours_to_geojson(
                    contour_results=contours,
                    output_filepath=save_geo,
                    metadata={"patient_id": pid, "image_id": imid, "architecture": arch_key, "true_cancer": c_true}
                )

                visualize_classification_xai(
                    base_img=base_img,
                    grad_cam=gcam,
                    grad_cam_pp=gcam_pp,
                    integrated_grads=ig,
                    ensembled_map=ens_map,
                    contour_overlay=overlay_rgb,
                    pred_prob=float(r["pred_prob"]),
                    pred_decision=str(r["pred_decision"]),
                    true_label=c_true,
                    decision_threshold=opt_thresh,
                    save_path=save_f,
                    dpi=120
                )

                xai_recs.append({
                    "architecture": arch_key,
                    "patient_id": pid,
                    "image_id": imid,
                    "cancer_true": c_true,
                    "true_label_folder": subfolder,
                    "pred_prob": round(float(r["pred_prob"]), 4),
                    "pred_decision": r["pred_decision"],
                    "report_path": save_f,
                    "geojson_path": save_geo
                })

            pd.DataFrame(xai_recs).to_csv(os.path.join(arch_xai_dir, f"xai_summary_{arch_key}.csv"), index=False)
            print(f"[✓] Rendered {len(xai_recs)} XAI reports for {arch_key.upper()} in {arch_xai_dir}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "=" * 90)
    print("[✓] ALL 18 TRANSFER LEARNING ARCHITECTURES COMPLETED!")
    print(f"[*] Benchmark summary saved to: {os.path.join(run_results_dir, 'all_models_benchmark_summary.csv')}")
    print("=" * 90)


if __name__ == "__main__":
    main()
