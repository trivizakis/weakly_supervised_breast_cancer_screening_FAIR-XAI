import os
import gc
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

# Path resolution
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
for p in [curr_dir, parent_dir, os.path.join(curr_dir, "transfer_learning")]:
    if p not in sys.path and os.path.exists(p):
        sys.path.append(p)

from main_ws import (
    ImageNetTransferClassifier,
    TARGET_IMG_SIZE,
    XAI_CORE_MULTIPLIER,
    XAI_MARGIN_MULTIPLIER,
    XAI_MIN_CONTOUR_AREA,
    XAI_IG_STEPS,
    XAI_PIXEL_SPACING_MM,
    RANDOM_SEED
)
from preprocessing_RSNA_MAMO import read_and_window_dicom
from MammographyXAI import MammographyXAI
from ViTXAIEnsembler import ViTXAIEnsembler
from BIRADSShapeDescriptorExtractor import BIRADSShapeDescriptorExtractor
from export_contours_to_geojson import export_contours_to_geojson
from visualize_clf_xai import visualize_classification_xai


def compute_metrics(y_true, y_probs, threshold=0.5, beta=1.5):
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
        "cancer_cases": n_pos,
        "non_cancer_cases": n_neg,
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
    }


def assign_age_group(age) -> str:
    if pd.isna(age):
        return "Unknown"
    age = float(age)
    if age < 20:
        return "<20"
    elif age < 40:
        return "20-40"
    elif age < 50:
        return "40-50"
    else:
        return "50+"


def plot_subgroup_analysis(df_subgroups: pd.DataFrame, save_path: str):
    """
    Renders an publication-quality 4-panel comparative visualization of subgroup fairness
    across Age Groups, Sites, and Density Groups.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor("#121212")

    palette = {
        "Age Group": "#00bcd4",
        "Site ID": "#ff9800",
        "Breast Density": "#9c27b0"
    }

    metrics = [
        ("AUC", "Area Under ROC Curve (AUC)", axes[0, 0]),
        ("balanced_ACC", "Balanced Accuracy (BACC)", axes[0, 1]),
        ("SN", "Sensitivity (Recall)", axes[1, 0]),
        ("SP", "Specificity", axes[1, 1])
    ]

    for col_key, title_str, ax in metrics:
        ax.set_facecolor("#1e1e1e")
        bars = []
        labels = []
        colors = []

        for _, row in df_subgroups.iterrows():
            labels.append(f"{row['subgroup']}\n(N={row['total_samples']})")
            bars.append(row[col_key])
            colors.append(palette.get(row["group_category"], "#4caf50"))

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, bars, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold", color="white")
        ax.set_xlabel("Score", fontsize=11, fontweight="bold", color="white")
        ax.set_title(title_str, fontsize=13, fontweight="bold", color="white", pad=10)
        ax.set_xlim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3, axis="x")

        # Value annotations
        for i, val in enumerate(bars):
            ax.text(val + 0.015, i, f"{val:.4f}", va="center", fontsize=9, fontweight="bold", color="white")

    # Legend for categories
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=palette["Age Group"], label="Age Groups [20-40, 40-50, 50+]"),
        plt.Rectangle((0, 0), 1, 1, color=palette["Site ID"], label="Acquisition Sites [Site 1, Site 2]"),
        plt.Rectangle((0, 0), 1, 1, color=palette["Breast Density"], label="Breast Density Groups [A, B, C, D, Unknown]")
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True, facecolor="#1e1e1e", edgecolor="white", fontsize=11)

    plt.suptitle("DenseNet-169 Subgroup Demographic & Fairness Audit (Held-Out Testing Set, N = 462)", fontsize=16, fontweight="bold", color="white", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[✓] Subgroup comparison plot saved to: {save_path}")


def main():
    print("=" * 85)
    print("[*] DENSENET-169 IN-DEPTH TESTING AUDIT & SUBGROUP ANALYSIS")
    print("=" * 85)

    base_dir = "/mnt/fourT/projects4TB/MAMO"
    run_dir = os.path.join(base_dir, "main_ws", "run_ws_20261030_092511")
    results_dir = os.path.join(run_dir, "results")
    splits_dir = os.path.join(run_dir, "splits")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    xai_base_dir = os.path.join(run_dir, "rsna_xai_outputs", "densenet169")
    data_dir = os.path.join(base_dir, "rsna-breast-cancer-detection")

    ckpt_path = os.path.join(checkpoints_dir, "best_densenet169.pth")
    preds_path = os.path.join(results_dir, "test_predictions_densenet169.csv")
    test_split_path = os.path.join(splits_dir, "testing_set.csv")

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    assert os.path.exists(preds_path), f"Predictions not found: {preds_path}"

    df_preds = pd.read_csv(preds_path)
    print(f"[*] Loaded testing set predictions: {len(df_preds)} examinations.")

    # Calibrated optimal threshold for DenseNet-169
    opt_thresh = 0.21976588628762542
    print(f"[*] Calibrated Optimal Decision Threshold: {opt_thresh:.4f}")

    # =========================================================================
    # PART 1: SUBGROUP FAIRNESS ANALYSIS
    # =========================================================================
    print("\n--- [Part 1: Subgroup Stratification Analysis] ---")
    df_preds["age_group_user"] = df_preds["age"].apply(assign_age_group)
    df_preds["site_label"] = df_preds["site_id"].apply(lambda s: f"Site {int(s)}")
    df_preds["density_label"] = df_preds["density_clean"].apply(lambda d: f"Density {d}")

    subgroups_records = []

    categories = [
        ("Age Group", "age_group_user", ["20-40", "40-50", "50+"]),
        ("Site ID", "site_label", ["Site 1", "Site 2"]),
        ("Breast Density", "density_label", ["Density A", "Density B", "Density C", "Density D", "Density Unknown"])
    ]

    for cat_name, col_name, order in categories:
        for sub in order:
            sub_df = df_preds[df_preds[col_name] == sub].reset_index(drop=True)
            if len(sub_df) == 0:
                continue

            y_true = sub_df["cancer"].values
            y_probs = sub_df["pred_prob"].values
            m = compute_metrics(y_true, y_probs, threshold=opt_thresh, beta=1.5)

            rec = {
                "group_category": cat_name,
                "subgroup": sub,
                **m
            }
            subgroups_records.append(rec)
            print(f"  • {cat_name:15} | {sub:16} | N={m['total_samples']:3} (Cancer={m['cancer_cases']:2}, Benign={m['non_cancer_cases']:2}) | AUC={m['AUC']:.4f} | BACC={m['balanced_ACC']:.4f} | SN={m['SN']:.4f} | SP={m['SP']:.4f}")

    df_subgroups = pd.DataFrame(subgroups_records)
    subgroups_csv = os.path.join(results_dir, "subgroup_metrics_densenet169.csv")
    subgroups_json = os.path.join(results_dir, "subgroup_metrics_densenet169.json")
    df_subgroups.to_csv(subgroups_csv, index=False)
    with open(subgroups_json, "w") as f:
        json.dump(subgroups_records, f, indent=2)
    print(f"[✓] Saved subgroup metrics table -> {subgroups_csv}")

    # Plot Subgroup Comparison
    subgroup_plot_path = os.path.join(results_dir, "subgroup_performance_densenet169.png")
    plot_subgroup_analysis(df_subgroups, subgroup_plot_path)

    # =========================================================================
    # PART 2: EXTRACT XAI MAPS FOR DENSENET-169
    # =========================================================================
    print("\n--- [Part 2: DenseNet-169 XAI Suite Generation] ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing DenseNet-169 model on device: {device}")

    model = ImageNetTransferClassifier("densenet169", pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    xai_engine = MammographyXAI(model=model, device=device)
    ensembler = ViTXAIEnsembler(xai_engine=None)
    shape_extractor = BIRADSShapeDescriptorExtractor(pixel_spacing_mm=XAI_PIXEL_SPACING_MM)

    mal_xai_dir = os.path.join(xai_base_dir, "malignant")
    ben_xai_dir = os.path.join(xai_base_dir, "benign")
    os.makedirs(mal_xai_dir, exist_ok=True)
    os.makedirs(ben_xai_dir, exist_ok=True)

    # Stratified Selection of representative samples across subgroups:
    # 1. Include ALL malignant and benign from smaller strata:
    #    - Age 20-40 (all 3 benign)
    #    - Age 40-50 (all 14 malignant, 14 benign)
    #    - Density A (all 8 malignant, 8 benign)
    #    - Density D (all 5 malignant, 5 benign)
    # 2. Add sampled cases from Density B, Density C, and Site 2 to reach 50 malignant and 50 benign (100 total).
    smaller_strata_mask = (
        (df_preds["age_group_user"].isin(["20-40", "40-50"])) |
        (df_preds["density_clean"].isin(["A", "D"]))
    )
    core_cohort = df_preds[smaller_strata_mask]
    remaining_cohort = df_preds[~smaller_strata_mask]

    pos_core = core_cohort[core_cohort["cancer"] == 1]
    neg_core = core_cohort[core_cohort["cancer"] == 0]

    needed_pos = max(0, 50 - len(pos_core))
    needed_neg = max(0, 50 - len(neg_core))

    pos_extra = remaining_cohort[remaining_cohort["cancer"] == 1].sample(n=min(needed_pos, len(remaining_cohort[remaining_cohort["cancer"] == 1])), random_state=RANDOM_SEED)
    neg_extra = remaining_cohort[remaining_cohort["cancer"] == 0].sample(n=min(needed_neg, len(remaining_cohort[remaining_cohort["cancer"] == 0])), random_state=RANDOM_SEED)

    selected_xai = pd.concat([pos_core, pos_extra, neg_core, neg_extra]).drop_duplicates(subset=["patient_id", "image_id"]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"[*] Selected stratified representative XAI cohort: {len(selected_xai)} examinations ({selected_xai['cancer'].sum()} Malignant / {len(selected_xai)-selected_xai['cancer'].sum()} Benign)")

    search_cache_dirs = [
        os.path.join(base_dir, "preprocessed_cache", "train"),
        os.path.join(base_dir, "preprocessed_cache", "val"),
        os.path.join(base_dir, "preprocessed_cache", "test"),
        os.path.join(base_dir, "main_ws", "augmented_cache")
    ]

    xai_records = []
    t0_xai = time.time()

    for idx, r in selected_xai.iterrows():
        pid = str(r["patient_id"])
        imid = str(r["image_id"])
        key = f"{pid}_{imid}"
        c_true = int(r["cancer"])
        pred_p = float(r["pred_prob"])
        pred_dec = str(r["pred_decision"])

        # Load image
        base_img = None
        for cdir in search_cache_dirs:
            cand = os.path.join(cdir, f"{key}.npy")
            if os.path.exists(cand):
                try:
                    base_img = np.load(cand).astype(np.float32)
                    break
                except Exception:
                    pass

        if base_img is None:
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

        # Compute XAI Maps
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
        dest_dir = mal_xai_dir if c_true == 1 else ben_xai_dir

        save_png = os.path.join(dest_dir, f"{pid}_{imid}_clf_xai_report.png")
        save_geo = os.path.join(dest_dir, f"{pid}_{imid}_contours.geojson")

        # Export GeoJSON
        export_contours_to_geojson(
            contour_results=contours,
            output_filepath=save_geo,
            metadata={
                "patient_id": pid,
                "image_id": imid,
                "architecture": "densenet169",
                "true_cancer": c_true,
                "pred_prob": round(pred_p, 4),
                "pred_decision": pred_dec,
                "age": float(r["age"]),
                "age_group": r["age_group_user"],
                "site_id": int(r["site_id"]),
                "density": str(r["density_clean"])
            }
        )

        # Render 6-Panel Visualization
        visualize_classification_xai(
            base_img=base_img,
            grad_cam=gcam,
            grad_cam_pp=gcam_pp,
            integrated_grads=ig,
            ensembled_map=ens_map,
            contour_overlay=overlay_rgb,
            pred_prob=pred_p,
            pred_decision=pred_dec,
            true_label=c_true,
            decision_threshold=opt_thresh,
            save_path=save_png,
            dpi=120
        )

        xai_records.append({
            "architecture": "densenet169",
            "patient_id": pid,
            "image_id": imid,
            "cancer_true": c_true,
            "true_label_folder": subfolder,
            "pred_prob": round(pred_p, 4),
            "pred_decision": pred_dec,
            "age": float(r["age"]),
            "age_group": r["age_group_user"],
            "site_id": int(r["site_id"]),
            "density": str(r["density_clean"]),
            "report_path": save_png,
            "geojson_path": save_geo,
            "num_core_contours": len(contours.get("core_contours", [])),
            "num_margin_contours": len(contours.get("margin_contours", []))
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(selected_xai):
            print(f"  [{idx+1:3}/{len(selected_xai)}] Rendered XAI report for {subfolder.upper()} case {key} ({time.time()-t0_xai:.1f}s)")

    df_xai_summary = pd.DataFrame(xai_records)
    xai_summary_path = os.path.join(xai_base_dir, "xai_summary_densenet169.csv")
    df_xai_summary.to_csv(xai_summary_path, index=False)
    print(f"[✓] DenseNet-169 XAI summary saved -> {xai_summary_path}")
    print(f"[✓] Rendered {len(df_xai_summary)} XAI reports in {xai_base_dir}")

    print("\n" + "=" * 85)
    print("[✓] DENSENET-169 ANALYSIS AND XAI EXTRACTION COMPLETED SUCCESSFULLY!")
    print("=" * 85)


if __name__ == "__main__":
    main()
