# Weakly Supervised Breast Cancer Screening with Fairness & Explainable AI (FAIR-XAI)

A comprehensive deep transfer learning and explainable artificial intelligence (XAI) framework for binary mammographic lesion classification and demographic fairness audits.

## Highlights
- **18 Pretrained ImageNet Transfer Architectures**: Fine-tuning sweep across DenseNet (`121`, `169`, `201`), ResNet (`50`, `101`, `152`, `V2-50`, `V2-101`, `V2-152`), VGG (`16`, `19`), Inception (`V3`, `ResNet-V2`), MobileNet (`V1`, `V2`), NASNet (`Large`, `MNASNet`), and Xception.
- **Offline 40-Fold Data Augmentation**: Large-scale spatial, affine, and photometric augmentation pipeline applied strictly to training partitions with zero leakage to validation/testing cohorts.
- **Clinical Sensitivity Calibration**: Automated operating point threshold optimization maximizing clinical $F_{\beta}$ ($\beta = 1.5$) sensitivity on validation distributions.
- **Subgroup Fairness Audits**: Multi-axial fairness evaluation across age groups (`[20-40, 40-50, 50+]`), acquisition scanner sites (`[Site 1, Site 2]`), and breast density categories (`[A, B, C, D, Unknown]`).
- **Explainability Suite (XAI)**: 6-panel explainability pipeline featuring Grad-CAM, Grad-CAM++, Integrated Gradients, Harmonic Ensembling, and Dual Iso-Confidence Contour Outlines (Core Nidus vs. Margin).

## Repository Structure
- `main_ws.py`: Main transfer learning training, calibration, and benchmark orchestration pipeline.
- `analyze_densenet169.py`: In-depth subgroup fairness audit and XAI extraction suite for DenseNet-169.
- `transfer_learning/`: Model wrappers and definitions from standard backbones and 3D application adaptations.
- `MammographyXAI.py` & `MammoCLIPViTXAI.py`: Gradient, path attribution, and attention saliency engines.
- `ViTXAIEnsembler.py`: Harmonic ensembling and dual iso-confidence boundary extraction.
- `preprocessing_RSNA_MAMO.py`: Standardized mammographic preprocessing and anatomical pectoral muscle excision.
- `senAnals.py`: Clinical sensitivity calibration and threshold optimization.
- `visualize_clf_xai.py` & `visualize_detector_xai.py`: Publication-grade diagnostic report visualization generators.

## Usage
```bash
python main_ws.py --help
python analyze_densenet169.py
```
