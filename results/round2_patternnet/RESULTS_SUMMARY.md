# SG-CAM — PatternNet results

Generated Thu Oct 22 21:57:00 2026 | device: Tesla T4 | torch 2.11.0+cu128

Dataset root: `/kaggle/input/datasets/samitsaleem/patternnet-scene-classification-dataset/PatternNet_Images` | 30399 images, 38 classes | split sizes: {'train': 18239, 'val': 6080, 'test': 6080}

Representative run for figures: `mobilenet_v2_sgcam_cl2t8_i256_d256_lr0.0001_bs16_ep15_s0`

## T3_training_parameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Batch size | 16 |
| Learning-rate schedule | Step decay ×0.1 every 10 epochs |
| Epochs | 15 (best epoch selected on validation accuracy) |
| Loss function | Cross-entropy (scene labels only) |
| Input size | 256×256 |
| CLAHE | clip limit 2.0, tile grid 8×8 (L channel, LAB) |
| Data split | Stratified 60/20/20 train/validation/test, split seed 42 |
| Random seeds | 0, 1, 2 |
| Mixed precision | FP16 autocast |
| Semantic encoder | nvidia/segformer-b2-finetuned-ade-512-512 (frozen) |
| SG-CAM dimension d | 256 |

## T4_main_results

| Dataset | Model | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) | Weighted F1 (%) | Cohen's κ (×100) | Top-5 accuracy (%) | Macro AUC (OvR) | Seeds (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| PatternNet (test) | SG-CAM (MobileNetV2) | 99.57 ± 0.03 | 99.57 ± 0.02 | 99.57 ± 0.03 | 99.57 ± 0.02 | 99.57 ± 0.02 | 99.56 ± 0.03 | 99.99 ± 0.01 | 1.0000 ± 0.0000 | 3 |

## T_per_class_metrics

| Class | Test images | Precision (%) | Recall (%) | F1 (%) |
|---|---|---|---|---|
| Airplane | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Baseball Field | 160 | 100.00 ± 0.00 | 99.79 ± 0.36 | 99.90 ± 0.18 |
| Basketball Court | 160 | 98.95 ± 0.36 | 97.71 ± 0.72 | 98.32 ± 0.19 |
| Beach | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Bridge | 160 | 100.00 ± 0.00 | 98.75 ± 0.62 | 99.37 ± 0.32 |
| Cemetery | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Chaparral | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Christmas Tree Farm | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Closed Road | 160 | 99.37 ± 0.01 | 98.96 ± 1.30 | 99.16 ± 0.66 |
| Coastal Mansion | 160 | 98.32 ± 0.36 | 97.71 ± 0.36 | 98.01 ± 0.18 |
| Crosswalk | 160 | 99.79 ± 0.36 | 100.00 ± 0.00 | 99.90 ± 0.18 |
| Dense Residential | 160 | 98.76 ± 0.61 | 98.96 ± 0.36 | 98.86 ± 0.18 |
| Ferry Terminal | 160 | 98.16 ± 1.06 | 99.38 ± 0.00 | 98.76 ± 0.53 |
| Football Field | 160 | 99.79 ± 0.36 | 100.00 ± 0.00 | 99.90 ± 0.18 |
| Forest | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Freeway | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Golf Course | 160 | 99.79 ± 0.36 | 100.00 ± 0.00 | 99.90 ± 0.18 |
| Harbor | 160 | 99.37 ± 0.00 | 99.17 ± 0.36 | 99.27 ± 0.18 |
| Intersection | 160 | 98.97 ± 0.35 | 100.00 ± 0.00 | 99.48 ± 0.18 |
| Mobile Home Park | 160 | 100.00 ± 0.00 | 99.38 ± 0.00 | 99.69 ± 0.00 |
| Nursing Home | 160 | 100.00 ± 0.00 | 98.75 ± 0.62 | 99.37 ± 0.32 |
| Oil Gas Field | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Oil Well | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Overpass | 160 | 99.59 ± 0.36 | 100.00 ± 0.00 | 99.79 ± 0.18 |
| Parking Lot | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Parking Space | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Railway | 160 | 98.97 ± 0.35 | 99.79 ± 0.36 | 99.38 ± 0.00 |
| River | 160 | 99.59 ± 0.36 | 100.00 ± 0.00 | 99.79 ± 0.18 |
| Runway | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Runway Marking | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Shipping Yard | 160 | 100.00 ± 0.00 | 99.79 ± 0.36 | 99.90 ± 0.18 |
| Solar Panel | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Sparse Residential | 160 | 96.12 ± 0.69 | 97.92 ± 0.36 | 97.01 ± 0.47 |
| Storage Tank | 160 | 98.97 ± 0.35 | 100.00 ± 0.00 | 99.48 ± 0.18 |
| Swimming Pool | 160 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |
| Tennis Court | 160 | 100.00 ± 0.00 | 99.17 ± 0.36 | 99.58 ± 0.18 |
| Transformer Station | 160 | 99.79 ± 0.36 | 99.38 ± 0.00 | 99.58 ± 0.18 |
| Wastewater Treatment Plant | 160 | 99.38 ± 0.62 | 98.96 ± 0.36 | 99.17 ± 0.18 |

## T8_ablation

| Configuration | CLAHE | Semantic stream (SegFormer-B2) | Fusion | Accuracy (%) | Macro F1 (%) | Macro AUC (OvR) | Seeds (n) | McNemar p vs. full (max over seeds) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 (baseline, no CLAHE) | No | No | None | – | – | – | 0 | – |
| MobileNetV2 + CLAHE | Yes | No | None | 99.68 ± 0.03 | 99.68 ± 0.03 | 1.0000 ± 0.0000 | 3 | 3.27e-01 |
| MobileNetV2 + CLAHE + SegFormer (concatenation) | Yes | Yes | Concatenation | – | – | – | 0 | – |
| MobileNetV2 + CLAHE + SegFormer (addition) | Yes | Yes | Element-wise addition | – | – | – | 0 | – |
| MobileNetV2 + SegFormer + SG-CAM (no CLAHE) | No | Yes | SG-CAM | – | – | – | 0 | – |
| Full model: CLAHE + SegFormer + SG-CAM | Yes | Yes | SG-CAM | 99.57 ± 0.03 | 99.57 ± 0.02 | 1.0000 ± 0.0000 | 3 | – |

## T7_backbones

| Backbone | Without SG-CAM: accuracy (%) | With SG-CAM: accuracy (%) | Without SG-CAM: macro F1 (%) | With SG-CAM: macro F1 (%) | Δ accuracy (pp) | Trainable params with SG-CAM (M) | Seeds (n) |
|---|---|---|---|---|---|---|---|
| MobileNetV2 | 99.68 ± 0.03 | 99.57 ± 0.03 | 99.68 ± 0.03 | 99.57 ± 0.02 | -0.11 | 3.95 | 3/3 |
| ResNet-50 | – | – | – | – | – | – | 0/0 |
| DenseNet-121 | – | – | – | – | – | – | 0/0 |
| EfficientNet-B0 | – | – | – | – | – | – | 0/0 |

## T9_sensitivity

| Parameter | Value | Validation accuracy (%) | Validation macro F1 (%) | Epochs | Seed |
|---|---|---|---|---|---|
| CLAHE clip limit | 1.5 | – | – | 10 | 0 |
| CLAHE clip limit | 2 (default) | – | – | 10 | 0 |
| CLAHE clip limit | 3 | – | – | 10 | 0 |
| CLAHE tile grid size | 6×6 | – | – | 10 | 0 |
| CLAHE tile grid size | 8×8 (default) | – | – | 10 | 0 |
| CLAHE tile grid size | 10×10 | – | – | 10 | 0 |
| Embedding dimension d | 128 | – | – | 10 | 0 |
| Embedding dimension d | 256 (default) | – | – | 10 | 0 |
| Embedding dimension d | 512 | – | – | 10 | 0 |
| Learning rate | 1e-05 | – | – | 10 | 0 |
| Learning rate | 0.0001 (default) | – | – | 10 | 0 |
| Learning rate | 0.001 | – | – | 10 | 0 |
| Batch size | 8 | – | – | 10 | 0 |
| Batch size | 16 (default) | – | – | 10 | 0 |
| Batch size | 32 | – | – | 10 | 0 |

## T5_complexity_measured

| Model | Trainable params (M) | Frozen params (M) | Total params (M) | GFLOPs / image | Latency, batch 1 (ms) | Throughput, batch 8 (img/s) |
|---|---|---|---|---|---|---|
| MobileNetV2 baseline (no semantic stream) | 2.90 | 0.00 | 2.90 | 0.78 | 5.5 | 1261.0 |
| MobileNetV2 + SegFormer-B2 (concatenation) | 3.23 | 27.46 | 30.69 | 30.52 | 27.1 | 160.6 |
| MobileNetV2 + SegFormer-B2 + SG-CAM (proposed) | 3.95 | 27.46 | 31.41 | 30.63 | 28.0 | 157.7 |
| SG-CAM module only | 1.05 | 0.00 | 1.05 | 0.138 | – | – |

## Runs still pending

None
