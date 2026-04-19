# VIGA-Det

Official implementation of **VIGA-Det: An End-to-End Rotated Object Detection Framework for UAV-View RGB-IR Dual-Modality Traffic Scenes**.

VIGA-Det targets four compound difficulties in UAV traffic surveillance: tiny / dense targets, weak spatial alignment between RGB and IR, long-range context requirements, and densely rotated bounding boxes. The framework is built on the Ultralytics YOLOv11 OBB pipeline and is composed of four core modules:

1. **SPD-Conv** — lossless space-to-depth downsampling that preserves tiny-target detail.
2. **SIAGA** (Spatial Illumination-Aware Gated Alignment) — per-pixel illumination reliability estimation + deformable cross-modality alignment + gated fusion, regularised by an entropy-gated Variational Information Bottleneck (VIB).
3. **SSM-CSP** — CSP-style selective state-space block with four-way (row-forward / row-reverse / column-forward / column-reverse) scanning, plugged into P3 and P4 only.
4. **ProbIoU** — 2D Gaussian-based rotated-box regression loss.

The joint objective follows paper Eq. (26):

```
L_total = 7.5 · L_ProbIoU + 0.5 · L_Cls + 0.01 · L_VIB
```

---

## 1. Repository layout

```
my_training_setup/
├── A18_Project/
│   ├── code/
│   │   ├── train.py            # paper-aligned training entry
│   │   └── eval_and_vis.py     # validation + visualisation entry
│   └── datasets/
│       └── dataset2_obb/
│           └── data.yaml       # dataset template (images NOT included)
└── ultralytics_yolo11_mamba/   # modified Ultralytics codebase
    └── ultralytics/
        ├── cfg/models/11/
        │   └── yolo11-spatial-iaga-obb-spd.yaml   # VIGA-Det network config
        ├── nn/modules/
        │   └── block.py        # Spatial_IAGA_Module / C2f_Mamba / SPDConv / ...
        └── utils/
            └── loss.py         # v8OBBLoss aligned to Eq. (26)
```

---

## 2. Environment

Tested with:

- Python 3.10
- PyTorch 2.1+ with CUDA 11.8 / 12.1
- NVIDIA GPU with SM ≥ 8.0 (required by `mamba-ssm` CUDA kernels; RTX 3090 in our experiments)

Install the modified Ultralytics in editable mode (so `block.py` / `loss.py` changes take effect), then the remaining runtime dependencies:

```bash
# 1. PyTorch (pick the CUDA wheel matching your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. This patched Ultralytics fork
cd ultralytics_yolo11_mamba && pip install -e . && cd ..

# 3. Extra dependencies (mamba-ssm, causal-conv1d)
pip install -r requirements.txt
```

> **Note**: `mamba-ssm` only runs on CUDA GPUs. The `C2f_Mamba` block in `block.py` falls back to identity on CPU (training must happen on GPU).

> **Ultralytics telemetry** (optional): the bundled Ultralytics code sends anonymous crash reports via Sentry by default. To disable, run `yolo settings sync=False` once after installation.

---

## 3. Dataset

We evaluate on **DroneVehicle** (Sun *et al.*, 2022). The dataset is **not** shipped here — download the official release and arrange files as:

```
datasets/dataset2_obb/
├── data.yaml              # provided template
├── train/{images,labels}/
├── val/{images,labels}/
└── test/{images,labels}/
```

Labels use Ultralytics OBB format: `cls x_c y_c w h theta` with normalised coordinates, `theta` in radians.

### Required preprocessing (paper §4.1.1)

The following steps are **not in this repo** — the paper describes them and we assume the user runs them before training:

1. **Border cropping**: the official DroneVehicle images are 840×712 with large white borders; crop away the borders (authors crop to an aspect ratio near 5:4, yielding 640×512 after down-scaling in the dataloader).
2. **Cross-modality label unification**: map each RGB annotation onto the IR frame; when an RGB–IR pair has rotated IoU > 0.3 treat it as the same instance and keep the IR box as supervision; otherwise take the union.
3. **Day / night split (for evaluation only)**: classify a test image as night when the RGB mean pixel intensity is below 60.

Edit `A18_Project/datasets/dataset2_obb/data.yaml` and set `path:` to the absolute path of your prepared dataset.

---

## 4. Training

```bash
cd A18_Project/code
python train.py
```

Default hyperparameters (already set in `train.py`, matching paper §4.1.2):

| Item | Value |
|------|-------|
| Epochs | 100 |
| Batch size | 8 |
| Image size | 640 (long side) → 640×512 after padding |
| Optimizer | AdamW |
| lr0 / lrf | 1e-3 / 0.01 |
| Weight decay | 5e-4 |
| Warmup epochs | 5 |
| Augmentation | fliplr 0.5, scale 0.5, translate 0.1, erasing 0.4, HSV off |
| `close_mosaic` | 10 (last 10 epochs) |

Checkpoints are written to `runs/obb/viga_det_paper_aligned/` (git-ignored).

---

## 5. Evaluation

```bash
cd A18_Project/code
python eval_and_vis.py
```

`eval_and_vis.py`:

- Runs `model.val(split="test")` and prints mAP50 / mAP50-95 / per-class AP.
- Saves PR curves and a confusion matrix under `runs/obb/.../val`.
- Runs `model.predict` on the first 10 test images with oriented boxes drawn.

Adjust `MODEL_PATH`, `DATA_CFG`, `VAL_IMAGES_DIR` at the top of `eval_and_vis.py` before running.

---

## 6. Module cheat sheet

| Module | File | Paper Eq. |
|--------|------|-----------|
| `SPDConv` | `ultralytics/nn/modules/block.py` | Eq. (1)(2) |
| `Spatial_IAGA_Module` | `ultralytics/nn/modules/block.py` | Eq. (2)(3)(4)(7)(9)(12)(14)(16) |
| `C2f_Mamba` (SSM-CSP, 4-way scan) | `ultralytics/nn/modules/block.py` | Eq. (17)(18) + §3.3 |
| `RotatedBboxLoss` (ProbIoU) | `ultralytics/utils/loss.py` | Eq. (22)(23)(24)(25) |
| `v8OBBLoss.loss` (paper total) | `ultralytics/utils/loss.py` | Eq. (26) |
| Network graph | `ultralytics/cfg/models/11/yolo11-spatial-iaga-obb-spd.yaml` | Fig. 2 |

---

## 7. License and citation

Code is released under the AGPL-3.0 license inherited from Ultralytics YOLOv11 (`ultralytics_yolo11_mamba/LICENSE`). If you build on this work, please cite the accompanying competition paper (VIGA-Det) together with:

- Sun *et al.*, "Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning", TCSVT 2022 (DroneVehicle).
- Gu & Dao, "Mamba: Linear-time sequence modeling with selective state spaces", 2023.
- Murrugarra-Llerena *et al.*, "ProbIoU: probabilistic intersection over union for rotated bounding boxes", 2024.
