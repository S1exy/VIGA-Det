# VIGA-Det

An end-to-end rotated object detection framework for UAV-view RGB-IR dual-modality traffic scenes, built on a modified Ultralytics YOLOv11 OBB pipeline.

## Project Structure

```
my_training_setup/
├── .gitignore
├── README.md
├── requirements.txt
├── A18_Project/
│   ├── code/
│   │   ├── train.py
│   │   └── eval_and_vis.py
│   └── datasets/
│       └── dataset2_obb/
│           └── data.yaml
└── ultralytics_yolo11_mamba/
    └── ultralytics/
        ├── cfg/models/11/yolo11-spatial-iaga-obb-spd.yaml
        ├── nn/modules/block.py
        └── utils/loss.py
```

## Deployment

```bash
# 1. PyTorch (pick the CUDA wheel matching your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install the patched Ultralytics fork
cd ultralytics_yolo11_mamba && pip install -e . && cd ..

# 3. Extra dependencies
pip install -r requirements.txt
```
