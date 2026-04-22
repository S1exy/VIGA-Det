"""VIGA-Det 评估与可视化脚本.

对 OBB 检测模型执行:
  1. 测试集定量指标 (mAP50 / mAP50-95 / per-class AP)
  2. 抽取若干张测试图进行旋转框可视化推理

使用前请根据本地环境修改 MODEL_PATH, DATA_CFG, VAL_IMAGES_DIR.
"""
import os
import glob

from ultralytics import YOLO
from ultralytics.utils import torch_utils


def safe_get_flops(model, imgsz=640):
    return 0.0


torch_utils.get_flops = safe_get_flops


# ===== 路径配置 (请按本地环境修改) =====
MODEL_PATH = "../runs/obb/viga_det_paper_aligned/weights/best.pt"
DATA_CFG = "../datasets/dataset2_obb/data.yaml"
VAL_IMAGES_DIR = "../datasets/dataset2_obb/test/images"
NUM_VIS_IMAGES = 10


def main():
    print("[VIGA-Det] 加载权重:", MODEL_PATH)
    model = YOLO(MODEL_PATH, task="obb")

    # ---------- 定量评估 ----------
    print("[VIGA-Det] 在测试集上评估 OBB 指标 ...")
    metrics = model.val(
        data=DATA_CFG,
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        plots=True,
        save_json=True,
        conf=0.001,
        iou=0.7,
    )

    print("=" * 40)
    print("核心测试指标")
    print("=" * 40)
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print("-- per-class mAP50-95 --")
    for i, class_name in enumerate(metrics.names.values()):
        print(f"  {class_name:<12}: {metrics.box.maps[i]:.4f}")

    # ---------- 可视化推理 ----------
    print(f"[VIGA-Det] 抽取 {NUM_VIS_IMAGES} 张测试图做旋转框可视化 ...")
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    img_paths = sorted(
        p for e in exts for p in glob.glob(os.path.join(VAL_IMAGES_DIR, e))
    )[:NUM_VIS_IMAGES]

    if not img_paths:
        print("  [!] 未找到测试图, 请检查 VAL_IMAGES_DIR 与后缀.")
        return

    results = model.predict(
        source=img_paths,
        save=True,
        imgsz=640,
        conf=0.25,
        device=0,
        line_width=2,
        show_labels=True,
        show_conf=True,
    )
    print(f"  可视化结果保存到: {results[0].save_dir}")


if __name__ == "__main__":
    main()
