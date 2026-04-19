"""训练脚本: VIGA-Det (paper-aligned).

对齐论文 §3 VIGA-Det 方法章节与 §4.1.2 实现细节:
  - SPD-Conv 细节保留
  - SIAGA 跨模态对齐融合 (含 (1-W_ill) 加权 IR 分支 + VIB 重参数化)
  - SSM-CSP 四方向扫描长程上下文
  - ProbIoU 旋转框回归
  - 总损失: L_total = 7.5·L_ProbIoU + 0.5·L_Cls + 0.01·L_VIB  (论文式 26)

使用前请根据本地路径修改 DATA_CFG 和 MODEL_CFG.
"""
import faulthandler
faulthandler.enable()
import cv2
cv2.setNumThreads(0)

from ultralytics import YOLO
from ultralytics.utils import torch_utils


def safe_get_flops(model, imgsz=640):
    """DCNv2 参与时 thop 无法正确计算 FLOPs, 这里暂时屏蔽."""
    return 0.0


torch_utils.get_flops = safe_get_flops


# ===== 路径配置 (请按本地环境修改) =====
DATA_CFG = "../datasets/dataset2_obb/data.yaml"
MODEL_CFG = "yolo11-spatial-iaga-obb-spd.yaml"


def main():
    model = YOLO(MODEL_CFG, task="obb")

    model.train(
        # ---- 训练规模 (论文 §4.1.2) ----
        data=DATA_CFG,
        epochs=100,
        batch=8,
        imgsz=640,          # 配合预处理后 840x672 原图 → 缩放填充为 640x512
        device=0,

        project="VIGA_Det",
        name="viga_det_paper_aligned",
        exist_ok=True,

        # ---- 优化器 (论文 §4.1.2) ----
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,           # 终值 lr = lr0 * lrf = 1e-5
        weight_decay=5e-4,
        warmup_epochs=5.0,

        # ---- 数据增强 (论文 §4.1.2 明确列出的 4 项 + close_mosaic) ----
        # HSV 禁用以保护红外热辐射分布
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        fliplr=0.5,         # 概率 0.5 水平翻转
        scale=0.5,          # 因子 0.5 随机缩放
        translate=0.1,      # 因子 0.1 随机平移
        erasing=0.4,        # 概率 0.4 随机擦除
        close_mosaic=10,    # 最后 10 epoch 关闭 Mosaic

        # ---- 训练策略 ----
        cache=False,
        amp=False,
        workers=8,
        deterministic=False,
    )


if __name__ == "__main__":
    main()
