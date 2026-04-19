import torch
from mamba_ssm import Mamba

print("--- 开始 Mamba 核心算子测试 ---")

# 检查 CUDA
if not torch.cuda.is_available():
    print("❌ 错误: 没找到 GPU")
    exit()

device = "cuda"
print(f"✅ 使用设备: {torch.cuda.get_device_name(0)}")

try:
    # 定义一个小模型
    # 这一步会调用底层的 CUDA 编译算子，如果环境不对，这里会直接报错
    model = Mamba(
        d_model=16,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    # 构造假数据
    x = torch.randn(2, 64, 16).to(device)

    # 前向传播 (最关键的一步)
    y = model(x)

    print(f"✅ 计算成功！输出形状: {y.shape}")
    print("🎉 恭喜！您的 Mamba 环境可以在 RTX 3090 上完美运行！")

except Exception as e:
    print(f"❌ 测试失败！报错信息: {e}")
