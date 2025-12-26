import torch

print(f"PyTorch版本: {torch.__version__}")

if torch.cuda.is_available():
    print("✅ CUDA可用 - GPU支持已启用")
    print(f"   CUDA版本: {torch.version.cuda}")
    print(f"   可用GPU数量: {torch.cuda.device_count()}")
    print(f"   当前GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ CUDA不可用 - 仅限CPU")
    print("   请检查: 1) NVIDIA驱动 2) CUDA安装 3) PyTorch的CUDA版本")

# 可选：测试简单的张量运算
x = torch.rand(3, 3)
print(f"\n测试张量运算: {x.mean():.4f}")