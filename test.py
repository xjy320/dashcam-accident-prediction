print('='*50)
print('🚀 深度学习环境验证报告')
print('='*50)

# 1. 验证PyTorch（核心）
torch_available = False
cuda_available = False 
try:
    import torch
    torch_available = True
    cuda_available = torch.cuda.is_available()
    print(f'✅ PyTorch 版本: {torch.__version__}')
    print(f'   CUDA 可用: {cuda_available}')
    if cuda_available:
        print(f'   GPU 设备: {torch.cuda.get_device_name(0)}')
    # 自动选择训练设备，并做一次设备张量验证
    device = 'cuda' if cuda_available else 'cpu'
    print(f'   当前训练设备: {device}')
    try:
        _x = torch.rand(2, 3, device=device)
        print(f'   设备张量示例: device={_x.device}, shape={_x.shape}')
    except Exception as dev_e:
        print(f'   设备张量创建失败: {dev_e}')
except Exception as e:
    print(f'❌ PyTorch 导入失败: {e}')

# 2. 验证其他核心库
libs = ['numpy']
for lib in libs:
    try:
        __import__(lib)
        version = __import__(lib).__version__ if hasattr(__import__(lib), '__version__') else '未知'
        print(f'✅ {lib:12} 版本: {version}')
    except:
        print(f'❌ {lib:12} 导入失败')

print('='*50)
if cuda_available:
    print('🎉 完美！GPU加速已就绪，可以开始训练模型了！')
else:
    print('⚠️  GPU不可用，训练会较慢，建议后续配置GPU或使用Colab')
print('='*50)