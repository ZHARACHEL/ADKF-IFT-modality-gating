"""
快速检查门控权重 - 不需要GPU，不需要加载数据
直接读取模型checkpoint分析
"""
import torch
import os

print("=" * 60)
print("快速检查门控权重（CPU模式，不加载数据）")
print("=" * 60)

# 找到所有模型
outputs_dir = "./outputs"
models = []

for folder in sorted(os.listdir(outputs_dir)):
    if folder.startswith("FSMol_ADKTModel"):
        model_path = os.path.join(outputs_dir, folder, "best_validation.pt")
        if os.path.exists(model_path):
            models.append((folder, model_path))

print(f"\n找到 {len(models)} 个模型")

# 分析每个模型
for folder, model_path in models[-5:]:  # 只看最近5个
    print(f"\n{'='*60}")
    print(f"模型: {folder}")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('model_config')
        state_dict = checkpoint.get('model_state_dict', {})
        
        # 配置信息
        if config:
            gating = getattr(config, 'use_modality_gating', 'N/A')
            adapter = getattr(config, 'use_statistics_adapter', 'N/A')
            print(f"  use_modality_gating: {gating}")
            print(f"  use_statistics_adapter: {adapter}")
        
        # 分析门控权重
        print(f"\n  门控网络权重分析:")
        gate_params = []
        for name, param in state_dict.items():
            if 'gate' in name.lower():
                gate_params.append((name, param))
                print(f"    {name}:")
                print(f"      shape: {param.shape}")
                print(f"      mean: {param.mean().item():.4f}")
                print(f"      std: {param.std().item():.4f}")
                print(f"      min: {param.min().item():.4f}")
                print(f"      max: {param.max().item():.4f}")
        
        if not gate_params:
            print("    (无门控参数)")
        
        # 分析adapter权重
        print(f"\n  Adapter权重分析:")
        adapter_params = []
        for name, param in state_dict.items():
            if 'adapter' in name.lower():
                adapter_params.append((name, param))
                print(f"    {name}:")
                print(f"      shape: {param.shape}")
                print(f"      mean: {param.mean().item():.4f}")
                print(f"      std: {param.std().item():.4f}")
        
        if not adapter_params:
            print("    (无Adapter参数)")
            
    except Exception as e:
        print(f"  错误: {e}")

print(f"\n{'='*60}")
print("检查完成")
print("=" * 60)
