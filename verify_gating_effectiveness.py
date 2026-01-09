"""
快速验证门控机制是否有效的测试脚本
检查项目：
1. 模型是否包含门控模块
2. 门控权重是否在合理范围内
3. 门控权重是否会因不同输入而变化
"""

import torch
import sys
from pyprojroot import here as project_root
sys.path.insert(0, str(project_root()))

from fs_mol.utils.adaptive_dkt_utils import ADKTModelTrainer
from fs_mol.data import FSMolDataset, DataFold
from fs_mol.data.dkt import get_dkt_task_sample_iterable, get_dkt_batcher
from fs_mol.utils.torch_utils import torchify

def verify_gating():
    print("=" * 60)
    print("🔍 门控机制有效性验证")
    print("=" * 60)
    
    # 1. 加载已训练的模型
    model_path = "./outputs/FSMol_ADKTModel_gnn+ecfp+pc-descs+fc_2025-12-28_01-18-12/best_validation.pt"
    print(f"\n📂 加载模型: {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    model = ADKTModelTrainer.build_from_model_file(model_path, device=device)
    model = model.to(device)
    model.eval()
    
    # 2. 检查门控模块是否存在
    print("\n" + "=" * 60)
    print("✅ 检查 1: 门控模块是否存在")
    print("=" * 60)
    
    if hasattr(model, 'modality_gate'):
        print("   ✓ 模型包含 modality_gate 模块")
        gate_params = sum(p.numel() for p in model.modality_gate.parameters())
        print(f"   ✓ 门控网络参数数量: {gate_params}")
    else:
        print("   ✗ 模型不包含 modality_gate 模块！")
        return
    
    # 3. 检查门控配置
    print("\n" + "=" * 60)
    print("✅ 检查 2: 门控配置")
    print("=" * 60)
    print(f"   use_modality_gating: {model.config.use_modality_gating}")
    print(f"   gating_hidden_dim: {model.config.gating_hidden_dim}")
    
    # 4. 用不同任务测试门控权重变化
    print("\n" + "=" * 60)
    print("✅ 检查 3: 门控权重在不同任务上的变化")
    print("=" * 60)
    
    dataset = FSMolDataset.from_directory("./fs-mol-dataset")
    batcher = get_dkt_batcher(max_num_graphs=128)
    
    task_iterator = iter(get_dkt_task_sample_iterable(
        dataset=dataset,
        data_fold=DataFold.VALIDATION,
        num_samples=1,
        max_num_graphs=128,
        support_size=32,
        query_size=64,
        repeat=False,
    ))
    
    gate_weights_list = []
    task_names = []
    
    print("\n   测试不同任务的门控权重:")
    for i, task_sample in enumerate(task_iterator):
        if i >= 5:  # 只测试5个任务
            break
        
        task_sample = torchify(task_sample, device=device)
        batch = task_sample.batches[0]
        
        with torch.no_grad():
            # 前向传播，触发门控计算
            _ = model(batch, train_loss=None)
        
        if hasattr(model, 'last_gate_weights') and model.last_gate_weights:
            gate_weights = {k: v.item() for k, v in model.last_gate_weights.items()}
            gate_weights_list.append(gate_weights)
            task_names.append(task_sample.task_name)
            
            gate_str = " | ".join([f"{k}: {v:.4f}" for k, v in gate_weights.items()])
            print(f"   任务 {i+1} ({task_sample.task_name[:20]}...): {gate_str}")
    
    # 5. 分析门控权重
    print("\n" + "=" * 60)
    print("📊 门控权重统计分析")
    print("=" * 60)
    
    if gate_weights_list:
        modalities = list(gate_weights_list[0].keys())
        for mod in modalities:
            values = [gw[mod] for gw in gate_weights_list]
            mean_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            
            print(f"\n   {mod}:")
            print(f"      均值: {mean_val:.4f}")
            print(f"      范围: [{min_val:.4f}, {max_val:.4f}]")
            print(f"      方差: {variance:.6f}")
    
    # 6. 结论
    print("\n" + "=" * 60)
    print("📋 结论")
    print("=" * 60)
    
    if gate_weights_list:
        # 检查权重是否有变化
        modalities = list(gate_weights_list[0].keys())
        has_variance = any(
            max([gw[mod] for gw in gate_weights_list]) - min([gw[mod] for gw in gate_weights_list]) > 0.01
            for mod in modalities
        )
        
        if has_variance:
            print("   ✅ 门控权重随任务变化，门控机制正在工作！")
        else:
            print("   ⚠️ 门控权重变化很小，可能需要更多训练或调整超参数")
        
        # 检查权重范围
        all_in_range = all(
            0 < gw[mod] < 1 
            for gw in gate_weights_list 
            for mod in modalities
        )
        if all_in_range:
            print("   ✅ 门控权重在有效范围 (0, 1) 内")
        else:
            print("   ⚠️ 部分门控权重可能接近边界值")
    else:
        print("   ❌ 未能获取门控权重，请检查门控集成")

if __name__ == "__main__":
    verify_gating()
