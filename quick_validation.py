"""
快速验证门控和Adapter有效性的脚本
无需完整训练，直接分析已训练模型的门控行为
"""
import torch
import numpy as np
import os
import sys
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, '.')

from fs_mol.utils.adaptive_dkt_utils import ADKTModelTrainer
from fs_mol.data import FSMolDataset, DataFold
from fs_mol.data.dkt import get_dkt_task_sample_iterable
from fs_mol.utils.torch_utils import torchify

def analyze_gate_weights(model_path, num_tasks=30):
    """分析门控权重在不同任务上的分布"""
    
    print("=" * 70)
    print("🔍 门控权重分析")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📱 设备: {device}")
    
    # 加载模型
    print(f"📂 加载模型: {model_path}")
    model = ADKTModelTrainer.build_from_model_file(model_path, device=device)
    model = model.to(device)
    model.eval()
    
    # 检查配置
    print(f"\n⚙️ 模型配置:")
    print(f"   use_modality_gating: {model.config.use_modality_gating}")
    print(f"   use_statistics_adapter: {model.config.use_statistics_adapter}")
    print(f"   used_features: {model.config.used_features}")
    
    if not model.config.use_modality_gating:
        print("\n❌ 该模型未启用门控，无法分析门控权重！")
        return
    
    # 加载数据集
    print(f"\n📊 加载验证集...")
    dataset = FSMolDataset.from_directory("./fs-mol-dataset")
    
    task_iter = iter(get_dkt_task_sample_iterable(
        dataset=dataset,
        data_fold=DataFold.VALIDATION,
        num_samples=1,
        max_num_graphs=64,
        support_size=16,
        query_size=32,
        repeat=False,
    ))
    
    # 收集门控权重
    gate_weights_by_modality = defaultdict(list)
    task_names = []
    
    print(f"\n🔬 分析 {num_tasks} 个任务的门控权重...")
    
    for i in range(num_tasks):
        try:
            task_sample = next(task_iter)
            task_sample = torchify(task_sample, device=device)
            batch = task_sample.batches[0]
            
            with torch.no_grad():
                _ = model(batch, train_loss=None)
            
            if hasattr(model, 'last_gate_weights') and model.last_gate_weights:
                task_names.append(task_sample.task_name)
                for modality, weight in model.last_gate_weights.items():
                    gate_weights_by_modality[modality].append(weight.item())
                
                if (i + 1) % 10 == 0:
                    print(f"   已处理 {i+1}/{num_tasks} 个任务")
        except StopIteration:
            break
    
    # 分析结果
    print("\n" + "=" * 70)
    print("📊 门控权重统计分析")
    print("=" * 70)
    
    for modality, weights in gate_weights_by_modality.items():
        weights = np.array(weights)
        print(f"\n【{modality.upper()}】")
        print(f"   均值: {weights.mean():.4f}")
        print(f"   标准差: {weights.std():.4f}")
        print(f"   最小值: {weights.min():.4f}")
        print(f"   最大值: {weights.max():.4f}")
        print(f"   范围: {weights.max() - weights.min():.4f}")
    
    # 判断门控是否有效
    print("\n" + "=" * 70)
    print("🎯 有效性判断")
    print("=" * 70)
    
    all_stds = [np.array(w).std() for w in gate_weights_by_modality.values()]
    avg_std = np.mean(all_stds)
    
    all_ranges = [np.array(w).max() - np.array(w).min() for w in gate_weights_by_modality.values()]
    avg_range = np.mean(all_ranges)
    
    print(f"\n   平均标准差: {avg_std:.4f}")
    print(f"   平均范围: {avg_range:.4f}")
    
    if avg_std < 0.05:
        print("\n   ⚠️ 门控权重变化很小（std < 0.05）")
        print("   → 门控可能没有学到有意义的模态选择")
        print("   → 建议：增加训练步数或调整门控网络结构")
    elif avg_std < 0.1:
        print("\n   📊 门控权重有一定变化（0.05 < std < 0.1）")
        print("   → 门控正在学习，但效果可能不够明显")
    else:
        print("\n   ✅ 门控权重变化明显（std > 0.1）")
        print("   → 门控正在根据任务特性调整模态权重")
    
    # 检查模态间差异
    if len(gate_weights_by_modality) > 1:
        modalities = list(gate_weights_by_modality.keys())
        means = [np.array(gate_weights_by_modality[m]).mean() for m in modalities]
        max_diff = max(means) - min(means)
        
        print(f"\n   模态间均值差异: {max_diff:.4f}")
        if max_diff > 0.1:
            print("   ✅ 不同模态有不同的平均权重")
        else:
            print("   ⚠️ 所有模态权重相似，门控可能没有学到区分")
    
    return gate_weights_by_modality, task_names


def quick_ablation_check():
    """快速对比不同配置的模型"""
    
    print("\n" + "=" * 70)
    print("🧪 快速消融分析")
    print("=" * 70)
    
    outputs_dir = "./outputs"
    models_found = []
    
    for folder in os.listdir(outputs_dir):
        model_path = os.path.join(outputs_dir, folder, "best_validation.pt")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                config = checkpoint.get('model_config')
                if config:
                    models_found.append({
                        'folder': folder,
                        'path': model_path,
                        'gating': getattr(config, 'use_modality_gating', 'Unknown'),
                        'adapter': getattr(config, 'use_statistics_adapter', 'Unknown'),
                    })
            except:
                pass
    
    print(f"\n找到 {len(models_found)} 个模型:")
    for m in models_found[-5:]:  # 显示最近5个
        print(f"   📁 {m['folder']}")
        print(f"      Gate: {m['gating']}, Adapter: {m['adapter']}")
    
    return models_found


if __name__ == "__main__":
    # 1. 查找最新模型
    print("🔍 查找最新训练的模型...")
    outputs = sorted([d for d in os.listdir("outputs") if d.startswith("FSMol_ADKTModel")])
    
    if outputs:
        latest = outputs[-1]
        model_path = f"outputs/{latest}/best_validation.pt"
        
        if os.path.exists(model_path):
            print(f"📂 使用模型: {latest}")
            
            # 分析门控权重
            gate_weights, task_names = analyze_gate_weights(model_path, num_tasks=30)
            
            # 快速消融检查
            quick_ablation_check()
        else:
            print(f"❌ 模型文件不存在: {model_path}")
    else:
        print("❌ 未找到训练输出目录")
