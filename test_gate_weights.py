"""快速测试门控权重是否正常输出"""
import torch
import sys
from pyprojroot import here as project_root
sys.path.insert(0, str(project_root()))

from fs_mol.utils.adaptive_dkt_utils import ADKTModelTrainer
from fs_mol.data import FSMolDataset, DataFold
from fs_mol.data.dkt import get_dkt_task_sample_iterable
from fs_mol.utils.torch_utils import torchify

print("=" * 60)
print("🔍 测试门控权重是否正常工作")
print("=" * 60)

# 加载刚训练的模型
model_path = r".\outputs\FSMol_ADKTModel_gnn+ecfp+pc-descs+fc_2025-12-27_19-06-08\best_validation.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n📂 加载模型: {model_path}")
model = ADKTModelTrainer.build_from_model_file(model_path, device=device)
model = model.to(device)

# 检查门控模块
print(f"\n✅ 检查门控模块:")
print(f"   use_modality_gating: {model.config.use_modality_gating}")
print(f"   used_features: {model.config.used_features}")

if hasattr(model, 'modality_gate'):
    print(f"   ✓ 存在 modality_gate 模块")
    print(f"   门控网络结构: {model.modality_gate}")
else:
    print(f"   ✗ 不存在 modality_gate 模块!")
    exit(1)

# 加载一个任务进行测试
print(f"\n🧪 加载测试数据...")
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

# 测试3个任务
print(f"\n📊 测试门控权重输出:")
for i in range(3):
    try:
        task_sample = next(task_iter)
        task_sample = torchify(task_sample, device=device)
        batch = task_sample.batches[0]
        
        model.eval()
        with torch.no_grad():
            _ = model(batch, train_loss=None)
        
        # 检查 last_gate_weights
        if hasattr(model, 'last_gate_weights') and model.last_gate_weights:
            print(f"\n   任务 {i+1} ({task_sample.task_name}):")
            for modality, weight in model.last_gate_weights.items():
                print(f"      {modality}: {weight.item():.4f}")
        else:
            print(f"\n   任务 {i+1}: ⚠️ last_gate_weights 为空!")
            print(f"   has attr: {hasattr(model, 'last_gate_weights')}")
            if hasattr(model, 'last_gate_weights'):
                print(f"   value: {model.last_gate_weights}")
    except StopIteration:
        break

print("\n" + "=" * 60)
print("✅ 测试完成")
print("=" * 60)
