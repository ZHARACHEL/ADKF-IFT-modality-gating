"""
FS-Mol 测试结果完整报告
"""
import os
import pandas as pd
import numpy as np

results_dir = "outputs/FSMol_Eval_ADKTModel_2025-12-29_11-37-40"

print("=" * 70)
print("🎯 FS-Mol 测试结果报告")
print("=" * 70)

# 收集所有结果
all_results = []

for f in os.listdir(results_dir):
    if f.endswith("_eval_results.csv"):
        task_name = f.replace("_eval_results.csv", "")
        try:
            df = pd.read_csv(os.path.join(results_dir, f))
            for _, row in df.iterrows():
                all_results.append({
                    'task': task_name,
                    'support_size': row.get('num_train', row.get('num_train_requested', None)),
                    'roc_auc': row.get('roc_auc', None),
                    'avg_prec': row.get('avg_prec', None),
                    'acc': row.get('acc', None),
                    'f1': row.get('f1', None),
                })
        except Exception as e:
            pass

df_all = pd.DataFrame(all_results)

print(f"\n📊 总体统计")
print("-" * 50)
print(f"任务数: {df_all['task'].nunique()}")
print(f"总样本数: {len(df_all)}")

print(f"\n📈 各指标汇总")
print("-" * 50)
metrics = ['roc_auc', 'avg_prec', 'acc', 'f1']
for m in metrics:
    if m in df_all.columns:
        values = df_all[m].dropna()
        if len(values) > 0:
            print(f"\n{m.upper()}:")
            print(f"   均值: {values.mean():.4f}")
            print(f"   标准差: {values.std():.4f}")
            print(f"   中位数: {values.median():.4f}")
            print(f"   最小值: {values.min():.4f}")
            print(f"   最大值: {values.max():.4f}")

# 按 support size 分组
print(f"\n📊 按 Support Size 分组统计")
print("-" * 50)
if 'support_size' in df_all.columns:
    for size in sorted(df_all['support_size'].unique()):
        subset = df_all[df_all['support_size'] == size]
        roc = subset['roc_auc'].dropna()
        ap = subset['avg_prec'].dropna()
        if len(roc) > 0:
            print(f"\nSupport Size = {int(size) if pd.notna(size) else 'N/A'}:")
            print(f"   AUROC: {roc.mean():.4f} ± {roc.std():.4f}")
            print(f"   Avg Prec: {ap.mean():.4f} ± {ap.std():.4f}")

print("\n" + "=" * 70)
print("✅ 报告生成完成")
print("=" * 70)
