"""
汇总所有测试结果，计算平均 AUROC
"""
import os
import pandas as pd
import numpy as np

results_dir = "outputs/FSMol_Eval_ADKTModel_2025-12-29_11-37-40"

all_roc_auc = []
all_avg_prec = []

for f in os.listdir(results_dir):
    if f.endswith("_eval_results.csv"):
        try:
            df = pd.read_csv(os.path.join(results_dir, f))
            if 'roc_auc' in df.columns:
                all_roc_auc.extend(df['roc_auc'].dropna().tolist())
            if 'avg_prec' in df.columns:
                all_avg_prec.extend(df['avg_prec'].dropna().tolist())
        except Exception as e:
            print(f"Error reading {f}: {e}")

print("=" * 60)
print("测试结果汇总")
print("=" * 60)
print(f"\n总样本数: {len(all_roc_auc)}")
print(f"\n📊 AUROC:")
print(f"   均值: {np.mean(all_roc_auc):.4f}")
print(f"   标准差: {np.std(all_roc_auc):.4f}")
print(f"   中位数: {np.median(all_roc_auc):.4f}")
print(f"\n📊 Avg Precision:")
print(f"   均值: {np.mean(all_avg_prec):.4f}")
print(f"   标准差: {np.std(all_avg_prec):.4f}")
print(f"   中位数: {np.median(all_avg_prec):.4f}")
print("=" * 60)
