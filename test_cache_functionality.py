#!/usr/bin/env python
"""
测试 FSMolDataset 缓存功能

此脚本验证：
1. 缓存文件的创建和加载
2. 缓存加速效果
3. 数据一致性
"""

import os
import time
import sys

# 确保可以导入 fs_mol
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fs_mol.data import FSMolDataset, DataFold

def main():
    print("=" * 60)
    print(" FSMolDataset 缓存功能测试")
    print("=" * 60)
    print()
    
    # 配置
    dataset_path = "./fs-mol-dataset"
    cache_file = "test_dataset_cache.pkl"
    
    # 检查数据集是否存在
    if not os.path.exists(dataset_path):
        print(f"[错误] 数据集目录不存在: {dataset_path}")
        print()
        print("请先解压数据集：")
        print("  - 在 Windows: .\\extract_fsmol_windows.ps1")
        print("  - 在 WSL/Linux: tar -xf fsmol.tar && mv fs-mol fs-mol-dataset")
        return 1
    
    # 清理旧缓存
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"[清理] 删除旧缓存文件: {cache_file}")
        print()
    
    # 测试1：首次加载（无缓存）
    print("-" * 60)
    print(" 测试 1: 首次加载（无缓存）")
    print("-" * 60)
    print()
    
    start = time.time()
    dataset1 = FSMolDataset.from_directory(
        dataset_path,
        cache_path=cache_file
    )
    time1 = time.time() - start
    
    print()
    print(f"[结果] 耗时: {time1:.2f} 秒")
    print(f"[结果] 训练集任务数: {dataset1.get_num_fold_tasks(DataFold.TRAIN)}")
    print(f"[结果] 验证集任务数: {dataset1.get_num_fold_tasks(DataFold.VALIDATION)}")
    print(f"[结果] 测试集任务数: {dataset1.get_num_fold_tasks(DataFold.TEST)}")
    print()
    
    # 验证缓存文件是否被创建
    if os.path.exists(cache_file):
        cache_size = os.path.getsize(cache_file) / 1024  # KB
        print(f"[✓] 缓存文件已创建: {cache_file} ({cache_size:.2f} KB)")
    else:
        print(f"[✗] 缓存文件未创建！")
        return 1
    print()
    
    # 测试2：第二次加载（有缓存）
    print("-" * 60)
    print(" 测试 2: 第二次加载（从缓存）")
    print("-" * 60)
    print()
    
    start = time.time()
    dataset2 = FSMolDataset.from_directory(
        dataset_path,
        cache_path=cache_file
    )
    time2 = time.time() - start
    
    print()
    print(f"[结果] 耗时: {time2:.2f} 秒")
    print(f"[结果] 训练集任务数: {dataset2.get_num_fold_tasks(DataFold.TRAIN)}")
    print(f"[结果] 验证集任务数: {dataset2.get_num_fold_tasks(DataFold.VALIDATION)}")
    print(f"[结果] 测试集任务数: {dataset2.get_num_fold_tasks(DataFold.TEST)}")
    print()
    
    # 计算加速比
    speedup = time1 / time2 if time2 > 0 else 0
    print(f"[性能] 加速比: {speedup:.2f}x")
    print()
    
    # 测试3：验证数据一致性
    print("-" * 60)
    print(" 测试 3: 数据一致性验证")
    print("-" * 60)
    print()
    
    checks = [
        ("训练集任务数", 
         dataset1.get_num_fold_tasks(DataFold.TRAIN),
         dataset2.get_num_fold_tasks(DataFold.TRAIN)),
        ("验证集任务数",
         dataset1.get_num_fold_tasks(DataFold.VALIDATION),
         dataset2.get_num_fold_tasks(DataFold.VALIDATION)),
        ("测试集任务数",
         dataset1.get_num_fold_tasks(DataFold.TEST),
         dataset2.get_num_fold_tasks(DataFold.TEST)),
    ]
    
    all_passed = True
    for name, val1, val2 in checks:
        if val1 == val2:
            print(f"[✓] {name}: {val1} == {val2}")
        else:
            print(f"[✗] {name}: {val1} != {val2}")
            all_passed = False
    
    print()
    
    # 测试4：force_reload 参数
    print("-" * 60)
    print(" 测试 4: force_reload 参数")
    print("-" * 60)
    print()
    
    print("使用 force_reload=True 重新加载...")
    start = time.time()
    dataset3 = FSMolDataset.from_directory(
        dataset_path,
        cache_path=cache_file,
        force_reload=True
    )
    time3 = time.time() - start
    
    print()
    print(f"[结果] 耗时: {time3:.2f} 秒")
    print(f"[结果] 应该接近首次加载时间: {time1:.2f} 秒")
    print()
    
    # 清理
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"[清理] 删除测试缓存文件: {cache_file}")
    
    # 总结
    print()
    print("=" * 60)
    print(" 测试总结")
    print("=" * 60)
    print()
    
    if all_passed and speedup > 2.0:
        print("[✓] 所有测试通过！")
        print(f"[✓] 缓存功能正常，加速比: {speedup:.2f}x")
        print()
        return 0
    else:
        if not all_passed:
            print("[✗] 数据一致性检查失败！")
        if speedup <= 2.0:
            print(f"[警告] 加速比较低: {speedup:.2f}x（预期 > 2.0x）")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
