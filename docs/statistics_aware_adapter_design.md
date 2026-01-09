# Statistics-Aware Adapter 设计文档

## 1. 概述

**Statistics-Aware Adapter** 是一个轻量级的可学习预处理模块，专门用于 ECFP 分子指纹和 Physicochemical Descriptors (PC-descs) 这两种"固定来源"的特征。

### 设计目标
- **降噪**：通过可学习的尺度调整减少异常值的影响
- **尺度对齐**：使不同来源的特征具有可比较的数值范围
- **提升 Gate 可靠性**：让 Modality Gate 接收到更稳定的输入统计量

### 非目标
- ❌ 不是为了提升表达能力
- ❌ 不是端到端重学特征
- ❌ 不引入复杂的 attention 或 transformer 结构


## 2. 模块架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Statistics-Aware Adapter                          │
│                                                                     │
│   输入特征 x [N, D]                                                  │
│        │                                                            │
│        ├──────────────────────────────────────────┐                 │
│        │                                          │                 │
│        ▼                                          ▼                 │
│   ┌─────────────────┐                     ┌─────────────────┐       │
│   │ compute_stats   │                     │    直接传递      │       │
│   │                 │                     │                 │       │
│   │ - batch_mean    │                     │                 │       │
│   │ - batch_std     │                     │                 │       │
│   │ - col_mean_mean │                     │                 │       │
│   │ - col_mean_std  │                     │                 │       │
│   │ - l2_mean       │                     │                 │       │
│   │ - l2_std        │                     │                 │       │
│   └────────┬────────┘                     │                 │       │
│            │ [6]                          │                 │       │
│            ▼                              │                 │       │
│   ┌─────────────────┐                     │                 │       │
│   │   Tiny MLP      │                     │                 │       │
│   │  6 → 4 → 2      │                     │                 │       │
│   │ (delta_scale,   │                     │                 │       │
│   │  delta_shift)   │                     │                 │       │
│   └────────┬────────┘                     │                 │       │
│            │                              │                 │       │
│            ▼                              │                 │       │
│   ┌─────────────────┐                     │                 │       │
│   │ scale = softplus│                     │                 │       │
│   │   (base_scale   │                     │                 │       │
│   │  + delta_scale) │                     │                 │       │
│   │                 │                     │                 │       │
│   │ shift = base +  │                     │                 │       │
│   │        delta    │                     │                 │       │
│   └────────┬────────┘                     │                 │       │
│            │                              │                 │       │
│            └──────────────┬───────────────┘                 │       │
│                           │                                 │       │
│                           ▼                                 │       │
│                    x * scale + shift                        │       │
│                           │                                 │       │
│                           ▼                                 │       │
│                   输出特征 [N, D]                            │       │
└─────────────────────────────────────────────────────────────────────┘
```


## 3. 输入统计量

| 统计量 | 含义 | 作用 |
|--------|------|------|
| `batch_mean` | 所有元素的全局均值 | 反映特征的整体偏移 |
| `batch_std` | 所有元素的全局标准差 | 反映特征的整体离散程度 |
| `col_mean_mean` | 各列均值的均值 | 反映各维度的平均分布 |
| `col_mean_std` | 各列均值的标准差 | 反映各维度的不均衡程度 |
| `l2_mean` | 样本 L2 范数的均值 | 反映样本的平均"能量" |
| `l2_std` | 样本 L2 范数的标准差 | 反映样本间的能量差异 |


## 4. 与 Modality Gate 的关系

```
原始特征
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Statistics-Aware Adapter                                   │
│  • 决定 "怎么用" 这个模态                                    │
│  • 输出：尺度调整后的特征                                    │
│  • 作用：降噪、对齐、稳定化                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Modality Gate                                              │
│  • 决定 "用不用" 这个模态                                    │
│  • 输出：0-1 权重                                           │
│  • 作用：模态选择、重要性加权                                │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
融合后的特征 → FC → GP
```

**互补性分析**：
- Gate 是离散的"开关"概念（虽然是连续值，但语义是权重）
- Adapter 是连续的"调整"概念（尺度变换）
- Gate 作用于整个模态；Adapter 作用于特征值本身


## 5. Inductive Bias

1. **Batch-Adaptive Calibration**
   - 同一 batch 内的样本共享相同的 scale/shift
   - 这与 few-shot 设置一致：support set 定义任务特定的尺度

2. **Statistics-Driven Transformation**
   - 变换由统计量驱动，而非原始特征值
   - 这避免了过拟合到具体的特征模式

3. **Identity Initialization**
   - 初始化接近恒等变换
   - 模型需要"学习到"偏离恒等的必要性

4. **Global Scale Assumption**
   - 假设同一个特征的所有维度需要相似的尺度调整
   - 这比 per-dimension 调整（参数量 O(D)）更加正则化


## 6. 参数量分析

```
StatisticsAwareAdapter:
├── adapter (Tiny MLP):
│   ├── Linear(6 → 4): 6×4 + 4 = 28 params
│   └── Linear(4 → 2): 4×2 + 2 = 10 params
├── base_scale: 1 param
└── base_shift: 1 param
────────────────────────────────────────
Total: 40 parameters per adapter

如果使用 ECFP + PC-descs:
2 × 40 = 80 parameters
```

对比：
- Modality Gate: ~83 params
- 两个 Adapters: ~80 params
- 总增加: ~160 params（远小于 1k 的约束）


## 7. 为什么适合 Few-Shot + GP

1. **参数量极小**
   - 减少过拟合风险
   - 训练效率高

2. **统计量驱动**
   - 利用 batch 级信息而非样本级
   - 与 few-shot 的"少样本学全局"思想一致

3. **改善 GP 输入**
   - GP 的核函数对特征尺度敏感
   - 适配后的特征更稳定，减少 Cholesky 分解失败

4. **可解释**
   - scale 和 shift 有明确的物理含义
   - 便于分析和调试


## 8. 使用方式

```python
# 在 config 中启用
config = ADKTModelConfig(
    used_features="gnn+ecfp+pc-descs+fc",
    use_modality_gating=True,       # 启用门控
    use_statistics_adapter=True,    # 启用适配器
    adapter_hidden_dim=4,           # 适配器隐藏维度
)

# 模型会自动：
# 1. 为 ECFP 创建 ecfp_adapter
# 2. 为 PC-descs 创建 pc_adapter
# 3. 在 forward 中自动应用
```


## 9. 实验建议

### 消融实验设计
1. **Baseline**: 无 Adapter，无 Gate
2. **Gate Only**: 有 Gate，无 Adapter
3. **Adapter Only**: 有 Adapter，无 Gate
4. **Full Model**: 有 Gate + Adapter

### 观察指标
- 验证集 Avg. Precision
- Gate 权重分布
- Adapter 的 scale/shift 学习曲线
- GP 数值稳定性（Cholesky 警告次数）


## 10. 潜在改进方向

1. **Per-Modality Statistics**
   - 当前：所有模态共享同一种统计量计算方式
   - 改进：为不同模态设计不同的统计量

2. **Adaptive Hidden Dim**
   - 当前：固定 hidden_dim=4
   - 改进：根据特征维度自适应

3. **Regularization**
   - 添加 scale 的 L2 正则化，防止过度缩放
