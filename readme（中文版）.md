<!--
<script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "FS-Mol",
    "description": "A Few-Shot Learning Dataset of Molecules",
    "url": "https://github.com/microsoft/FS-Mol/tree/main/datasets",
    "license": "https://creativecommons.org/licenses/by-sa/3.0/",
    "isAccessibleForFree" : true,
  }
</script>
-->

# 元学习用于分子性质预测的自适应深核高斯过程（ADKF-IFT，ICLR 2023）

<div align="center">

[![Paper](http://img.shields.io/badge/paper-arxiv.2205.02708-B31B1B.svg)](https://arxiv.org/abs/2205.02708)
[![Conference](http://img.shields.io/badge/ICLR-2023-4b44ce.svg)](https://openreview.net/forum?id=KXRSh0sdVTP)

</div>

这是论文《Meta-learning Adaptive Deep Kernel Gaussian Processes for Molecular Property Prediction》（ICLR 2023）中提出的方法 `Adaptive Deep Kernel Fitting with Implicit Function Theorem (ADKF-IFT)` 的官方 PyTorch 实现。请阅读我们的论文 [[arXiv](https://arxiv.org/abs/2205.02708), [OpenReview](https://openreview.net/forum?id=KXRSh0sdVTP)] 以获取对 ADKF-IFT 方法的详细说明。

我们在 FS-Mol 和 MoleculeNet 上实现了 ADKF-IFT（在本仓库中称为 ADKT）、DKL、DKT 和 CNP。我们将 PAR 的官方代码适配到了 FS-Mol。对于适合做回归的所有模型，我们也在 FS-Mol 上提供了回归任务的代码，相关内容位于 `fs_mol` 文件夹。

论文中 FS-Mol 基准的所有**原始结果数据**、绘图，以及生成结果图的笔记本均位于 `visualize_results` 文件夹。我们的 ADKF-IFT 模型（分类与回归）检查点可以从 [figshare](https://figshare.com/articles/online_resource/adkf-ift-weights_zip/22070105) 下载。

运行 ADKF-IFT 的 MoleculeNet 实验代码位于 `MoleculeNet` 文件夹。请按照该文件夹中的 `README.md` 指南进行配置与运行。

此外，复现四个具有代表性的域外分子设计实验（预测与贝叶斯优化）的代码位于 `bayes_opt` 文件夹。

如果您认为我们的论文、代码或原始结果数据对您的研究有帮助，请考虑引用我们的论文：

```
@inproceedings{chen2023metalearning,
  title={Meta-learning Adaptive Deep Kernel Gaussian Processes for Molecular Property Prediction},
  author={Wenlin Chen and Austin Tripp and Jos{\'e} Miguel Hern{\'a}ndez-Lobato},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=KXRSh0sdVTP}
}
```

## 免责声明

本代码库基于 [FS-Mol](https://github.com/microsoft/FS-Mol) 与 [PAR](https://github.com/tata1661/PAR-NeurIPS21) 仓库的分支。README、许可协议等文件在此基础上有所复制与修改。

# 在 FS-Mol 上进行 ADKF-IFT 的元训练/测试说明

以下命令可用于完成仓库的基本设置：

```bash
# 克隆 PAR 子模块
git submodule update --init --recursive

# 创建并激活 conda 环境
conda env create -f environment.yml
conda activate adkf-ift-fsmol

# 下载并解压数据集
wget -O fs-mol-dataset.tar https://figshare.com/ndownloader/files/31345321
tar -xf fs-mol-dataset.tar  # 将创建目录 ./fs-mol
rm fs-mol-dataset.tar # 删除 tar 文件以节省空间
mv fs-mol fs-mol-dataset # 重命名文件夹以更清晰

# 下载并解压预训练模型权重
wget -O adkf-ift-weights.zip https://figshare.com/ndownloader/files/39203102
unzip adkf-ift-weights.zip  # 将生成 2 个 .pt 文件
```

分类任务的元训练：
```bash
dataset_dir="./fs-mol-dataset"  # 按需修改
python fs_mol/adaptive_dkt_train.py "$dataset_dir"
```

回归任务的元训练：
```bash
dataset_dir="./fs-mol-dataset"  # 按需修改
python fs_mol/adaptive_dkt_train.py "$dataset_dir" --use-numeric-labels
```

元测试：

```bash
# 如果您自行训练了模型，检查点路径类似：
# "./outputs/FSMol_ADKTModel_gnn+ecfp+fc_{YYYY-MM_DD_HH-MM-SS}/best_validation.pt"
# 否则，可直接使用下方的预训练模型：
model_checkpoint="./adkf-ift-classification.pt"  # 按需修改
python fs_mol/adaptive_dkt_test.py "$model_checkpoint" "$dataset_dir"
```

分类任务的元测试结果可通过运行以下命令收集：
```bash
eval_id="YYYY-MM_DD_HH-MM-SS" # 按需修改（需先完成元测试）
python fs_mol/plotting/collect_eval_runs.py ADKT "./outputs/FSMol_Eval_ADKTModel_${eval_id}" # 按需修改
```

回归任务的元测试结果可通过运行以下命令收集：
```bash
eval_id="YYYY-MM_DD_HH-MM-SS" # 按需修改（需先完成元测试）
python fs_mol/plotting/collect_eval_runs.py ADKTNUMERIC "./outputs/FSMol_Eval_ADKTModel_${eval_id}" --metric r2 # 按需修改
```

随后可使用 `visualize_results` 文件夹中的笔记本进行结果可视化。

---
---

以下为 FS-Mol 仓库的原始 README 内容的中文翻译。

# FS-Mol：分子少样本学习数据集

本仓库包含 FS-Mol：分子少样本学习数据集 的数据与代码。

## 安装

1. 克隆或下载本仓库
2. 安装依赖

   ```
   cd FS-Mol

   conda env create -f environment.yml
   conda activate fsmol
   ```

本仓库将分子注意力 Transformer（MAT）的基线代码作为子模块添加。因此，若需运行 MAT，需通过 `git clone --recurse-submodules` 克隆本仓库。或者，您也可以先正常克隆本仓库，再通过 `git submodule update --init` 设置子模块。若未设置 MAT 子模块，本仓库的其他部分仍可正常使用。

## 数据

数据集可通过下载 [FS-Mol Data](https://figshare.com/ndownloader/files/31345321) 获取，包含 `train`、`valid` 与 `test` 三个文件夹。此外，我们在 `datasets/fsmol-0.1.json` 文件中指定了每个数据折的默认任务列表。需注意，完整数据集包含更多任务。若希望使用所有可用的训练任务，请在训练脚本中使用参数 `--task_list_file datasets/entire_train_set.json`。随着更多数据通过 ChEMBL 提供，任务列表也将用于未来迭代中的 FS-Mol 版本管理。

任务以独立的压缩 [JSONLines](https://jsonlines.org/) 文件存储，每一行对应该任务的一个数据点信息。
每个数据点以固定结构的 JSON 字典存储：
```json
{
    "SMILES": "SMILES_STRING",
    "Property": "ACTIVITY BOOL LABEL",
    "Assay_ID": "CHEMBL ID",
    "RegressionProperty": "ACTIVITY VALUE",
    "LogRegressionProperty": "LOG ACTIVITY VALUE",
    "Relation": "ASSUMED RELATION OF MEASURED VALUE TO TRUE VALUE",
    "AssayType": "TYPE OF ASSAY",
    "fingerprints": [...],
    "descriptors": [...],
    "graph": {
        "adjacency_lists": [
           [... SINGLE BONDS AS PAIRS ...],
           [... DOUBLE BONDS AS PAIRS ...],
           [... TRIPLE BONDS AS PAIRS ...]
        ],
        "node_types": [...ATOM TYPES...],
        "node_features": [...NODE FEATURES...],
    }
}
```

### FSMolDataset
`fs_mol.data.FSMolDataset` 类为 Python 提供了对该少样本数据集的训练/验证/测试任务的程序化访问。
可通过数据目录创建实例：`FSMolDataset.from_directory(/path/to/dataset)`。
关于如何使用 `FSMolDataset` 的更多细节与示例，请参阅 `fs_mol/notebooks/dataset.ipynb`。

## 评估新模型

我们在 `fs_mol.utils.eval_utils.eval_model()` 中提供了 FS-Mol 评估方法的实现。
这是与框架无关的 Python 方法，我们在 `notebooks/evaluation.ipynb` 中详细演示了如何使用它来评估新模型。

请注意，我们的基线测试脚本（`fs_mol/baseline_test.py`、`fs_mol/maml_test.py`、`fs_mol/mat_test`、`fs_mol/multitask_test.py` 和 `fs_mol/protonet_test.py`）也使用了该方法，并可作为集成示例：包括在 TensorFlow 中进行按任务微调（`maml_test.py`）、在 PyTorch 中进行微调（`mat_test.py`），以及在 scikit-learn 模型上进行单任务训练（`baseline_test.py`）。
这些脚本也支持使用 `--task_list_file` 参数来选择不同的测试任务集，按需使用。

## 基线模型实现

我们提供了三种关键的少样本学习方法的实现：多任务学习、模型无关的元学习（MAML）和原型网络（PN），以及单任务基线与分子注意力 Transformer（MAT）[论文](https://arxiv.org/abs/2002.08264v1)、[代码](https://github.com/lucidrains/molecule-attention-transformer) 的评估。

所有结果及相关绘图均位于 `baselines/` 目录。

这些基线方法在 FS-Mol 数据集上的运行方式如下：

### kNN 与随机森林 —— 单任务基线

我们的 kNN 与随机森林基线通过对业界标准参数集合进行网格搜索获得，详见脚本 `baseline_test.py`。

可如下运行单任务的基线评估，并在 kNN 与 randomForest 模型中进行选择：

```bash
python fs_mol/baseline_test.py /path/to/data --model {kNN, randomForest}
```

### 分子注意力 Transformer

分子注意力 Transformer（MAT）[论文](https://arxiv.org/abs/2002.08264v1)、[代码](https://github.com/lucidrains/molecule-attention-transformer)。

MAT 的评估方式如下：

```bash
python fs_mol/mat_test.py /path/to/pretrained-mat /path/to/data
```

### GNN-MAML 预训练与评估

GNN-MAML 模型基于数据集的分子图表示进行操作。该模型由一个 8 层的 GNN 组成，节点嵌入维度为 128。GNN 使用 “Edge-MLP” 消息传递。该模型在支持集大小为 16 的设置下，按照 MAML 过程 [Finn 2017](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf) 进行训练。检查点中使用的超参数为 `maml_train.py` 的默认设置。

当前默认设置用于训练可在此获取的 GNN-MAML 最终版本：

```bash
python fs_mol/maml_train.py /path/to/data 
```

评估运行方式：

```bash
python fs_mol/maml_test.py /path/to/data --trained_model /path/to/gnn-maml-checkpoint
```

### GNN-MT 预训练与评估

GNN-MT 模型基于数据集的分子图表示进行操作。该模型由一个 10 层的 GNN 组成，节点嵌入维度为 128。模型使用主邻域聚合（PNA）消息传递。检查点中使用的超参数为 `multitask_train.py` 的默认设置。该方法与 [Hu 2019](https://arxiv.org/abs/1905.12265v1) 中的仅任务训练方法具有相似性。

```bash
python fs_mol/multitask_train.py /path/to/data 
```

评估运行方式：

```bash
python fs_mol/multitask_test.py /path/to/gnn-mt-checkpoint /path/to/data
```

### 原型网络（PN）预训练与评估

原型网络方法 [Snell 2017](https://proceedings.neurips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf) 会提取支持集数据点的表示，并据此对正负样本进行分类。我们采用 Mahalonobis 距离作为查询点到类别原型的度量。

```bash
python fs_mol/protonet_train.py /path/to/data 
```

评估运行方式：

```bash
python fs_mol/protonet_test.py /path/to/pn-checkpoint /path/to/data
```

## 可用的模型检查点

我们为 `GNN-MAML`、`GNN-MT` 与 `PN` 提供了预训练模型，可通过 [figshare](https://figshare.com/projects/FS-Mol_Dataset_and_Models/125797) 的链接下载。

| 模型名称 | 描述 | 检查点文件 |
| ---------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| GNN-MAML   | 支持集大小 16。8 层 GNN。Edge MLP 消息传递。              | [MAML-Support16_best_validation.pkl](https://figshare.com/ndownloader/files/31346701) |
| GNN-MT     | 10 层 GNN。PNA 消息传递                                        | [multitask_best_model.pt](https://figshare.com/ndownloader/files/31338334)              |
| PN         | 10 层 GGN，PNA 消息传递。ECFP+GNN，Mahalonobis 距离度量 | [PN-Support64_best_validation.pt](https://figshare.com/ndownloader/files/31307479)    |


## 定义、训练与评估新的模型实现

`fs_mol` 中的一系列训练与测试脚本展示了如何灵活地定义少样本模型与单任务模型。

我们在 `notebooks/integrating_torch_models.ipynb` 中给出了如何使用抽象类 `AbstractTorchFSMolModel` 集成一个新的通用 PyTorch 模型的详细示例，并指出下述评估过程已在 `fs_mol/baseline_test.py`（sklearn 模型）与 `fs_mol/maml_test.py`（基于 TensorFlow 的 GNN 模型）中演示。

## 贡献

本项目欢迎贡献与建议。大多数贡献需要您同意一份贡献者许可协议（CLA），声明您拥有并确实授予我们使用您的贡献的权利。详情请访问：https://cla.opensource.microsoft.com。

当您提交拉取请求（PR）时，CLA 机器人会自动判断您是否需要提供 CLA，并对 PR 进行相应的标注（例如状态检查、评论）。请按机器人提供的说明操作。对于使用我们 CLA 的所有仓库，您只需执行一次。

本项目采用了 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。
更多信息请参阅 [行为准则常见问题](https://opensource.microsoft.com/codeofconduct/faq/)，或通过邮件联系 [opencode@microsoft.com](mailto:opencode@microsoft.com) 获取更多问题或意见的支持。

## 商标

本项目可能包含用于项目、产品或服务的商标或标识。对 Microsoft 商标或标识的授权使用需遵循并符合 [Microsoft 的商标与品牌指南](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)。
对本项目的修改版本中使用 Microsoft 的商标或标识，不得造成混淆或暗示 Microsoft 的赞助。
任何第三方商标或标识的使用需遵循相应第三方的政策。