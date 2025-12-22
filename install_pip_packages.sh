#!/bin/bash
# 完成 pip 包安装的脚本（增加超时时间和重试）

set -e

echo "=========================================="
echo "安装剩余的 pip 包"
echo "=========================================="
echo ""

# 激活环境
source ~/.bashrc
conda activate adkf-ift-fsmol

# 增加 pip 超时时间
export PIP_DEFAULT_TIMEOUT=300

echo "[1/4] 安装基础 pip 包..."
pip install --timeout=300 --retries=5 botorch gpytorch docopt "dpu-utils>=0.2.13"

echo ""
echo "[2/4] 安装 TensorFlow 和 tf2-gnn..."
pip install --timeout=300 --retries=5 "tensorflow>=2.4,<2.12" "tf2-gnn~=2.12.0"

echo ""
echo "[3/4] 安装其他依赖..."
pip install --timeout=300 --retries=5 more-itertools mysql-connector-python==8.0.17 pyprojroot "py-repo-root~=1.1.1" xlsxwriter autorank openpyxl

echo ""
echo "[4/4] 安装 torch-scatter（使用预编译版本）..."
pip install --timeout=300 --retries=5 torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

echo ""
echo "=========================================="
echo "✅ 所有包安装完成！"
echo "=========================================="
echo ""

# 验证安装
echo "验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_scatter; print('torch-scatter: OK')"
python -c "import rdkit; print('rdkit: OK')"
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"

echo ""
echo "环境配置完成！现在可以下载数据集和模型了。"
