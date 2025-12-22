#!/bin/bash
# 修复 ADKF-IFT Conda 环境的脚本

set -e

echo "=========================================="
echo "修复 ADKF-IFT Conda 环境"
echo "=========================================="
echo ""

# 删除已有的环境（如果存在）
echo "[1/3] 清理旧环境..."
if ~/miniconda3/bin/conda env list | grep -q "adkf-ift-fsmol"; then
    echo "删除旧的 adkf-ift-fsmol 环境..."
    ~/miniconda3/bin/conda env remove -n adkf-ift-fsmol -y
fi

# 创建基础环境
echo ""
echo "[2/3] 创建基础 Python 环境..."
~/miniconda3/bin/conda create -n adkf-ift-fsmol python=3.7.10 -y

# 激活环境并安装依赖
echo ""
echo "[3/3] 安装依赖包（分步安装以避免冲突）..."

# 使用 conda run 来在环境中执行命令
echo "  - 安装 rdkit..."
~/miniconda3/bin/conda install -n adkf-ift-fsmol -c conda-forge rdkit=2020.09.1.0 -y

echo "  - 安装基础科学计算包..."
~/miniconda3/bin/conda install -n adkf-ift-fsmol numpy=1.19.2 matplotlib scikit-learn pandas seaborn tqdm typing-extensions -y

echo "  - 安装 PyTorch (CUDA 11.3)..."
~/miniconda3/bin/conda install -n adkf-ift-fsmol -c pytorch pytorch=1.10.0 cudatoolkit=11.3 -y

echo "  - 安装 pip 包..."
~/miniconda3/bin/conda run -n adkf-ift-fsmol pip install botorch gpytorch docopt "dpu-utils>=0.2.13"

echo "  - 安装 TensorFlow..."
~/miniconda3/bin/conda run -n adkf-ift-fsmol pip install "tensorflow>=2.4,<2.12"

echo "  - 安装 tf2-gnn..."
~/miniconda3/bin/conda run -n adkf-ift-fsmol pip install "tf2-gnn~=2.12.0"

echo "  - 安装其他依赖..."
~/miniconda3/bin/conda run -n adkf-ift-fsmol pip install more-itertools mysql-connector-python==8.0.17 pyprojroot "py-repo-root~=1.1.1" xlsxwriter autorank openpyxl

echo "  - 安装 torch-scatter (使用预编译版本)..."
~/miniconda3/bin/conda run -n adkf-ift-fsmol pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

echo ""
echo "=========================================="
echo "✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 激活环境："
echo "   source ~/.bashrc"
echo "   conda activate adkf-ift-fsmol"
echo ""
echo "2. 验证安装："
echo "   python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\")'"
echo "   python -c 'import torch_scatter; print(\"torch-scatter 安装成功\")'"
echo ""
