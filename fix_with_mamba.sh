#!/bin/bash
# 使用 Mamba 快速解决依赖冲突的脚本

set -e

echo "=========================================="
echo "使用 Mamba 修复 ADKF-IFT 环境"
echo "=========================================="
echo ""

# 步骤 1: 安装 Mamba（更快的 conda 替代品）
echo "[1/4] 安装 Mamba 求解器..."
~/miniconda3/bin/conda install -n base -c conda-forge mamba -y

# 步骤 2: 删除旧环境
echo ""
echo "[2/4] 清理旧环境..."
if ~/miniconda3/bin/conda env list | grep -q "adkf-ift-fsmol"; then
    echo "删除旧的 adkf-ift-fsmol 环境..."
    ~/miniconda3/bin/conda env remove -n adkf-ift-fsmol -y
fi

# 步骤 3: 使用 Mamba 创建环境
echo ""
echo "[3/4] 使用 Mamba 创建环境（更快更稳定）..."
~/miniconda3/bin/mamba env create -f /mnt/c/Users/rachel/Desktop/ADKF-IFT-main/environment.yml

# 步骤 4: 修复 torch-scatter 安装
echo ""
echo "[4/4] 修复 torch-scatter 安装..."
echo "删除旧的 torch-scatter..."
~/miniconda3/bin/conda run -n adkf-ift-fsmol pip uninstall torch-scatter -y || true

echo "安装预编译的 torch-scatter..."
~/miniconda3/bin/conda run -n adkf-ift-fsmol pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

echo ""
echo "=========================================="
echo "✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "验证安装："
~/miniconda3/bin/conda run -n adkf-ift-fsmol python -c "import torch; print(f'PyTorch: {torch.__version__}')"
~/miniconda3/bin/conda run -n adkf-ift-fsmol python -c "import torch_scatter; print('torch-scatter: OK')"
~/miniconda3/bin/conda run -n adkf-ift-fsmol python -c "import rdkit; print('rdkit: OK')"

echo ""
echo "下一步："
echo "1. 激活环境："
echo "   source ~/.bashrc"
echo "   conda activate adkf-ift-fsmol"
echo ""
