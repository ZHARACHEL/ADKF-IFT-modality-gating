#!/bin/bash
# ADKF-IFT 项目 WSL2 自动配置脚本

set -e  # 遇到错误立即停止

echo "=========================================="
echo "ADKF-IFT 项目 WSL2 环境配置脚本"
echo "=========================================="
echo ""

# 步骤 1: 更新系统
echo "[1/8] 更新系统包..."
sudo apt update && sudo apt upgrade -y

# 步骤 2: 安装必要工具
echo ""
echo "[2/8] 安装必要工具 (wget, git, unzip, build-essential)..."
sudo apt install -y wget git unzip build-essential

# 步骤 3: 安装 Miniconda
echo ""
echo "[3/8] 安装 Miniconda..."
cd ~
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    echo "Miniconda 安装完成"
else
    echo "Miniconda 已存在，跳过安装"
fi

# 初始化 conda
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 步骤 4: 复制项目到 WSL2
echo ""
echo "[4/8] 复制项目到 WSL2..."
if [ ! -d "$HOME/ADKF-IFT-main" ]; then
    cp -r /mnt/c/Users/rachel/Desktop/ADKF-IFT-main ~/
    echo "项目已复制到 ~/ADKF-IFT-main"
else
    echo "项目目录已存在，跳过复制"
fi

cd ~/ADKF-IFT-main

# 步骤 5: 初始化 Git 子模块
echo ""
echo "[5/8] 初始化 Git 子模块..."
git submodule update --init --recursive

# 步骤 6: 创建 Conda 环境
echo ""
echo "[6/8] 创建 Conda 环境 (这可能需要 10-20 分钟)..."
if ! conda env list | grep -q "adkf-ift-fsmol"; then
    ~/miniconda3/bin/conda env create -f environment.yml
    echo "Conda 环境创建完成"
else
    echo "Conda 环境已存在，跳过创建"
fi

# 步骤 7: 下载数据集
echo ""
echo "[7/8] 下载 FS-Mol 数据集 (约 1.5 GB)..."
if [ ! -d "$HOME/ADKF-IFT-main/fs-mol-dataset" ]; then
    wget -O fs-mol-dataset.tar https://figshare.com/ndownloader/files/31345321
    tar -xf fs-mol-dataset.tar
    mv fs-mol fs-mol-dataset
    rm fs-mol-dataset.tar
    echo "数据集下载完成"
else
    echo "数据集已存在，跳过下载"
fi

# 步骤 8: 下载预训练权重
echo ""
echo "[8/8] 下载预训练模型权重..."
if [ ! -f "$HOME/ADKF-IFT-main/adkf-ift-classification.pt" ]; then
    wget -O adkf-ift-weights.zip https://figshare.com/ndownloader/files/39203102
    unzip adkf-ift-weights.zip
    rm adkf-ift-weights.zip
    echo "模型权重下载完成"
else
    echo "模型权重已存在，跳过下载"
fi

echo ""
echo "=========================================="
echo "✅ 配置完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "1. 运行以下命令激活环境："
echo "   source ~/.bashrc"
echo "   conda activate adkf-ift-fsmol"
echo ""
echo "2. 测试预训练模型："
echo "   cd ~/ADKF-IFT-main"
echo "   python fs_mol/adaptive_dkt_test.py ./adkf-ift-classification.pt ./fs-mol-dataset"
echo ""
