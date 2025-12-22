# Windows 下运行 ADKF-IFT 项目 - 快速开始指南

## ✅ 你已经完成的步骤

- [x] 安装 WSL2
- [x] 安装 Ubuntu

## 🚀 接下来的步骤（两种方式任选其一）

---

### 方式 1：使用自动化脚本（推荐，最简单）

#### 步骤 1：进入 Ubuntu 环境

在 PowerShell 中输入：
```powershell
wsl
```

或者从开始菜单打开 "Ubuntu" 应用。

#### 步骤 2：运行自动配置脚本

在 Ubuntu 终端中依次输入：

```bash
# 进入项目目录
cd /mnt/c/Users/rachel/Desktop/ADKF-IFT-main

# 给脚本添加执行权限
chmod +x setup_wsl.sh

# 运行配置脚本
./setup_wsl.sh
```

**说明**：
- 脚本会自动完成所有配置（更新系统、安装工具、下载数据等）
- 整个过程需要 30-60 分钟（主要是下载时间）
- 你可以去喝杯咖啡，等待完成 ☕

#### 步骤 3：激活环境并测试

配置完成后，运行：

```bash
# 重新加载配置
source ~/.bashrc

# 激活 conda 环境
conda activate adkf-ift-fsmol

# 进入项目目录
cd ~/ADKF-IFT-main

# 测试预训练模型
python fs_mol/adaptive_dkt_test.py ./adkf-ift-classification.pt ./fs-mol-dataset
```

---

### 方式 2：手动逐步配置（更可控）

如果你想了解每一步在做什么，可以手动执行：

#### 1. 进入 Ubuntu
```powershell
wsl
```

#### 2. 更新系统
```bash
sudo apt update && sudo apt upgrade -y
```

#### 3. 安装工具
```bash
sudo apt install -y wget git unzip build-essential
```

#### 4. 安装 Miniconda
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

#### 5. 复制项目
```bash
cp -r /mnt/c/Users/rachel/Desktop/ADKF-IFT-main ~/
cd ~/ADKF-IFT-main
```

#### 6. 初始化子模块
```bash
git submodule update --init --recursive
```

#### 7. 创建 Conda 环境
```bash
conda env create -f environment.yml
conda activate adkf-ift-fsmol
```

#### 8. 下载数据集
```bash
wget -O fs-mol-dataset.tar https://figshare.com/ndownloader/files/31345321
tar -xf fs-mol-dataset.tar
mv fs-mol fs-mol-dataset
rm fs-mol-dataset.tar
```

#### 9. 下载模型权重
```bash
wget -O adkf-ift-weights.zip https://figshare.com/ndownloader/files/39203102
unzip adkf-ift-weights.zip
rm adkf-ift-weights.zip
```

#### 10. 测试运行
```bash
python fs_mol/adaptive_dkt_test.py ./adkf-ift-classification.pt ./fs-mol-dataset
```

---

## 📝 常见问题

### Q1: 如何在 PowerShell 和 Ubuntu 之间切换？

- **进入 Ubuntu**：在 PowerShell 中输入 `wsl`
- **退出 Ubuntu**：在 Ubuntu 终端中输入 `exit`

### Q2: 如何访问 Windows 文件？

在 Ubuntu 中，Windows 的 C 盘路径是：`/mnt/c/`

例如：
- `C:\Users\rachel\Desktop` → `/mnt/c/Users/rachel/Desktop`

### Q3: 如何访问 WSL2 中的文件？

在 Windows 文件资源管理器地址栏输入：
```
\\wsl$\Ubuntu\home\你的用户名\ADKF-IFT-main
```

### Q4: 下载速度太慢怎么办？

可以在 Windows 浏览器中下载文件，然后复制到 WSL2：

1. 在浏览器中下载：
   - 数据集: https://figshare.com/ndownloader/files/31345321
   - 权重: https://figshare.com/ndownloader/files/39203102

2. 在 Ubuntu 中复制：
   ```bash
   cp /mnt/c/Users/rachel/Downloads/文件名 ~/ADKF-IFT-main/
   ```

### Q5: 如何查看配置进度？

脚本运行时会显示进度，例如：
```
[1/8] 更新系统包...
[2/8] 安装必要工具...
...
```

---

## 🎯 推荐使用方式 1（自动化脚本）

对于新手来说，使用自动化脚本是最简单的方式：

1. 在 PowerShell 中输入 `wsl` 进入 Ubuntu
2. 运行 `cd /mnt/c/Users/rachel/Desktop/ADKF-IFT-main`
3. 运行 `chmod +x setup_wsl.sh && ./setup_wsl.sh`
4. 等待完成即可！

---

## 📞 需要帮助？

如果遇到任何问题，可以：
1. 查看详细指南：`wsl2_setup_guide.md`
2. 查看项目 README：`readme（中文版）.md`
3. 向我提问！
