# setup_windows.ps1
# Setup script for ADKF-IFT on Windows
# Usage: Right-click and "Run with PowerShell" or run `.\setup_windows.ps1` in terminal

Write-Host "Starting ADKF-IFT Windows Setup..." -ForegroundColor Green

# 1. Create Conda Environment
$EnvName = "adkf-windows"
Write-Host "Creating Conda environment: $EnvName (Python 3.7)..."
conda create -n $EnvName python=3.7 -y

# 2. Install Core Dependencies (PyTorch, RDKit) via Conda
# Note: Using specific PyTorch 1.10.0 with CUDA 11.3 compatible with Windows
Write-Host "Installing PyTorch 1.10.0 (CUDA 11.3) and RDKit..."
conda run -n $EnvName conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -y
conda run -n $EnvName conda install rdkit==2020.09.1.0 -c rdkit -y
conda run -n $EnvName conda install numpy=1.19.2 scikit-learn pandas seaborn tqdm typing-extensions matplotlib -y

# 3. Install Pip Dependencies
Write-Host "Installing pip dependencies..."
# Install specific torch-scatter for Windows/CUDA 11.3
# Using the official PyG wheel
conda run -n $EnvName pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0%2Bcu113.html

# Install other packages
conda run -n $EnvName pip install botorch gpytorch docopt "dpu-utils>=0.2.13" "tensorflow>=2.4" "tf2-gnn~=2.12.0" more-itertools mysql-connector-python==8.0.17 pyprojroot "py-repo-root~=1.1.1" xlsxwriter autorank azureml-core openpyxl

Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "To run the training:"
Write-Host "1. conda activate $EnvName"
Write-Host "2. python fs_mol/adaptive_dkt_train.py ./fs-mol-dataset --features 'gnn+ecfp+pc-descs+fc' --save-dir outputs/windows_run"
