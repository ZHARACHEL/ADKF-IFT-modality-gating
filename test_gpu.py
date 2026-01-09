import torch# 若不报错，且能输出版本号，则说明安装成功
print("PyTorch 版本：", torch.__version__)
# 查看CUDA是否可用（有显卡则输出True，无显卡则输出False）
print("CUDA 可用：", torch.cuda.is_available())