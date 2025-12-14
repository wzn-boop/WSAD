import torch

# 检查CUDA是否可用
print(torch.cuda.is_available())
print(torch.__version__)
# 创建一个随机张量并将其移动到GPU上
tensor = torch.rand((3, 3), device='cuda')
print(tensor)