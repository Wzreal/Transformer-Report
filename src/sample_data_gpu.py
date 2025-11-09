import torch
import random

# 固定随机种子，保证样本可复现
random.seed(42)
torch.manual_seed(42)

# 数据集路径（替换为你的实际路径，确保已生成train_en-de.pt和validation_en-de.pt）
train_data_path = "../data/processed/train_en-de.pt"
val_data_path = "../data/processed/validation_en-de.pt"

# 加载数据
print("加载数据集...")
train_data = torch.load(train_data_path)
val_data = torch.load(val_data_path)
print(f"原始数据：训练集{len(train_data)}样本，验证集{len(val_data)}样本")

# 抽样：训练集10%，验证集30%
train_sample_ratio = 0.1  # 10%训练样本（核心提速！）
val_sample_ratio = 0.3    # 30%验证样本（保证评估稳定）
sampled_train_size = int(len(train_data) * train_sample_ratio)
sampled_val_size = int(len(val_data) * val_sample_ratio)

# 随机抽样
sampled_train_indices = random.sample(range(len(train_data)), sampled_train_size)
sampled_val_indices = random.sample(range(len(val_data)), sampled_val_size)
sampled_train_data = [train_data[i] for i in sampled_train_indices]
sampled_val_data = [val_data[i] for i in sampled_val_indices]

# 保存缩小后的数据集（避免覆盖原始数据）
small_train_path = "../data/processed/train_en-de_gpu_small.pt"
small_val_path = "../data/processed/validation_en-de_gpu_small.pt"
torch.save(sampled_train_data, small_train_path)
torch.save(sampled_val_data, small_val_path)

print(f"抽样完成！")
print(f"缩小后训练集：{len(sampled_train_data)}样本（保存至{small_train_path}）")
print(f"缩小后验证集：{len(sampled_val_data)}样本（保存至{small_val_path}）")