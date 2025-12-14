# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
#
#
# # 定义LSTM特征提取器
# class LSTMFeatureExtractor(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(LSTMFeatureExtractor, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#
#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         return hn[-1]  # 返回最后一个隐藏状态作为特征向量
#
#
# # 定义异常检测模型
# class AnomalyDetector(nn.Module):
#     def __init__(self, input_size, hidden_size, sequence_length):
#         super(AnomalyDetector, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.output_layer = nn.Linear(hidden_size, input_size)
#         self.sequence_length = sequence_length
#
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         reconstructed = self.output_layer(lstm_out)
#         return reconstructed
#
# # 整合模型与训练流程
# class MultidimensionalAnomalyDetection:
#     def __init__(self, feature_dim, hidden_size, sequence_length):
#         self.feature_extractors = nn.ModuleList([LSTMFeatureExtractor(1, hidden_size) for _ in range(feature_dim)])
#         self.anomaly_detector = AnomalyDetector(hidden_size * feature_dim, hidden_size, sequence_length)
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
#         self.sequence_length = sequence_length
#
#     def parameters(self):
#         return list(self.feature_extractors.parameters()) + list(self.anomaly_detector.parameters())
#
#     def fit(self, X):
#         # X: shape [samples, sequence_length, features]
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         dataset = TensorDataset(X_tensor)
#         dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#
#         for epoch in range(10):  # Example: 10 epochs
#             for batch in dataloader:
#                 batch_x = batch[0]
#                 batch_x = batch_x.unsqueeze(-1)  # Add channel dimension
#                 feature_vectors = [extractor(batch_x[:, :, i:i + 1]) for i, extractor in enumerate(self.feature_extractors)]
#                 combined_features = torch.cat(feature_vectors, dim=2)
#                 reconstructed = self.anomaly_detector(combined_features)
#                 loss = torch.mean((reconstructed - batch_x.squeeze(-1)) ** 2)
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#             print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
#
#
# if __name__ == '__main__':
#     x = np.random.randn(2000, 20)
#
#     model = MultidimensionalAnomalyDetection(
#         feature_dim = x.shape[1], hidden_size = 100 , sequence_length = 100
#     )
#     model.fit(x)
#     #print(s.shape)

#
# import os
# import numpy as np
# import pandas as pd
#
# # 设置原始文件夹路径
# original_folder_path = r'E:\BaiduNetdiskDownload\SMAP&MSL\SMAP&MSL\train'
# # 设置新文件夹路径，用于保存转换后的csv文件
# new_folder_path = r'E:\sys\Desktop\data\_processed_data'
#
# # 确保新文件夹存在
# if not os.path.exists(new_folder_path):
#     os.makedirs(new_folder_path)
#
# # 遍历原始文件夹中的所有文件
# for filename in os.listdir(original_folder_path):
#     # 检查文件扩展名是否为.npy
#     if filename.endswith('.npy'):
#         # 构建.npy文件的完整路径
#         npy_file_path = os.path.join(original_folder_path, filename)
#
#         # 加载.npy文件
#         data_np = np.load(npy_file_path)
#
#         # 将NumPy数组转换为DataFrame
#         data_df = pd.DataFrame(data_np)
#
#         # 构建.csv文件的完整路径
#         csv_filename = filename[:-4] + '.csv'  # 去掉.npy扩展名，加上.csv扩展名
#         csv_file_path = os.path.join(new_folder_path, csv_filename)
#
#         # 保存DataFrame为.csv文件
#         data_df.to_csv(csv_file_path, index=False)
#         print(f"Converted '{npy_file_path}' to '{csv_file_path}'")
#
# print("所有.npy文件转换完成。")

import os

# 设置原始文件夹路径
original_folder_path = r'E:\sys\Desktop\data\_processed_data\test'
# 设置要添加的后缀
suffix = '_train'

# 递归遍历文件夹中的所有文件和子文件夹
for root, dirs, files in os.walk(original_folder_path):
    for file in files:
        # 构建文件的完整路径
        file_path = os.path.join(root, file)

        # 检查文件名是否已经包含后缀
        if suffix not in file:
            # 拆分文件名和扩展名
            file_base, file_extension = os.path.splitext(file)

            # 创建新的文件名，添加后缀
            new_file_name = f"{file_base}{suffix}{file_extension}"

            # 构建新的文件路径
            new_file_path = os.path.join(root, new_file_name)

            # 重命名文件
            os.rename(file_path, new_file_path)
            print(f"Renamed '{file_path}' to '{new_file_path}'")

# print("所有文件名更新完成。")


# import os
# import shutil
#
# # 设置原始文件夹路径
# original_folder_path = 'E:\sys\Desktop\data\_processed_data\MSL'
#
# # 遍历原始文件夹中的所有文件
# for filename in os.listdir(original_folder_path):
#     # 构建原始文件的完整路径
#     file_path = os.path.join(original_folder_path, filename)
#
#     # 检查是否为文件
#     if os.path.isfile(file_path):
#         # 拆分文件名和扩展名
#         file_base, file_extension = os.path.splitext(filename)
#
#         # 假设文件名的后一部分是由特定字符分隔的，例如'-'或'_'
#         # 这里我们使用'_'作为分隔符，你可以根据自己的需求修改
#         parts = file_base.rsplit('_', 1)  # 使用rsplit确保只分割最后一个'_'
#
#         # 检查分割后的列表长度，确保至少有两个部分：文件名前一部分和后一部分
#         if len(parts) > 1:
#             # 使用分割后的第一个部分作为新文件夹的名称
#             new_folder_name = parts[0]
#         else:
#             # 如果没有找到分隔符，使用原始文件名作为新文件夹的名称
#             new_folder_name = file_base
#
#         # 创建以文件名前一部分命名的新文件夹路径
#         new_folder_path = os.path.join(original_folder_path, new_folder_name)
#
#         # 如果新文件夹不存在，则创建它
#         if not os.path.exists(new_folder_path):
#             os.makedirs(new_folder_path)
#
#         # 构建目标文件的完整路径
#         new_file_path = os.path.join(new_folder_path, filename)
#
#         # 将文件移动到新文件夹
#         shutil.move(file_path, new_file_path)
#         print(f"Moved '{file_path}' to '{new_file_path}'")
#
# print("所有文件已移动到新的文件夹下。")