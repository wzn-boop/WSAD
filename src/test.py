import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.mountings import mounting_handler
from src.models.mountings.networks_class.rnn import LSTMEncoder


class LstmEnsemble():
    def __init__(self, feature_dim, hidden_size, batch_size):
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.feature_models = nn.ModuleList([LSTMEncoder(1, hidden_size, hidden_size) for _ in range(feature_dim)])
        self.fusion_model = LSTMEncoder(hidden_size * feature_dim, hidden_size, 1)  # Adjusted for combined feature vector
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.get_parameters(), lr=0.001)
        self.epochs = 10

    def get_parameters(self):
        return list(self.feature_models.parameters()) + list(self.fusion_model.parameters())

    def training_prepare(self, X, y):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return train_loader

    def fit(self, X, y):
        train_loader = self.training_prepare(X, y)
        for epoch in range(self.epochs):
            for batch_x, batch_y in train_loader:
                feature_vectors = []
                for i, model in enumerate(self.feature_models):
                    feature_input = batch_x[:, :, i:i + 1]
                    feature_output = model(feature_input)
                    feature_vectors.append(feature_output)

                combined_features = torch.cat(feature_vectors, dim=-1)
                prediction = self.fusion_model(combined_features)
                loss = self.criterion(prediction.squeeze(-1), batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def decision_function(self, X):
        batch_size = X.shape[0]
        self.fusion_model.eval()
        with torch.no_grad():
            feature_vectors = []
            for i, model in enumerate(self.feature_models):
                feature_input = torch.tensor(X[:, :, i:i + 1], dtype=torch.float32)
                feature_output = model(feature_input)
                feature_vectors.append(feature_output)

            combined_features = torch.cat(feature_vectors, dim=1)
            combined_features = combined_features.view(batch_size, 1, -1)  # 调整形状以匹配 LSTM 的输入要求
            prediction = self.fusion_model(combined_features)
        return prediction.squeeze(-1).numpy()

#
# if __name__ == '__main__':
#     x = np.random.randn(2000, 20, 1)  # Example data: 2000 samples, 20 time steps, 1 dimension
#     y = np.random.randn(2000, 1)  # Example labels
#
#     model = LstmEnsemble(feature_dim=1, hidden_size=50, batch_size=32)
#     model.fit(x, y)
#     s = model.decision_function(x)
#     print(s.shape)
#     print(s)

#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
#
# # 生成一些示例的正常数据（这里假设数据是一维时间序列）
# normal_data = np.sin(np.linspace(0, 10, 100))  # 正弦波示例数据
#
# # 将数据转换为Tensor
# normal_data = torch.tensor(normal_data, dtype=torch.float32).view(-1, 1)  # 调整形状为（序列长度，特征维度）
#
# # 定义LSTM模型
# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(LSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.linear = nn.Linear(hidden_size, input_size)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.linear(out)
#         return out
#
# # 初始化模型和优化器
# input_size = 1  # 输入数据的特征维度
# hidden_size = 64  # LSTM隐藏层的大小
# num_layers = 1  # LSTM的层数
# model = LSTM(input_size, hidden_size, num_layers)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# num_epochs = 100
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     output = model(normal_data.unsqueeze(0))  # 在第0维增加一个维度作为batch
#     loss = criterion(output.squeeze(0), normal_data)  # 计算重构误差
#     loss.backward()
#     optimizer.step()
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
#
# # 训练完成后，模型已经学习到了正常数据的模式
#
# # 可以使用训练好的模型对其他时间序列数据进行重构，然后计算重构误差进行异常检测
