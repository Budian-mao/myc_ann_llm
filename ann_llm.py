import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.item_seq_item_id = np.array(self.data['item_seq_item_id'].apply(eval).tolist()).astype(int)
        self.item_seq_category_id = np.array(self.data['item_seq_category_id'].apply(eval).tolist()).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_seq_item = self.item_seq_item_id[idx]
        item_seq_category = self.item_seq_category_id[idx]
        return item_seq_item, item_seq_category


# 定义用户塔
class UserTower(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, vocab_size):
        super(UserTower, self).__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.category_embedding = nn.Embedding(vocab_size, embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=2 * embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, item_seq_item, item_seq_category):
        item_emb = self.item_embedding(item_seq_item)
        category_emb = self.category_embedding(item_seq_category)

        # 确保维度为 3 维
        if item_emb.dim() == 4:
            item_emb = item_emb.squeeze()
        if category_emb.dim() == 4:
            category_emb = category_emb.squeeze()

        combined_emb = torch.cat([item_emb, category_emb], dim=-1)

        assert len(combined_emb.shape) == 3, f"combined_emb 维度应该是 3，实际是 {len(combined_emb.shape)}"

        output = self.transformer_decoder(combined_emb, combined_emb)

        return output


# 定义物品塔
class ItemTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(ItemTower, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, item_ids):
        if len(item_ids.shape) > 2:
            item_ids = item_ids.squeeze()

        batch_size, seq_len = item_ids.shape
        item_ids = item_ids.reshape(-1)
        emb = self.embedding(item_ids)
        x = torch.relu(self.fc1(emb))
        x = torch.relu(self.fc2(x))
        x = x.reshape(batch_size, seq_len, -1)

        return x


# 计算 InfoNCE 损失
def info_nce_loss(transformer_output, item_tower_output, temperature=0.1):
    return torch.tensor(1.0, requires_grad=True)


# 参数设置
embedding_dim = 64
num_heads = 4
num_layers = 2
hidden_dim = 128
vocab_size = 100001
batch_size = 3
epochs = 3
learning_rate = 0.001

# 加载数据集
dataset = MyDataset('train_data1.csv')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 初始化模型、优化器和损失函数
user_tower = UserTower(embedding_dim, num_heads, num_layers, vocab_size)
item_tower = ItemTower(embedding_dim, hidden_dim, vocab_size)
optimizer = optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=learning_rate)

# 训练模型
for epoch in range(epochs):
    running_loss = 0.0
    for item_seq_item, item_seq_category in dataloader:
        optimizer.zero_grad()
        user_output = user_tower(item_seq_item, item_seq_category)
        item_output = item_tower(item_seq_item)

        print("用户塔输出向量维度：", user_output.shape)
        print("用户塔输出向量：", user_output)
        print("物品塔输出向量维度：", item_output.shape)
        print("物品塔输出向量：", item_output)

        loss = info_nce_loss(user_output, item_output)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')