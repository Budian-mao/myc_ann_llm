import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

##存档代码
# 定义数据集类（不变）
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


# 定义用户塔（不变）
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
        if item_emb.dim() == 4:
            item_emb = item_emb.squeeze()
        if category_emb.dim() == 4:
            category_emb = category_emb.squeeze()
        combined_emb = torch.cat([item_emb, category_emb], dim=-1)
        assert len(combined_emb.shape) == 3, f"combined_emb 维度应该是 3，实际是 {len(combined_emb.shape)}"
        output = self.transformer_decoder(combined_emb, combined_emb)
        return output


# 定义物品塔（不变）
class ItemTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(ItemTower, self).__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.category_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, item_ids, category_ids):
        if item_ids.dim() > 2:
            item_ids = item_ids.squeeze()
        if category_ids.dim() > 2:
            category_ids = category_ids.squeeze()
        batch_size, seq_len = item_ids.shape
        item_emb = self.item_embedding(item_ids.reshape(-1))
        category_emb = self.category_embedding(category_ids.reshape(-1))
        combined_emb = torch.cat([item_emb, category_emb], dim=-1)
        x = torch.relu(self.fc1(combined_emb))
        x = torch.relu(self.fc2(x))
        x = x.reshape(batch_size, seq_len, -1)
        return x


# 仅修改此损失函数
# 仅修改此损失函数
# 仅修改此损失函数
def info_nce_loss(transformer_output, item_tower_output, temperature=0.1):
    # 1. 最后维度归一化
    user_norm = torch.nn.functional.normalize(transformer_output, dim=-1)
    item_norm = torch.nn.functional.normalize(item_tower_output, dim=-1)

    # 2. 切片得到 [batch, seq_len-1, dim]
    user_slice = user_norm[:, :-1, :]
    item_slice = item_norm[:, 1:, :]

    # 3. 计算原始retreive_score矩阵 [L, L]，其中L = batch*(seq_len-1)
    batch_size, seq_len_minus_1, dim = user_slice.shape
    L = batch_size * seq_len_minus_1  # 总长度
    user_flat = user_slice.reshape(-1, dim)  # [L, dim]
    item_flat = item_slice.reshape(-1, dim)  # [L, dim]
    retreive_score = torch.matmul(user_flat, item_flat.T)  # [L, L]

    # 4. 生成positive_score矩阵：对角线保留原始值，其他位置为1e-8
    positive_score = torch.full_like(retreive_score, 1e-8)  # 初始化全为1e-8
    diag_indices = torch.arange(L, device=retreive_score.device)  # 对角线索引
    positive_score[diag_indices, diag_indices] = retreive_score[diag_indices, diag_indices]  # 填充对角线

    # 5. 生成neg_score矩阵：每行n对应的样本内部列设为1e-8，其他保留原始值
    neg_score = retreive_score.clone()  # 复制原始矩阵
    for n in range(L):
        k = n // seq_len_minus_1
        start_col = k * seq_len_minus_1
        end_col = (k + 1) * seq_len_minus_1
        neg_score[n, start_col:end_col] = 1e-8

    # 6. 计算每一行的正样本分数和所有样本分数（使用你的方法）
    pos_exp = torch.exp(positive_score / temperature)  # 正样本指数
    neg_exp = torch.exp(neg_score / temperature)  # 负样本指数

    pos_sum = torch.sum(pos_exp, dim=1)  # 正样本指数之和
    all_sum = torch.sum(pos_exp + neg_exp, dim=1)  # 所有样本指数之和

    # 7. 计算每个样本的 InfoNCE 损失（等价于 -log(pos_sum/all_sum)）
    pos_log = torch.log(pos_sum)
    all_log = torch.log(all_sum)
    loss_per_sample = -(pos_log - all_log)  # 等价于 -log(pos_sum/all_sum)

    # 8. 计算平均损失
    final_loss = torch.mean(loss_per_sample)

    # 9. 打印中间结果（可选）
    print(f"positive_score矩阵维度：{positive_score.shape}")
    print(f"neg_score矩阵维度：{neg_score.shape}")
    print(f"最终损失: {final_loss.item()}")

    return final_loss


# 参数设置及训练逻辑（不变）
embedding_dim = 64
num_heads = 4
num_layers = 2
hidden_dim = 128
vocab_size = 100001
batch_size = 3
epochs = 3
learning_rate = 0.001

dataset = MyDataset('train_data1.csv')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

user_tower = UserTower(embedding_dim, num_heads, num_layers, vocab_size)
item_tower = ItemTower(embedding_dim, hidden_dim, vocab_size)
optimizer = optim.Adam(list(user_tower.parameters()) + list(item_tower.parameters()), lr=learning_rate)

for epoch in range(epochs):
    running_loss = 0.0
    for item_seq_item, item_seq_category in dataloader:
        optimizer.zero_grad()
        user_output = user_tower(item_seq_item, item_seq_category)
        item_output = item_tower(item_seq_item, item_seq_category)
        loss = info_nce_loss(user_output, item_output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')