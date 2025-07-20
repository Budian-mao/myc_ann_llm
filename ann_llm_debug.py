import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize  # 用于numpy数组的归一化


# 用户行为序列数据集类
class UserSequenceDataset(Dataset):
    def __init__(self, data_path, min_seq_len=2):
        self.data = pd.read_csv(data_path)
        self.item_seq_item_id = np.array(self.data['item_seq_item_id'].apply(eval).tolist()).astype(int)
        self.item_seq_category_id = np.array(self.data['item_seq_category_id'].apply(eval).tolist()).astype(int)
        self.min_seq_len = min_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_seq_item = self.item_seq_item_id[idx]
        item_seq_category = self.item_seq_category_id[idx]

        if len(item_seq_item) < self.min_seq_len:
            padding_length = self.min_seq_len - len(item_seq_item)
            item_seq_item = np.pad(item_seq_item, (0, padding_length), mode='constant')
            item_seq_category = np.pad(item_seq_category, (0, padding_length), mode='constant')

        return item_seq_item, item_seq_category


# 静态物品数据集类
class ItemDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.item_id = np.array(self.data['item_id']).astype(int)
        self.category_id = np.array(self.data['category_id']).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.item_id[idx], self.category_id[idx]


# 用户塔模型
class UserTower(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, vocab_size):
        super(UserTower, self).__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.category_embedding = nn.Embedding(vocab_size, embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=2 * embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, item_seq_item, item_seq_category):
        if item_seq_item.dim() > 2:
            item_seq_item = item_seq_item.squeeze()
        if item_seq_category.dim() > 2:
            item_seq_category = item_seq_category.squeeze()

        item_emb = self.item_embedding(item_seq_item)
        category_emb = self.category_embedding(item_seq_category)

        item_emb = torch.squeeze(item_emb) if item_emb.dim() > 3 else item_emb
        category_emb = torch.squeeze(category_emb) if category_emb.dim() > 3 else category_emb

        if item_emb.dim() == 2:
            item_emb = item_emb.unsqueeze(1)
        if category_emb.dim() == 2:
            category_emb = category_emb.unsqueeze(1)

        combined_emb = torch.cat([item_emb, category_emb], dim=-1)
        assert len(
            combined_emb.shape) == 3, f"combined_emb 维度应该是 3，实际是 {len(combined_emb.shape)}，形状为{combined_emb.shape}"

        output = self.transformer_decoder(combined_emb, combined_emb)
        return output


# 物品塔模型
class ItemTower(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(ItemTower, self).__init__()
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.category_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(2 * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, item_ids, category_ids):
        while item_ids.dim() > 2:
            item_ids = item_ids.squeeze(1)
        while category_ids.dim() > 2:
            category_ids = category_ids.squeeze(1)

        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(1)
        if category_ids.dim() == 1:
            category_ids = category_ids.unsqueeze(1)

        batch_size, seq_len = item_ids.shape
        item_emb = self.item_embedding(item_ids.reshape(-1))
        category_emb = self.category_embedding(category_ids.reshape(-1))
        combined_emb = torch.cat([item_emb, category_emb], dim=-1)

        x = torch.relu(self.fc1(combined_emb))
        x = torch.relu(self.fc2(x))
        x = x.reshape(batch_size, seq_len, -1)
        return x


# InfoNCE损失函数
def info_nce_loss(transformer_output, item_tower_output, temperature=0.1):
    user_norm = torch.nn.functional.normalize(transformer_output, dim=-1)
    item_norm = torch.nn.functional.normalize(item_tower_output, dim=-1)

    user_slice = user_norm[:, :-1, :]
    item_slice = item_norm[:, 1:, :]

    batch_size, seq_len_minus_1, dim = user_slice.shape
    L = batch_size * seq_len_minus_1
    user_flat = user_slice.reshape(-1, dim)
    item_flat = item_slice.reshape(-1, dim)
    retreive_score = torch.matmul(user_flat, item_flat.T)

    positive_score = torch.full_like(retreive_score, 1e-8)
    diag_indices = torch.arange(L, device=retreive_score.device)
    positive_score[diag_indices, diag_indices] = retreive_score[diag_indices, diag_indices]

    neg_score = retreive_score.clone()
    for n in range(L):
        k = n // seq_len_minus_1
        start_col = k * seq_len_minus_1
        end_col = (k + 1) * seq_len_minus_1
        neg_score[n, start_col:end_col] = 1e-8

    pos_exp = torch.exp(positive_score / temperature)
    neg_exp = torch.exp(neg_score / temperature)

    pos_sum = torch.sum(pos_exp, dim=1)
    all_sum = torch.sum(pos_exp + neg_exp, dim=1)

    pos_log = torch.log(pos_sum)
    all_log = torch.log(all_sum)
    loss_per_sample = -(pos_log - all_log)
    final_loss = torch.mean(loss_per_sample)

    print(f"positive_score矩阵维度：{positive_score.shape}")
    print(f"neg_score矩阵维度：{neg_score.shape}")
    print(f"最终损失: {final_loss.item()}")

    return final_loss


# 主程序
if __name__ == "__main__":
    # 参数设置
    embedding_dim = 64
    num_heads = 4
    num_layers = 2
    hidden_dim = 128
    vocab_size = 100001
    batch_size = 3
    epochs = 1
    learning_rate = 0.001

    # 训练过程
    train_dataset = UserSequenceDataset('train_data1.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    user_tower = UserTower(embedding_dim, num_heads, num_layers, vocab_size)
    item_tower = ItemTower(embedding_dim, hidden_dim, vocab_size)
    optimizer = optim.Adam(
        list(user_tower.parameters()) + list(item_tower.parameters()),
        lr=learning_rate
    )

    user_tower.train()
    item_tower.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for item_seq_item, item_seq_category in train_dataloader:
            optimizer.zero_grad()

            if item_seq_item.dim() > 2:
                item_seq_item = item_seq_item.squeeze()
                item_seq_category = item_seq_category.squeeze()

            user_output = user_tower(item_seq_item, item_seq_category)
            item_output = item_tower(item_seq_item, item_seq_category)
            loss = info_nce_loss(user_output, item_output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_dataloader):.4f}')

    # 推理阶段：生成用户向量（并归一化）
    user_tower.eval()
    test_dataset = UserSequenceDataset('test_data1.csv')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    user_res_list = []
    processed_indices = []

    with torch.no_grad():
        for batch_idx, (item_seq_item, item_seq_category) in enumerate(test_dataloader):
            if item_seq_item.dim() > 2:
                item_seq_item = item_seq_item.squeeze()
                item_seq_category = item_seq_category.squeeze()

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_dataset))
            batch_indices = list(range(start_idx, end_idx))

            user_output = user_tower(item_seq_item, item_seq_category)
            last_output = user_output[:, -1, :]  # 取最后一个输出

            # 对用户向量进行L2归一化（关键步骤）
            normalized_user = torch.nn.functional.normalize(last_output, dim=-1)

            user_res_list.extend(normalized_user[:len(batch_indices)].cpu().numpy().tolist())
            processed_indices.extend(batch_indices)

    assert len(user_res_list) == len(
        test_dataset), f"用户向量数量({len(user_res_list)})与测试数据行数({len(test_dataset)})不匹配"
    assert len(processed_indices) == len(
        test_dataset), f"处理的样本数量({len(processed_indices)})与测试数据行数({len(test_dataset)})不匹配"

    # 保存归一化后的用户向量
    test_data = test_dataset.data
    test_data['user_res'] = user_res_list
    test_data.to_csv('user_res.csv', index=False)
    print(f"归一化用户向量已保存，共 {len(user_res_list)} 条")

    # 推理阶段：生成物品向量（并归一化）
    item_tower.eval()
    item_dataset = ItemDataset('item_data1.csv')
    item_dataloader = DataLoader(item_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    item_res_list = []

    with torch.no_grad():
        for item_id, category_id in item_dataloader:
            item_id = item_id.unsqueeze(1)
            category_id = category_id.unsqueeze(1)
            item_output = item_tower(item_id, category_id)
            item_output = item_output.squeeze(1)  # 移除seq_len维度

            # 对物品向量进行L2归一化（关键步骤）
            normalized_item = torch.nn.functional.normalize(item_output, dim=-1)

            item_res_list.extend(normalized_item.cpu().numpy().tolist())

    assert len(item_res_list) == len(
        item_dataset), f"物品向量数量({len(item_res_list)})与物品数据行数({len(item_dataset)})不匹配"

    # 保存归一化后的物品向量
    item_data = item_dataset.data
    item_data['item_res'] = item_res_list
    item_data.to_csv('item_res.csv', index=False)
    print(f"归一化物品向量已保存，共 {len(item_res_list)} 条")