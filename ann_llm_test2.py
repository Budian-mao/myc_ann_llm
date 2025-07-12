import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# 用户行为序列数据集类（修复数据索引问题）
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

        # 确保序列长度至少为min_seq_len
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
        # 处理输入维度，确保是2D [batch_size, seq_len]
        if item_seq_item.dim() > 2:
            item_seq_item = item_seq_item.squeeze()
        if item_seq_category.dim() > 2:
            item_seq_category = item_seq_category.squeeze()

        # 获取嵌入
        item_emb = self.item_embedding(item_seq_item)  # [batch_size, seq_len, embedding_dim]
        category_emb = self.category_embedding(item_seq_category)  # [batch_size, seq_len, embedding_dim]

        # 移除冗余维度（仅删除size=1的维度）
        item_emb = torch.squeeze(item_emb) if item_emb.dim() > 3 else item_emb
        category_emb = torch.squeeze(category_emb) if category_emb.dim() > 3 else category_emb

        # 确保是3D
        if item_emb.dim() == 2:
            item_emb = item_emb.unsqueeze(1)
        if category_emb.dim() == 2:
            category_emb = category_emb.unsqueeze(1)

        # 拼接嵌入
        combined_emb = torch.cat([item_emb, category_emb], dim=-1)

        # 最终维度检查
        assert len(
            combined_emb.shape) == 3, f"combined_emb 维度应该是 3，实际是 {len(combined_emb.shape)}，形状为{combined_emb.shape}"

        # Transformer解码
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
        # 处理任意高维输入，只保留前两维
        while item_ids.dim() > 2:
            item_ids = item_ids.squeeze(1)  # 从第1维开始删除冗余维度
        while category_ids.dim() > 2:
            category_ids = category_ids.squeeze(1)

        # 确保至少是2维 [batch_size, seq_len]
        if item_ids.dim() == 1:
            item_ids = item_ids.unsqueeze(1)
        if category_ids.dim() == 1:
            category_ids = category_ids.unsqueeze(1)

        # 获取批次大小和序列长度
        batch_size, seq_len = item_ids.shape

        # 展平计算嵌入
        item_emb = self.item_embedding(item_ids.reshape(-1))  # [batch_size*seq_len, embedding_dim]
        category_emb = self.category_embedding(category_ids.reshape(-1))  # [batch_size*seq_len, embedding_dim]
        combined_emb = torch.cat([item_emb, category_emb], dim=-1)  # [batch_size*seq_len, 2*embedding_dim]

        # 全连接层处理
        x = torch.relu(self.fc1(combined_emb))
        x = torch.relu(self.fc2(x))
        # 恢复批次维度
        x = x.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        return x


# InfoNCE损失函数
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



# 主程序
if __name__ == "__main__":
    # 参数设置
    embedding_dim = 64
    num_heads = 4
    num_layers = 2
    hidden_dim = 128
    vocab_size = 100001  # 确保覆盖所有item_id和category_id
    batch_size = 3
    epochs = 6
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

    # 训练循环
    user_tower.train()
    item_tower.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for item_seq_item, item_seq_category in train_dataloader:
            optimizer.zero_grad()

            # 确保输入是2维
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

    # 推理阶段：生成用户向量（修复数据不匹配问题）
    user_tower.eval()
    test_dataset = UserSequenceDataset('test_data1.csv')

    # 关键修复：设置drop_last=True确保所有批次大小一致
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    user_res_list = []
    processed_indices = []  # 跟踪已处理的样本索引

    with torch.no_grad():
        for batch_idx, (item_seq_item, item_seq_category) in enumerate(test_dataloader):
            if item_seq_item.dim() > 2:
                item_seq_item = item_seq_item.squeeze()
                item_seq_category = item_seq_category.squeeze()

            # 记录批次中的样本索引
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(test_dataset))
            batch_indices = list(range(start_idx, end_idx))

            # 处理当前批次
            user_output = user_tower(item_seq_item, item_seq_category)
            last_output = user_output[:, -1, :]  # [batch_size, dim]

            # 确保只添加实际存在的样本向量
            user_res_list.extend(last_output[:len(batch_indices)].cpu().numpy().tolist())
            processed_indices.extend(batch_indices)

    # 验证向量数量与测试数据行数是否一致
    assert len(user_res_list) == len(
        test_dataset), f"用户向量数量({len(user_res_list)})与测试数据行数({len(test_dataset)})不匹配"
    assert len(processed_indices) == len(
        test_dataset), f"处理的样本数量({len(processed_indices)})与测试数据行数({len(test_dataset)})不匹配"

    # 保存用户向量
    test_data = test_dataset.data
    test_data['user_res'] = user_res_list
    test_data.to_csv('user_res.csv', index=False)
    print(f"用户向量已保存，共 {len(user_res_list)} 条")

    # 推理阶段：生成物品向量
    item_tower.eval()
    item_dataset = ItemDataset('item_data1.csv')
    item_dataloader = DataLoader(item_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    item_res_list = []

    with torch.no_grad():
        for item_id, category_id in item_dataloader:
            # 确保输入是2维 [batch_size, 1]
            item_id = item_id.unsqueeze(1)
            category_id = category_id.unsqueeze(1)
            item_output = item_tower(item_id, category_id)
            item_output = item_output.squeeze(1)  # 移除seq_len维度
            item_res_list.extend(item_output.cpu().numpy().tolist())

    # 验证物品向量数量与物品数据行数是否一致
    assert len(item_res_list) == len(
        item_dataset), f"物品向量数量({len(item_res_list)})与物品数据行数({len(item_dataset)})不匹配"

    # 保存物品向量
    item_data = item_dataset.data
    item_data['item_res'] = item_res_list
    item_data.to_csv('item_res.csv', index=False)
    print(f"物品向量已保存，共 {len(item_res_list)} 条")