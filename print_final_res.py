import pandas as pd

# 读取final_res.csv文件
try:
    df = pd.read_csv('final_res.csv')
except FileNotFoundError:
    print("错误：未找到final_res.csv文件，请检查文件路径是否正确")
    exit()

# 定义需要统计的列（按顺序排列）
columns_to_check = [
    'in_top_1', 'in_top_2', 'in_top_3', 'in_top_5',
    'in_top_10', 'in_top_20', 'in_top_30', 'in_top_50',
    'in_top_100', 'in_top_200', 'in_top_1000'
]

# 检查列是否存在
missing_columns = [col for col in columns_to_check if col not in df.columns]
if missing_columns:
    print(f"错误：文件中缺少以下列：{', '.join(missing_columns)}")
    exit()

# 统计每列True的概率（占比）
true_probabilities = {}
total_rows = len(df)

for col in columns_to_check:
    # 统计True的数量（自动忽略NaN值）
    true_count = df[col].sum()
    # 计算概率（保留4位小数）
    true_prob = round(true_count / total_rows, 4) if total_rows > 0 else 0
    true_probabilities[col] = true_prob

# 打印结果
print("各列True值的概率统计：")
print("-" * 30)
for col, prob in true_probabilities.items():
    # 格式化输出，按列名对齐
    print(f"{col.ljust(10)}: {prob:.2%}")  # 以百分比形式显示，保留2位小数