import pandas as pd

# 读取CSV文件
file_path = 'test_predictions.csv'
df = pd.read_csv(file_path)

# 添加一列Group，用于存储Name列前四个字符
df['Group'] = df['Name'].apply(lambda x: x[:4])

# 按Group分组
grouped = df.groupby('Group')

# 初始化计数器
total_groups = 0
successful_groups = 0

# 遍历每一组
for name, group in grouped:
    total_groups += 1
    # 根据Prediction列排序，取前10名
    top_10 = group.nlargest(8, 'Prediction')
    # 检查对应的Target值是否大于0.9
    success = top_10['Target'].gt(0.8).sum()
    if success >= 1:  # 如果至少有一个真实值大于0.9，则认为预测成功
        successful_groups += 1

# 计算成功概率
success_rate = successful_groups / total_groups if total_groups > 0 else 0

print(f"预测成功的组数: {successful_groups}")
print(f"全部的组数: {total_groups}")
print(f"预测成功的概率: {success_rate:.2%}")


