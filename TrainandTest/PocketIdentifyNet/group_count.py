import pandas as pd

# 读取CSV文件
file_path = 'test_predictions.csv'
df = pd.read_csv(file_path)

# 添加一列Group，用于存储Name列前四个字符
df['Group'] = df['Name'].apply(lambda x: x[:4])

# 按Group分组并统计每组的数量
group_counts = df['Group'].value_counts()

# 将结果保存到txt文件
output_file_path = 'group_counts.txt'
with open(output_file_path, 'w') as f:
    for group, count in group_counts.items():
        f.write(f"{group}: {count}\n")

print(f"Group counts have been saved to {output_file_path}")

