import pandas as pd

# 读取两个 CSV 文件
A_df = pd.read_csv(r'E:\PDBBind\Result\data_ac_70\data_protein1_5000_data2\DATA\csvtest1.csv')
B_df = pd.read_csv(r'E:\PDBBind\Result\data_ac_70\data_protein1_5000_data2\DATA\output_data2.csv')

# 去除 B_df 中第一列的 .pdb 后缀
B_df['Protein ID'] = B_df['Protein ID'].str.replace('.pdb', '')

# 使用 merge 函数按照第一列（Protein ID）进行匹配，并保留只存在于 A_df 中的行
merged_df = pd.merge(A_df, B_df, left_on='Protein ID', right_on='Protein ID', how='left', indicator=True)

# 打印匹配不到的内容
print("匹配不到的内容：")
print(merged_df[merged_df['_merge'] == 'left_only'])

# 删除匹配不到的行
result_df = merged_df[merged_df['_merge'] != 'left_only']
result_df.drop(columns=['_merge'], inplace=True)

# 保存结果到 csvtest.csv 文件中
result_df.to_csv(r'E:\PDBBind\Result\data_ac_70\data_protein1_5000_data2\DATA\csvtest1.csv', index=False)


