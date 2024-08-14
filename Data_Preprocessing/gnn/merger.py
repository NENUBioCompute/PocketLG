import pandas as pd

# 读取两个 CSV 文件
A_df = pd.read_csv(r'E:\PDBBind\Result\data_ac_70\data_protein1_5000_data2\DATA\csvtest.csv')
B_df = pd.read_csv(r'E:\PDBBind\Result\data_ac_70\data_protein1_5000_data2\DATA\output_data2.csv')

# 设置 B_df 的索引为第一列，并去除 .pdb 后缀
B_df.index = B_df['Protein ID'].str.replace('.pdb', '')

# 使用 A_df 的第一列（Protein ID）作为索引，并将匹配到的 B_df 的第二列（或其他列）的值赋给 A_df
A_df['Matched Value'] = A_df['Protein ID'].map(B_df['output_data'])

# 将结果保存回 A.csv 文件
A_df.to_csv(r'E:\PDBBind\Result\data_ac_70\data_protein1_5000_data2\DATA\csvtest.csv', index=False)
