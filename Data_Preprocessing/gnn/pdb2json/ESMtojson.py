import json
import pandas as pd

# 读取JSON文件
with open(r'E:\PDBBind\Result\5000-10000_tmscore_70actxt/pdb_26138json.json', 'r') as json_file:
    data = json.load(json_file)

# 读取CSV文件
csv_data = pd.read_csv(r'E:\PDBBind\Result\5000-10000_tmscore_70actxt\26138AllFeature.csv')

# 获取CSV文件中ESM列的数据
esm_values = csv_data['ESM']

# 遍历JSON数据并添加ESM值
for idx, entry in enumerate(data.values()):
    entry['ESM'] = esm_values[idx]

# 将更新后的数据写入JSON文件
with open(r'E:\PDBBind\Result\5000-10000_tmscore_70actxt/pdb_26138json.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

############把CSV文件中的ESM特征存到json文件中。