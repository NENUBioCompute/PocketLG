import csv
import os
import torch
import esm
from Bio import SeqIO, BiopythonParserWarning
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=BiopythonParserWarning)

def extract_protein_features_from_csv(input_csv, output_csv):

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    fieldnames = ['Protein ID', 'Feature Representation']
    with open(input_csv, 'r', newline='') as f_in, open(output_csv, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        count = 0

        for row in reader:

            pdb_file = row[0]
            protein_sequence = row[1]
            count+=1
            print('count', count)
            print('-----', pdb_file)


            # 计算特征表示
            feature_representation = calculate_feature_representation(model, alphabet, batch_converter, protein_sequence)

            # 将特征表示写入输出CSV文件
            writer.writerow({
                'Protein ID': pdb_file,
                'Feature Representation': feature_representation
            })

def calculate_feature_representation(model, alphabet, batch_converter, protein_sequence):
    # 生成随机或预定义的标签，以及将序列放入列表中以符合batch_converter的输入要求
    data = [("protein", protein_sequence)]

    # 使用batch_converter处理数据
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # 在不需要梯度的上下文中提取每个残基的表示
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # 通过平均生成每个序列的表示
    sequence_representation = token_representations[0, 1:batch_lens[0] - 1].mean(0)

    return sequence_representation.numpy().tolist()

if __name__ == "__main__":
    input_csv = r"C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_outSEQ.csv"  # 输入的CSV文件路径
    output_csv = r"C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_outESM.csv"  # 输出的CSV文件路径
    extract_protein_features_from_csv(input_csv, output_csv)

#****************************把预处理好的氨基酸序列的csv作为输出然后生成ESM特征csv文件**********************


