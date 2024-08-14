import os
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("PDB_structure", pdb_file)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() in amino_acids:
                    sequence += amino_acids[residue.get_resname()]
    return sequence

amino_acids = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E",
    "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I",
    "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N",
    "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S",
    "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

pdb_folder_path = r'C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_out'  # 替换为你的文件夹路径
pdb_files = [os.path.join(pdb_folder_path, f) for f in os.listdir(pdb_folder_path) if f.endswith('.pdb')]

sequences_data = []
for file in tqdm(pdb_files, desc="Extracting sequences"):
    file_name = os.path.splitext(os.path.basename(file))[0]  # 提取文件名（不含扩展名）
    sequence = extract_sequence_from_pdb(file)
    sequences_data.append([file_name, sequence])  # 将文件名和序列作为一对数据添加到列表

# 创建DataFrame
df = pd.DataFrame(sequences_data, columns=["File Name", "Sequence"])
csv_file_path = r"C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_out\protein_outSEQ.csv"  # 替换为你想要保存CSV文件的路径
df.to_csv(csv_file_path, index=False)

print(f"Saved sequences to {csv_file_path}")
################把pdb文件中的氨基酸序列转换成csv文件中。