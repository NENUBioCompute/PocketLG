import re
import json
import os

from tqdm import tqdm


def three_to_one(three_letter_code):
    """Convert three-letter amino acid code to one-letter code."""
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    return aa_dict.get(three_letter_code, 'X')


def parse_pdb(pdb_file):
    coords = {'N': [], 'CA': [], 'C': [], 'O': []}
    amino_acids = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                if atom_name in coords:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords[atom_name].append([x, y, z])
                    # Extract amino acid information only when encountering 'N' atom
                    if atom_name == 'N':
                        amino_acid = three_to_one(line[17:20].strip())
                        amino_acids.append(amino_acid)

    # Construct the sequence by joining the amino acids
    sequence = ''.join(amino_acids)
    return coords, sequence


def pdb_to_json_folder(folder_path):
    json_data_batch = {}
    pdb_files = [f for f in os.listdir(folder_path) if f.endswith('.pdb')]
    for file_name in tqdm(pdb_files, desc="Processing PDB files"):
            pdb_file = os.path.join(folder_path, file_name)
            pdb_name = os.path.basename(pdb_file)
            pdb_id = pdb_name.split('.')[0]
            pdb_coords, pdb_sequence = parse_pdb(pdb_file)
            pdb_data = {
                "PDB_id": pdb_id,
                "seq": pdb_sequence,
                "coords": pdb_coords
            }
            json_data_batch[pdb_name] = pdb_data
    return json_data_batch


def write_json_file(json_data, output_file):
    with open(output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


# Example usage
pdb_folder = r'C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_out'
output_file = r'C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_out\protein_outjson.json'
# import json
#
# # 定义 JSON 文件路径
# file_path = 'data.json'
#
# # 读取 JSON 文件内容
# with open(output_file, 'r') as file:
#     data = json.load(file)
#
# # 打印读取的 JSON 数据
# print(data)


json_data_batch = pdb_to_json_folder(pdb_folder)
write_json_file(json_data_batch, output_file)
print("JSON data has been written to", output_file)

