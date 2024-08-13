# @Time    : 2022/8/31 16:27
# @Author  : Yihang Bao
# @FileName: cmd_1.py
# @Description:

import argparse

f = open("./pdb_list2.txt", "r")
pdb_list = []
parser = argparse.ArgumentParser(description="transform to PDB")
parser.add_argument("--no")
args = parser.parse_args()
for i in f:
    if i[0] != "\n":
        pdb_list.append(i.strip())
f.close()
w_content = ""

cut_num = 11
cut_len = len(pdb_list) // cut_num
no = int(args.no)
if no != cut_num:
    pdb_list = pdb_list[(no - 1) * cut_len : no * cut_len]
else:
    pdb_list = pdb_list[(no - 1) * cut_len :]


for i in pdb_list:
    w_content += "python3 data_preprocessing0.py --pdbpath /D:/pdbtmdemo/pdb/" + i + "\n"

w = open("./cmd_" + args.no + ".sh", "w")
w.write(w_content)
w.close()
