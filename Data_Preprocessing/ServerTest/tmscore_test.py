import sys
from Bio.PDB.PDBParser import PDBParser
import subprocess
import warnings
import pickle
import random
import copy
from Bio.PDB import PDBIO
from pathlib import Path

# import argparse


sys.path.append("../")
def get_TMscore(protein_file_path, pocket_file_path, TMalign_path=None):
    #path = 'E:/PDBBind/data_dealer_tmscore3'
    # 设置 TMalign 执行文件的路径
    #if "win" in sys.platform:
    TMalign = "E:/PDBBind/data_dealer_tmscore3/TMscore.exe"
    # elif "linux" in sys.platform:
    #     TMalign = path / "TMalign"

    if TMalign_path is not None:
        TMalign = Path(TMalign_path).resolve()

    cmd = [str(TMalign), str(protein_file_path), str(pocket_file_path)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    lines = result.stdout.splitlines()
    print('!!!!!', lines)
    #
    # for line in lines:
    #     if "TM-score= " in line:
    #         TMscore = line.split(" ", 1)[1]
    #         TMscore = TMscore.split(" ", 1)[0]
    #         TMscore = float(TMscore)
    #
    #         return TMscore, type(TMscore)
    # 遍历每一行
    for line in lines:
        # 检查当前行是否包含 "TM-score    = "
        if "TM-score    = " in line:
            # 使用 split 切分字符串，获取包含 TM-score 的部分
            tm_score_part = line.split("TM-score    = ")[1]

            # 使用 split 切分字符串，获取 TM-score 的具体值
            tm_score_value = tm_score_part.split()[0]

            # 将 TM-score 转换为浮点数，并格式化为小数点后4位的字符串
            TMscore = "{:.4f}".format(float(tm_score_value))

            return TMscore,type(TMscore)

if __name__ == "__main__":
    protein_file_path ='D:\PDBbind_v2020_other_PL\DATA\Protein100\\1a0q_protein_44.pdb'
    pocket_file_path = 'D:\PDBbind_v2020_other_PL\DATA\Pocket100\\1a0q_pocket.pdb'

    # 调用 get_TMscore 函数
    TMscore = get_TMscore(protein_file_path, pocket_file_path)

    # 打印结果
    print(f'TM-score: {TMscore}')


