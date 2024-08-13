# @Time    : 2022/8/25 15:08
# @Author  : Yihang Bao
# @FileName: generate_pymol_cmd.py
# @Description:

from Bio.PDB.PDBParser import PDBParser
import warnings
import os
import argparse


def generate_pymol_cmd(file_path, score_list, max_vis):
    color_list = ['0x000079', '0x2828FF', '0x6A6AFF', '0x9393FF', '0xCECEFF', '0x460046', '0x930093', '0xE800E8', '0xFF77FF', '0xFFBFFF', '0x007979', '0x00CACA', '0x4DFFFF']
    w_cmd = 'show surface, all;\ncolor gray90, all;\nremove resn HOH;\n'
    co = 0

    for name in score_list:
        if co >= max_vis and not name[0].endswith('positive'):
            continue
        elif not name[0].endswith('positive'):
            co += 1
        parser = PDBParser()
        structure = parser.get_structure('1', file_path + '/domain_pdb/' + name[0] + '.pdb')
        structure = structure[0]
        for chain in structure:
            for residue in chain:
                w_cmd += 'select /' + name[0][:4] + '//' + residue.parent.id + '/' + str(residue.id[1]) + '\n'
                if name[0].endswith('positive'):
                    w_cmd += 'color ' + 'forest' + ', sele;\n'
                else:
                    w_cmd += 'color ' + color_list[co - 1] + ', sele;\n'
    f = open(file_path + '/' + name[0][:4] + '_pymol_cmd.txt', 'w')
    f.write(w_cmd)
    f.close()

def get_score(file_path, pdbpath):
    f = open(file_path + pdbpath.split('.')[-2].split('/')[-1] + '_result_domain.txt', 'r')
    score_list = []
    for line in f:
        if line[0] == '\n':
            continue
        score_list.append([line.split(',')[0].strip(), float(line.split('[')[1].split(']')[0].strip())])
    f.close()
    score_list.sort(key=lambda x: x[1], reverse=True)
    return score_list


if '__main__' == __name__:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='transform to PDB')
    parser.add_argument('--pdbpath')
    args = parser.parse_args()
    a = get_score('/root/autodl-tmp/output/', args.pdbpath)
    max_vis = 13
    generate_pymol_cmd('/root/autodl-tmp/output/', a, max_vis)
    os.system("sshpass -p Biodata123 scp /root/autodl-tmp/output/" + args.pdbpath.split('.')[-2].split('/')[-1] + "_pymol_cmd.txt root@8.142.75.82:/home/downloads")
    print('You can downloads your file through this link: http://8.142.75.82:84/' + args.pdbpath.split('.')[-2].split('/')[-1] + '_pymol_cmd.txt')

