# @Time    : 2022/8/24 10:48
# @Author  : Yihang Bao
# @FileName: data_preprocessing.py
# @Description:
import subprocess

import sys
from .site_info import Site_info
from PDB_info import PDBInfo
from Bio.PDB.PDBParser import PDBParser

import warnings
import pickle
import random
import copy
from Bio.PDB import PDBIO
from pathlib import Path

# import argparse


sys.path.append("../")
#from IEProtLib.py_utils.py_mol import PyPeriodicTable, PyProtein, PyProteinBatch


class data_preprocessing:
    def __init__(self, file_path, Pocket_path, result_path=None, TMscore_path=None, MSMS="msms", w_dir=None):
        if result_path is not None:
            self.result_path = Path(result_path).resolve()
        self.Pocket_path = Path(Pocket_path).resolve()
        self.file_path = Path(file_path).resolve()     # 整个蛋白质的文件路径
        self.msms = Path(MSMS).resolve()
        self.check_file_exist()
        self.check_file_format()
        self.w_dir = Path(w_dir).resolve()      # w_dir是tmp临时文件夹

    def check_file_format(self):
        if self.file_path.suffix != ".pdb":
            print(self.file_path)
            raise ValueError("File does not exist!")
        return True

    def check_file_exist(self):
        if not self.file_path.exists():
            print(self.file_path)
            raise ValueError("File does not exist!")
        return True

    def PDBInfo_Process(self):
        res_info = PDBInfo(str(self.file_path), MSMS=str(self.msms))
        return res_info

    def SiteInfo_Process(self, res_num):
        site_info = Site_info(self.PDBInfo_Process(), str(self.file_path))    # file_path是整个的蛋白质文件路径
        print('文件路径：：：：', self.file_path)
        dataset = site_info.get_dataset(res_num, 15, 10)
        return dataset

    def form_residue_list(self, residue_list):    # residue_list是某一个蛋白质的dataset
        positive_domain = []
        negative_domain = []
        for domain in residue_list:
            tmp_dic = []
            for residue in domain["site_res"]:
                tmp_dic.append((residue.resname, residue.parent.id, str(residue.id[1])))
            if domain["isPositive"] == 1:
                positive_domain.append(copy.deepcopy(tmp_dic))
            else:
                negative_domain.append(copy.deepcopy(tmp_dic))
        random.shuffle(negative_domain)
        return positive_domain, negative_domain

        # negative_domain = negative_domain[:len(positive_domain)]
        # output_file = open(
        #     '/home/baoyihang/TMPBDSniffer/test_data_prepare/positive_domain_list/' + file_name.split('.')[
        #         0] + '.pickle', 'wb')
        # pickle.dump(positive_domain, output_file)
        # output_file.close()
        # output_file = open(
        #     '/home/baoyihang/TMPBDSniffer/test_data_prepare/negative_domain_list/' + file_name.split('.')[
        #         0] + '.pickle', 'wb')
        # pickle.dump(negative_domain, output_file)
        # output_file.close()

    def form_structure(self):
        parser = PDBParser()
        structure = parser.get_structure("1", self.file_path)
        return structure[0]

    def from_domain_to_pdb(self, domain_list, tag="positive"):   # domain_list是一整个蛋白的数据
        out_file_no = 0
        self.tag = tag
        protein_file_paths = []
        count = 0

        for domain in domain_list:
            structure = self.form_structure()
            for chain in structure:
                del_list = []
                for residue in chain:
                    if (residue.resname, residue.parent.id, str(residue.id[1])) not in domain:
                        del_list.append(residue.id)
                for i in del_list:
                    chain.detach_child(i)
            del_list = []
            for chain in structure:
                if len(chain) == 0:
                    del_list.append(chain.id)
            for i in del_list:
                structure.detach_child(i)
            io = PDBIO()
            io.set_structure(structure)
            out_file_no += 1
            protein_file_path = self.w_dir / (self.file_path.stem + "_" + str(out_file_no) + "_" + tag + ".pdb")
            print('protein_file_path的路径：', protein_file_path)

            io.save(str(protein_file_path))
            protein_file_paths.append(protein_file_path)  # protein_file_paths列表是一整个蛋白的样本的列表

            # print('protein_file_paths列表：', protein_file_paths)
            # print('count=：', count)



            # 此处进行比较，然后删除采样后的文件（比较列表中的蛋白质样本和pocket）
            compare_one_pdb(protein_file_paths, self.Pocket_path)
           #  result = compare_one_pdb(protein_file_path, self.Pocket_path)
           # # if result is True:
           #  protein_file_path.unlink()


    def start_processing(self, res_num):
        print('res_num', res_num)
        preprocessed_data = self.SiteInfo_Process(res_num)   # 某个蛋白质的dataset
        print("Preprocessing finished!")
        positive_domain, negative_domain = self.form_residue_list(preprocessed_data)
        print("Domain form finished!")
        if positive_domain.__len__() > 0:
            self.from_domain_to_pdb(positive_domain, "positive")
        if negative_domain.__len__() > 0:
            self.from_domain_to_pdb(negative_domain, "negative")
        print("Domain PDB file form finished!")
        # self.generate_hdf5()
        # print('HDF5 file form finished!')


def get_TMscore(protein_file_path, pokcet_file_path, TMscore_path=None):
    path = Path(sys.argv[0]).parent.resolve()
    # 设置 TMalign 执行文件的路径
    #if "win" in sys.platform:
    TMscore = path / "TMscore.exe"

    if TMscore_path is not None:
        TMscore = Path(TMscore_path).resolve()

    cmd = [str(TMscore), str(protein_file_path), str(pokcet_file_path)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = result.stdout.splitlines()

    # for line in lines:
    #     if "TM-score= " in line:
    #         TMscore = line.split(" ", 1)[1]
    #         TMscore = TMscore.split(" ", 1)[0]
    #         TMscore = float(TMscore)
    #         print(TMscore, type(TMscore))
    #         return TMscore
    for line in lines:
        # 检查当前行是否包含 "TM-score    = "
        if "TM-score    = " in line:
            # 使用 split 切分字符串，获取包含 TM-score 的部分
            tm_score_part = line.split("TM-score    = ")[1]
            # 使用 split 切分字符串，获取 TM-score 的具体值
            tm_score_value = tm_score_part.split()[0]
            # 将 TM-score 转换为浮点数，并格式化为小数点后4位的字符串
            TMscore = "{:.4f}".format(float(tm_score_value))
            return float(TMscore)
# def get_TMscore(protein_file_path, pocket_file_path, TMalign_path=None):
#     path = Path(sys.argv[0]).parent.resolve()
#     # 设置 TMalign 执行文件的路径
#     if "win" in sys.platform:
#         TMalign = path / "TMalign.exe"
#     elif "linux" in sys.platform:
#         TMalign = path / "TMalign"
#
#     if TMalign_path is not None:
#         TMalign = Path(TMalign_path).resolve()
#
#     cmd = [str(TMalign), str(protein_file_path), str(pocket_file_path)]
#
#     result = subprocess.Popen(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     lines = result.stdout.read().splitlines()
#
#     for line in lines:
#         if "TM-score= " in line:
#             TMscore = line.split(" ", 1)[1]
#             TMscore = TMscore.split(" ", 1)[0]
#             TMscore = float(TMscore)
#             print('TMSCORE', TMscore)
#             return TMscore

def compare_one_pdb(Protein_path_file, Pocket_path, result_path=None):
    protein_file_path = Path(Protein_path_file).resolve()
    # 在关键步骤后添加打印日志
    # print("Processing:", protein_file_path.name)

    path = Path(sys.argv[0]).parent.resolve()
    out_put = path / "Protein_output_1ox2"  # 这是真正的输出路径
    # print('out_put',out_put)
    if result_path is not None:
        out_put = Path(result_path).resolve()

    if not out_put.exists():
        out_put.mkdir(parents=True, exist_ok=True)

    pokcet_files = []

    for file in Pocket_path.glob("*.pdb"):
        pokcet_files.append(file)

    out_dirs = []
    out_start = 0
    for _ in range(11):
        out_dirs.append(round(out_start, 1))
        out_start += 0.1

    max_save = len(pokcet_files)
    isOk = False
    for pokcet_file_path in pokcet_files:
        protein_filestem = protein_file_path.stem
        pokcet_filestem = pokcet_file_path.stem
        protein_prefix = protein_filestem.split("_", 1)[0]
        pokcet_prefix = pokcet_filestem.split("_", 1)[0]
        if pokcet_prefix == protein_prefix:
            result = get_TMscore(protein_file_path, pokcet_file_path)
            if result is not None:
                isOk = True
                # print(protein_file_path.name, result)
                for i in range(len(out_dirs)):
                    if result < out_dirs[i]:
                        out_put1 = out_put / (str(out_dirs[i - 1]) + "-" + str(out_dirs[i]))
                        if not out_put1.exists():
                            out_put1.mkdir(parents=True, exist_ok=True)
                        tmp_files = []
                        for tmp_file in out_put1.glob("*.pdb"):
                            tmp_files.append(tmp_file)
                        # if len(tmp_files) > max_save - 1:
                        #     break
                        # isEnough = False
                        # for tmp_file in tmp_files:
                        #     if pokcet_prefix in tmp_file.name:
                        #         isEnough = True
                        #         break
                        # if isEnough is True:
                        #     break
                        out_txt = out_put / (str(out_dirs[i - 1]) + "-" + str(out_dirs[i]) + ".txt")
                        with open(out_txt, "a+", encoding="utf-8-sig") as f:
                            f.write(protein_file_path.name + "  " + str(result) + "\n")
                        out_file = out_put1 / protein_file_path.name
                        try:
                            out_file.write_bytes(protein_file_path.read_bytes())
                            print(out_file)
                        except Exception:
                            import traceback

                            traceback.print_exc()
                        break
    return isOk


