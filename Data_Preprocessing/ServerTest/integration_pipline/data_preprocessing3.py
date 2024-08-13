# @Time    : 2023/12/21 18:05
# @Author  : Heng Chang
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

sys.path.append("../")

class data_preprocessing3:
    def __init__(self, file_path, Pocket_path, result_path=None, TMscore_path=None, MSMS="msms", w_dir=None):
        self.result_path = Path(result_path).resolve() if result_path else None
        self.Pocket_path = Path(Pocket_path).resolve()
        self.file_path = Path(file_path).resolve()  # 整个蛋白质的文件路径
        self.msms = Path(MSMS).resolve()
        self.check_file_exist()
        self.check_file_format()
        self.w_dir = Path(w_dir).resolve() if w_dir else None  # w_dir是tmp临时文件夹

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
        site_info = Site_info(self.PDBInfo_Process(), str(self.file_path))  # file_path是整个的蛋白质文件路径
        print('文件路径：：：：', self.file_path)
        dataset = site_info.get_dataset(res_num, 15, 10)
        return dataset

    def form_residue_list(self, residue_list):  # residue_list是某一个蛋白质的dataset
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

    def form_structure(self):
        parser = PDBParser()
        structure = parser.get_structure("1", self.file_path)
        return structure[0]

    def from_domain_to_pdb(self, domain_list, tag="positive"):  # domain_list是一整个蛋白的数据
        out_file_no = 0
        self.tag = tag
        protein_file_paths = []

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

        compare_one_pdb(protein_file_paths, self.Pocket_path, self.result_path)

    def start_processing(self, res_num):
        print('res_num', res_num)
        preprocessed_data = self.SiteInfo_Process(res_num)  # 某个蛋白质的dataset
        print("Preprocessing finished!")
        positive_domain, negative_domain = self.form_residue_list(preprocessed_data)
        print("Domain form finished!")
        if positive_domain:
            self.from_domain_to_pdb(positive_domain, "positive")
        if negative_domain:
            self.from_domain_to_pdb(negative_domain, "negative")
        print("Domain PDB file form finished!")

def get_TMscore(protein_file_path, pocket_file_path, TMscore_path=None):
    path = Path(sys.argv[0]).parent.resolve()
    TMscore = path / "TMscore.exe"
    if TMscore_path:
        TMscore = Path(TMscore_path).resolve()
    cmd = [str(TMscore), str(protein_file_path), str(pocket_file_path)]
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = result.stdout.splitlines()
    for line in lines:
        if "TM-score    = " in line:
            tm_score_value = line.split("TM-score    = ")[1].split()[0]
            TMscore = "{:.4f}".format(float(tm_score_value))
            return float(TMscore)

def compare_one_pdb(protein_file_paths, Pocket_path, result_path=None):
    from collections import defaultdict

    for protein_file_path in protein_file_paths:
        protein_id = Path(protein_file_path).stem.split("_", 1)[0]
        protein_dir = Path(result_path) / protein_id if result_path else Path("Protein_output_1ox2") / protein_id
        sample_dir = protein_dir / "samples"
        protein_dir.mkdir(parents=True, exist_ok=True)
        sample_dir.mkdir(parents=True, exist_ok=True)
        txt_file_path = protein_dir / "samples.txt"

        pocket_files = list(Pocket_path.glob("*.pdb"))
        out_dirs = [round(i * 0.1, 1) for i in range(11)]
        best_samples = defaultdict(lambda: (None, float('inf')))

        for pocket_file_path in pocket_files:
            protein_prefix = protein_file_path.stem.split("_", 1)[0]
            pocket_prefix = pocket_file_path.stem.split("_", 1)[0]

            if pocket_prefix == protein_prefix:
                result = get_TMscore(protein_file_path, pocket_file_path)
                if result is not None:
                    for i in range(len(out_dirs)):
                        if result < out_dirs[i]:
                            lower_bound, upper_bound = out_dirs[i - 1], out_dirs[i]
                            if best_samples[(lower_bound, upper_bound)][1] > result:
                                best_samples[(lower_bound, upper_bound)] = (protein_file_path, result)
                            break

        with open(txt_file_path, "w", encoding="utf-8-sig") as f:
            for (lower_bound, upper_bound), (best_sample, best_result) in best_samples.items():
                if best_sample:
                    f.write(f"{best_sample.name}  {best_result}\n")
                    out_file = sample_dir / best_sample.name
                    try:
                        out_file.write_bytes(best_sample.read_bytes())
                        print(out_file)
                    except Exception:
                        import traceback
                        traceback.print_exc()

    return True
