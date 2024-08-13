# @Time    : 2022/8/24 10:48
# @Author  : Yihang Bao
# @FileName: data_preprocessing.py
# @Description:
import os
import sys
from datadealer import file_path
from site_info import Site_info
from PDB_info import PDBInfo
from Bio.PDB.PDBParser import PDBParser
from pathlib import Path
sys.path.append('/E:/PDBBind/data_dealer_tmscore3/ServerTest/integration_pipline/PDB/PDBparser/DataFormat.py')
sys.path.append('E:/PDBBind/data_dealer_tmscore3/ServerTest/IEProtLib')
sys.path.append('E:/PDBBind/data_dealer_tmscore3/ServerTest/IEProtLib/py_utils/py_mol/PyPeriodicTable.py')
sys.path.append('E:/PDBBind/data_dealer_tmscore3/ServerTest/IEProtLib/py_utils/py_mol/PyProtein.py')
#from ..IEProtLib.py_utils.py_mol import PyPeriodicTable, PyProtein

class data_preprocessing:
    def __init__(self,  result_path=None, TMscore_path=None, MSMS="msms",
                 w_dir="E:\PDBBind\Result\\test"):

        if result_path is not None:
            self.result_path = Path(result_path).resolve()
        # self.Pocket_path = Path(Pocket_path).resolve()
        # self.file_path = Path(file_path).resolve()
        self.msms = Path(MSMS).resolve()
        # self.check_file_exist()
        # self.check_file_format()
        self.w_dir = Path(w_dir).resolve()

    # def check_file_format(self):  #####file_path, Pocket_path,
    #     if self.file_path.suffix != ".pdb":
    #         print(self.file_path)
    #         raise ValueError("File does not exist!")
    #     return True
    #
    # def check_file_exist(self):
    #     if not self.file_path.exists():
    #         print(self.file_path)
    #         raise ValueError("File does not exist!")
    #     return True
    #
    #
    # def form_structure(self):
    #     parser = PDBParser()
    #     structure = parser.get_structure("1", self.file_path)
    #     return structure[0]



    def generate_hdf5(self):
        file_list = []
        for root, dirs, files in os.walk(self.w_dir + 'domain_pdb/'):
            for name in files:
                if name[:4] != None:
                    file_list.append(name)
        #co = 0
        positive_name_list = ''
        negative_name_list = ''
        for file_name in file_list:
            periodicTable_ = PyPeriodicTable()
            curProtein = PyProtein(periodicTable_)
            curProtein.load_molecular_file(self.w_dir + 'domain_pdb/' + file_name)
            file_path = self.w_dir+'domain_pdb/' + file_name
            print('file_path!!!!', file_path)
            curProtein.compute_covalent_bonds()
            curProtein.compute_hydrogen_bonds()
            if len(curProtein.aminoNeighs_) == 0 or len(curProtein.aminoNeighsHB_) == 0 or len(
                    curProtein.aminoType_) < 10:
                continue
            if not os.path.exists(self.w_dir + 'domain_hdf5/'):
                os.mkdir(self.w_dir + 'domain_hdf5/')
            curProtein.save_hdf5(self.w_dir + 'domain_hdf5/' + file_name.split('.')[0] + '.hdf5')
            if file_name.split('.')[0].split('_')[-1] == 'positive':
                positive_name_list += file_name.split('.')[0] + '\n'
            else:
                negative_name_list += file_name.split('.')[0] + '\n'
        f = open(self.w_dir + file_path.split('.')[-2].split('/')[-1] + '_positive_domain_list.txt', 'w')
        f.write(positive_name_list)
        f.close()
        f = open(self.w_dir + file_path.split('.')[-2].split('/')[-1] + '_negative_domain_list.txt', 'w')
        f.write(negative_name_list)
        f.close()
        f = open(self.w_dir + file_path.split('.')[-2].split('/')[-1] + '_domain_list.txt', 'w')
        f.write(positive_name_list + negative_name_list)
        f.close()

    def start_processing(self):
        # print('res_num', res_num)
        # preprocessed_data = self.SiteInfo_Process(res_num)
        # print("Preprocessing finished!")
        # positive_domain, negative_domain = self.form_residue_list(preprocessed_data)
        # print("Domain form finished!")
        # if positive_domain.__len__() > 0:
        #     self.from_domain_to_pdb(positive_domain, "positive")
        # if negative_domain.__len__() > 0:
        #     self.from_domain_to_pdb(negative_domain, "negative")
        # print("Domain PDB file form finished!")
        self.generate_hdf5(self)
        print('HDF5 file form finished!')
if __name__ == "__main__":
    start_processing()







