import pickle
import os
import PDB.Utilities.Files.FileSysDealer as fs
import sys
import numpy as np
from pickle import UnpicklingError
import random

standard_aa_names = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS",
                     "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL",
                     "TRP", "TYR"]
aa3 = standard_aa_names
d3_to_index = {}

for i in range(0, 20):
    n3 = aa3[i]
    d3_to_index[n3] = i


def get_filelist( dir, Filelist):
    """get the file list under the direction"""
    newDir = dir
    if os.path.isfile(dir):
        suffix = os.path.splitext(dir)[1]
        if suffix == ".pickle":
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist


def get_all_dataset(path, ispositive):
    filelist = []
    filelist = get_filelist(path, filelist)
    print(len(filelist))
    positive_data_set = []
    for file in filelist:
        data_list = get_dataset(file, ispositive)
        positive_data_set += data_list
    print(len(positive_data_set))
    f = open(r'H:\biodata\pdbtm\all_data\positive_data37950.pickle', 'wb')
    pickle.dump(positive_data_set, f)
    f.close()


def res_name_number(res_name):
    return d3_to_index[res_name]


def get_dataset(path):
    data_list = []
    f = open(path, 'rb')
    print(path)
    site_list = pickle.load(f)
    f.close()
    for site in site_list:
        res_list = []
        for res in site['res_info']:
            res_info = [res['relative_ca_coord'], res_name_number(res['res_name'])]
            res_list.append(res_info)
        ispositive = site['isPositive']
        site_list = [res_list, ispositive]
        if len(site_list[0]) != 25:
            continue
        data_list.append(site_list)
    print(path)
    print(len(data_list))
    return data_list


def random_sample(path):
    f = open(path, 'rb')
    negative_data = pickle.load(f)
    print(len(negative_data))
    f.close()
    rs_data = random.sample(negative_data, 52050)
    f = open(r'H:\biodata\pdbtm\all_data\negative_data52050.pickle', 'wb')
    pickle.dump(rs_data, f)
    f.close()


# print(res_name_number('LYS'))

# get_all_dataset(r'H:\biodata\pdbtm\positive_dataset_pickle', 1)
# # get_dataset(r'H:\biodata\pdbtm\positive_dataset_pickle\1a0s.pdb.pickle', 1)
# random_sample(r'H:\biodata\pdbtm\all_data\negative_data143927.pickle')