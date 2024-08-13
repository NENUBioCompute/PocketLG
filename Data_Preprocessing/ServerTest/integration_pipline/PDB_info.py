# @Time    : 2022/8/24 13:11
# @Author  : Yihang Bao
# @FileName: PDB_info.py.py
# @Description:


import Bio.PDB
import numpy as np
import math
import os

# import Bio.PDBparser.ResidueDepth as rd
from Bio.PDB.ResidueDepth import *
from Bio.PDB.Polypeptide import is_aa
import time
import PDB.PDBparser.ParserStructure
import pickle
import threading
import queue
import multiprocessing as mp
import PDB
import warnings
from Bio.PDB.PDBParser import PDBConstructionWarning

# 在解析PDB文件之前添加以下代码以忽略警告
warnings.simplefilter("ignore", PDBConstructionWarning)


def takeSecond(elem):
    return elem[1]


def parse(file):
    parser = Bio.PDB.PDBParser(PERMISSIVE=1)
    structure = parser.get_structure("test", file)
    return structure


# PDBinfo类负责PDB的数据解析，通过调用PDBparser和Biopython完成数据解析并以pickle形式存储
class PDBInfo:
    structure = None
    site_info = None
    PDB_id = None

    def __init__(self, file, MSMS="msms"):
        self.msms = MSMS
        structure = parse(file)
        surface = get_surface(structure[0],MSMS=str(self.msms))
        residues = structure[0].get_residues()
        for residue in residues:
            residue.xtra["coord_center"] = self.res_coord_center(residue)
            residue.xtra["res_depth"] = residue_depth(residue, surface)
            residue.xtra["ca_depth"] = ca_depth(residue, surface)
            residue.xtra["ca_coord"] = self.ca_coord(residue)
        self.structure = structure
        self.site_info = self.get_site(file)
        self.PDB_id = file[-8:-4]

    def get_site(self, file):
        ps = PDB.PDBparser.ParserStructure.ParserStructure()
        dic = ps.parse(file)
        site_info = []
        for site in dic["SITE"]:
            site_res = []
            count = 0
            for res_info in site["resName"]:
                res_list = Bio.PDB.Selection.unfold_entities(self.structure[0][res_info["chainID"]], "R")
                for res in res_list:
                    if (
                        res_info["resName"] == res.get_resname().strip()
                        and int(res_info["seq"]) == res.get_id()[1]
                        and res_info["iCode"] == res.get_id()[2].strip()
                    ):
                        if is_aa(res, standard=True):
                            site_res.append(res)
            site["site_res"] = site_res
            site_info.append(site)
        return site_info
        # site_dict = DataProcess.parse_site.get_site_info(file)
        # site_info = {}
        # for site, seqs in site_dict.items():
        #     site_res_list = []
        #     for seq in seqs:
        #         res_list = Bio.PDBparser.Selection.unfold_entities(self.structure[0][seq[1]], 'R')
        #         for res in res_list:
        #             if seq[0] == res.get_id()[1] and seq[2] == res.get_id()[2]:
        #                 seq[0] = (res.get_id()[0], res.get_id()[1], seq[2])
        #                 if is_aa(res):
        #                     site_res_list.append(res)
        #     site_info[site] = site_res_list
        # return site_info

    # def get_radius(self):
    #     radius_list = []
    #     for site, res_list in self.site_info.items():
    #         if len(res_list) == 0:
    #             continue
    #         site_center, radius = self.site_coord_center(res_list)
    #         # radius = math.ceil(radius)
    #         radius_list.append(radius)
    #     return radius_list
    #
    # def get_distance(self, coord1, coord2):
    #     diff = coord1 - coord2
    #     return numpy.sqrt(numpy.dot(diff, diff))
    #
    # def site_coord_center(self, res_list):
    #     site_center = np.array([0.0, 0.0, 0.0])
    #     coord_list = []
    #     for res in res_list:
    #         coord = res.xtra["coord_center"]
    #         coord_list.append(coord)
    #         site_center += coord
    #     site_center = site_center/len(res_list)
    #     radius = 0.0
    #     for coord in coord_list:
    #         distance = self.get_distance(coord, site_center)
    #         if distance > radius:
    #             radius = distance
    #     return site_center, radius
    #
    def res_coord_center(self, res):
        coord = np.array([0.0, 0.0, 0.0])
        len = 0
        atom_list = res.get_atoms()
        for atom in atom_list:
            len += 1
            coord += atom.get_coord()
        coord /= len
        return coord

    def ca_coord(self, res):
        if res.has_id("CA"):
            ca = res["CA"]
            return ca.get_coord()
        else:
            return None

    # def get_depth(self):
    #     depth_list = []
    #     for site, res_list in self.site_info.items():
    #         depth = 0
    #         for res in res_list:
    #             if depth < res.xtra["res_depth"]:
    #                 depth = res.xtra["res_depth"]
    #         depth_list.append(depth)
    #     return depth_list
    #
    # def get_close_res(self, n, max_depth, coord_center):
    #     res_list = []
    #     distance_list = []
    #     for res in Bio.PDBparser.Selection.unfold_entities(self.structure[0], "R"):
    #         if is_aa(res):
    #             res_center = self.res_coord_center(res)
    #             distance = self.get_distance(res_center, coord_center)
    #             distance_list.append((res, distance))
    #     distance_list.sort(key=takeSecond)
    #     for res in distance_list:
    #         print(res, file=data)
    #         print(res[0].xtra["res_depth"],file=data)
    #     i = 0
    #     while n > 0:
    #         if distance_list[i][0].xtra["res_depth"] <= max_depth:
    #             res_list.append(distance_list[i][0])
    #             n -= 1
    #         i += 1
    #     return res_list


# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     file_path = "/home/baoyihang/TMPBDSniffer/integration_pipline/3bs0.pdb"
#     res_info = PDBInfo(file_path, MSMS="msms")
#     picklePath = file_path + ".pickle"
#     f = open(picklePath, "wb")
#     pickle.dump(res_info, f)
#     f.close()


# data = open(r"F:githubmy_codepdb118l.txt", "w")
# res_info = PDBInfo(r"H:iodataPDBfilepdbtest28pdb118l.ent")
# for site, res_list in res_info.site_info.items():
#     print(res_list, file=data)
#     center, radius = res_info.site_coord_center(res_list)
#     print(radius, file=data)
#     result_list = res_info.get_close_res(10, 6, center)
# res_info.print_center()
