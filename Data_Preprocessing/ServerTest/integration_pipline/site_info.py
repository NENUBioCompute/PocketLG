# @Time    : 2022/8/24 11:02
# @Author  : Yihang Bao
# @FileName: site_info.py
# @Description:

import os
import sys

import Bio.PDB
import numpy as np
from Bio.PDB.ResidueDepth import *
from Bio.PDB.Polypeptide import is_aa
import PDB.PDBparser.ParserStructure
from PDB.PDBparser.CallParser import callParser
import pickle
import threading
import queue
import multiprocessing as mp
from PDB.DataProcess.PDB_info import PDBInfo
import random
import warnings
from Bio.PDB.PDBParser import PDBConstructionWarning

# 在解析PDB文件之前添加以下代码以忽略警告
warnings.simplefilter("ignore", PDBConstructionWarning)

# site_info类用于处理单个数据，包含各种处理单个数据的方法
class Site_info:
    def __init__(self, PDB_info, pdb_path):
        self.PDB_info = PDB_info
        self.pdb_path = pdb_path
        self.add_site_binding()
        self.site_res_check()
        self.get_site_center_res_coord()

    def site_res_check(self):
        del_site_list = []
        for site in self.PDB_info.site_info:
            del_res_list = []
            for res in site["site_res"]:
                if res.xtra["ca_coord"] is None:
                    del_res_list.append(res)
            for res in del_res_list:
                site["site_res"].remove(res)
            if len(site["site_res"]) == 0:
                del_site_list.append(site)
        for site in del_site_list:
            self.PDB_info.site_info.remove(site)

    def add_site_binding(self):
        callparser = callParser()
        ps = callparser.parser(self.pdb_path)
        content = None
        for remark in ps["REMARK"]:
            if remark["remarkNum"] == 800:
                content = remark["empty"]
        if content is not None:
            for i, line in enumerate(content):
                if line[0:15] == "SITE_IDENTIFIER":
                    SITE_IDENTIFIER = line[17:20]
                    if content[i + 1][:13] == "EVIDENCE_CODE":
                        EVIDENCE_CODE = content[i + 1][15::]
                        if content[i + 2][:16] == "SITE_DESCRIPTION":
                            SITE_DESCRIPTION = content[i + 2][18::]
                        else:
                            SITE_DESCRIPTION
                    elif content[i + 1][:16] == "SITE_DESCRIPTION":
                        SITE_DESCRIPTION = content[i + 1][18::]
                        EVIDENCE_CODE = ""
                    else:
                        EVIDENCE_CODE = ""
                        SITE_DESCRIPTION = ""

                    for site in self.PDB_info.site_info:
                        if site["siteID"] == SITE_IDENTIFIER:
                            site["EVIDENCE_CODE"] = EVIDENCE_CODE
                            site["SITE_DESCRIPTION"] = SITE_DESCRIPTION

    def get_site_num(self):
        return len(self.PDB_info.site_info)

    def get_site_res_num(self):
        site_res_list = []
        for site in self.PDB_info.site_info:
            site_res_list.append(len(site["site_res"]))
        return site_res_list

    def get_site_info(self):
        site_list = []
        for i, site in enumerate(self.PDB_info.site_info):
            if len(site["site_res"]) == 0:
                continue
            site_res_coord = self.get_site_res_coord(site["site_res"])
            site_res_coord = self.get_relative_dis(site_res_coord, site["center_res_ca_coord"])
            site_res_coord = self.get_relative_depth(site_res_coord, site["center_res_ca_depth"])
            site_res_coord = self.get_res_distances(site_res_coord)
            site_list.append(site_res_coord)
        # for site in site_list:
        #     for res in site:
        #         print(res)
        return site_list

    def get_site_res_info(self, res_list, center_res_ca_coord, center_res_ca_depth):
        site_res_coord = self.get_site_res_coord(res_list)
        site_res_coord = self.get_relative_dis(site_res_coord, center_res_ca_coord)
        site_res_coord = self.get_relative_depth(site_res_coord, center_res_ca_depth)
        site_res_coord = self.get_res_distances(site_res_coord)
        return site_res_coord

    def get_site_res_coord(self, res_list):
        site_res_coord = []
        for res in res_list:
            res_info_dict = {}
            res_info_dict["coord_center"] = res.xtra["coord_center"]
            res_info_dict["ca_coord"] = res.xtra["ca_coord"]
            res_info_dict["res_depth"] = res.xtra["res_depth"]
            res_info_dict["ca_depth"] = res.xtra["ca_depth"]
            res_info_dict["res_name"] = res.get_resname()
            site_res_coord.append(res_info_dict)
        return site_res_coord

    def get_relative_dis(self, site_res_coord, center_res_ca_coord):
        for res in site_res_coord:
            res["relative_ca_coord"] = res["ca_coord"] - center_res_ca_coord
        return site_res_coord

    def get_relative_depth(self, site_res_coord, center_res_ca_depth):
        for res in site_res_coord:
            res["relative_ca_depth"] = abs(res["ca_depth"] - center_res_ca_depth)
        return site_res_coord

    def get_res_distances(self, site_res_coord):
        center_coord = np.array([0.0, 0.0, 0.0])
        res_dis_list = []
        ca_dis_list = []
        for res in site_res_coord:
            # res_dis = self.get_distance(res['relative_coord_center'], center_coord)
            ca_dis = self.get_distance(res["relative_ca_coord"], center_coord)
            # res_dis_list.append(res_dis)
            # res['res_dis'] = res_dis
            res["ca_dis"] = ca_dis
            ca_dis_list.append(ca_dis)
        # site['res_dis'] = res_dis_list
        # site['ca_dis'] = ca_dis_list
        return site_res_coord

    def get_site_center_res_coord(self):
        for site in self.PDB_info.site_info:
            if len(site["site_res"]) == 0:
                continue
            ca_center = np.array([0.0, 0.0, 0.0])
            for res in site["site_res"]:
                # print(type(ca_center))
                # print(type(res.xtra['ca_coord']))
                ca_center += res.xtra["ca_coord"]
            ca_center /= len(site["site_res"])
            res_dis_list = []
            for res in site["site_res"]:
                dis = self.get_distance(ca_center, res.xtra["ca_coord"])
                res_dis_list.append((res, dis))
            res_dis_list.sort(key=lambda x: x[1])
            center_res_ca_coord = res_dis_list[0][0].xtra["ca_coord"]
            site["center_res_ca_coord"] = center_res_ca_coord
            center_res_ca_depth = res_dis_list[0][0].xtra["ca_depth"]
            site["center_res_ca_depth"] = center_res_ca_depth
            # print(center_res_ca_coord)

    # input  res_list return res_center_coord res_center_depth
    def _get_center_depth(self, res_list):
        ca_center = np.array([0.0, 0.0, 0.0])
        for res in res_list:
            # print(type(ca_center))
            # print(type(res.xtra['ca_coord']))
            ca_center += res.xtra["ca_coord"]
        ca_center /= len(res_list)
        res_dis_list = []
        for res in res_list:
            dis = self.get_distance(ca_center, res.xtra["ca_coord"])
            res_dis_list.append((res, dis))
        res_dis_list.sort(key=lambda x: x[1])
        center_res_ca_coord = res_dis_list[0][0].xtra["ca_coord"]
        center_res_ca_depth = res_dis_list[0][0].xtra["ca_depth"]
        return center_res_ca_depth, center_res_ca_coord

    def get_close_res(self):
        for site in self.PDB_info.site_info:
            print("site")
            center_coord = site["center_res_ca_coord"]
            res_list = self._get_close_res(center_coord, 15, 5)
            for res in res_list:
                print(res.get_id())
            print("site_res")
            for res in site["site_res"]:
                if res in res_list:
                    print("yes")
                else:
                    print("no")

    def test_cover_rate(self, site_res, n, max_depth):
        correct_count = 0
        total_count = 0
        for site in self.PDB_info.site_info:
            if len(site["site_res"]) == site_res:
                center_coord = site["center_res_ca_coord"]
                for res in site["site_res"]:
                    if all(res.xtra["ca_coord"] == center_coord):
                        center_res_depth = res.xtra["ca_depth"]
                res_list = self._get_close_res(center_coord, n, max_depth, center_res_depth)
                for res in site["site_res"]:
                    total_count += 1
                    if res in res_list:
                        correct_count += 1
        return correct_count, total_count

    def _get_close_res(self, center_coord, n, max_depth, center_res_depth, res_list):
        res_dis_list = []
        for res in self.PDB_info.structure[0].get_residues():
            if not is_aa(res, standard=True):
                continue
            if res.xtra["ca_coord"] is None:
                continue
            dis = self.get_distance(center_coord, res.xtra["ca_coord"])
            res_dis_list.append((res, dis))
        res_dis_list.sort(key=lambda x: x[1])
        count = n
        for res_dis in res_dis_list:
            relative_depth = abs(res_dis[0].xtra["ca_depth"] - center_res_depth)
            if count > 0 and relative_depth <= max_depth and res_dis[0] not in res_list:
                res_list.append(res_dis[0])
                count -= 1
            else:
                continue
        return res_list

    def positive_set_count(self, max_numres):
        all_site_count = 0
        site_count = 0
        for site in self.PDB_info.site_info:
            all_site_count += 1
            if site["numRes"] > max_numres:
                pass
            else:
                site_count += 1
        print(site_count)
        return all_site_count, site_count

    def get_positive_set(self, num, max_depth, max_numres):
        set_list = []
        for site in self.PDB_info.site_info:
            if site["numRes"] > max_numres:
                continue
            site_dict = {}
            count = int(num)
            res_list = []
            center = site["center_res_ca_coord"]
            center_res_depth = site["center_res_ca_depth"]
            for res in site["site_res"]:
                res_list.append(res)
                count -= 1
            res_list = self._get_close_res(center, count, max_depth, center_res_depth, res_list)
            site_dict["siteID"] = site["siteID"]
            site_dict["numRes"] = site["numRes"]
            site_dict["site_res"] = res_list
            site_dict["EVIDENCE_CODE"] = site["EVIDENCE_CODE"]
            site_dict["SITE_DESCRIPTION"] = site["SITE_DESCRIPTION"]
            site_dict["center_res_ca_coord"] = center
            site_dict["center_res_ca_depth"] = center_res_depth
            site_dict["res_info"] = self.get_site_res_info(res_list, center, center_res_depth)
            site_dict["isPositive"] = 1
            set_list.append(site_dict)
        return set_list

    def get_negative_set(self, num, max_depth, max_numres):
        set_list = []
        exist_res_list = []
        for site in self.PDB_info.site_info:
            if site["numRes"] > max_numres:
                continue
            count = num
            res_list = []
            center = site["center_res_ca_coord"]
            center_res_depth = site["center_res_ca_depth"]
            for res in site["site_res"]:
                res_list.append(res)
                count -= 1
            res_list = self._get_close_res(center, count, max_depth, center_res_depth, res_list)
            exist_res_list += res_list
        for res in self.PDB_info.structure[0].get_residues():
            if res in exist_res_list or not is_aa(res, standard=True):
                continue
            center_res_ca_coord = res.xtra["ca_coord"]
            center_res_ca_depth = res.xtra["ca_depth"]
            res_list = []
            site_res_list = self._get_close_res(center_res_ca_coord, num, max_depth, center_res_ca_depth, res_list)
            is_available_flag = True
            for site_res in site_res_list:
                if site_res in exist_res_list:
                    is_available_flag = False
            if is_available_flag:
                site_dict = {}
                # exist_res_list += site_res_list
                site_dict["site_res"] = res_list
                site_dict["center_res_ca_coord"] = center_res_ca_coord
                site_dict["center_res_ca_depth"] = center_res_ca_depth
                site_dict["res_info"] = self.get_site_res_info(res_list, center_res_ca_coord, center_res_ca_depth)
                site_dict["isPositive"] = 0
                cmp_flag = True
                if len(res_list) != num:
                    cmp_flag = False
                for other_site_dict in set_list:
                    if len(set(other_site_dict["site_res"]).difference(set(site_dict["site_res"]))) == 0:
                        cmp_flag = False
                if cmp_flag:
                    set_list.append(site_dict)
        # set_list = random.sample(set_list, int(len(set_list)/25)+1)
        # print(len(set_list))
        return set_list

    def get_dataset(self, num, max_depth, max_numres):
        positive_set = self.get_positive_set(num, max_depth, max_numres)
        negative_set = self.get_negative_set(num, max_depth, max_numres)
        dataset = positive_set + negative_set
        if len(dataset) == 0:
            return 0

        return dataset

    def get_distance(self, coord1, coord2):
        diff = coord1 - coord2
        return numpy.sqrt(numpy.dot(diff, diff))


# if __name__ == "__main__":
#     filepath = "/home/baoyihang/TMPBDSniffer/integration_pipline/3bs0.pdb.pickle"
