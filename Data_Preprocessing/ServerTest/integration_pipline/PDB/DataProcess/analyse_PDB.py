import Bio.PDB
import numpy as np
from Bio.PDB.ResidueDepth import *
from Bio.PDB.Polypeptide import is_aa
import PDB.PDBparser.ParserStructure
import pickle
import threading
import queue
import multiprocessing as mp
from PDB.DataProcess.PDB_info import PDBInfo

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import os
import PDB.Utilities.Files.FileSysDealer as fs
from PDB.DataProcess.site_info import Site_info


class Analyse:
    def __init__(self):
        self.site_list = []

    def load_pickle(self):
        f = open(r'H:\biodata\pdbtm\All_site.pickle', 'rb')
        self.site_list = pickle.load(f)

    def get_site_pickle(self, dir):
        filelist = []
        filelist = self.get_filelist(dir, filelist)
        print(len(filelist))
        site_list = []
        site_count = 0
        for file in filelist:
            print(file)
            site_info = Site_info(file)
            site_count += site_info.get_site_num()
            site_list += site_info.get_site_info()
            print(len(site_info.get_site_info()))
        self.site_list = site_list
        print(len(filelist))
        print(len(site_list))
        print(site_count)
        f = open(r'H:\biodata\pdbtm\All_site.pickle', 'wb')
        pickle.dump(site_list, f)
        f.close()

    def get_filelist(self, dir, Filelist):
        """get the file list under the direction"""
        newDir = dir
        if os.path.isfile(dir):
            suffix = fs.FilSysDealer.suffix(dir)
            if suffix == "pickle":
                Filelist.append(dir)
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                newDir = os.path.join(dir, s)
                self.get_filelist(newDir, Filelist)
        return Filelist

    def get_site_num(self):
        return len(self.site_list)

    def get_site_res_num(self):
        site_res_list = []
        for site in self.site_list:
            site_res_list.append(len(site))
        return site_res_list

    def get_res_num_plot(self, site_res_list):
        xrange = range(min(site_res_list), max(site_res_list)+1)
        x = []
        y = []
        site_count = 0
        print(len(site_res_list))
        print(site_count)
        for i in xrange:
            x.append(i)
            y.append(round(site_res_list.count(i)/len(site_res_list), 4))
        print(x)
        print(y)



        plt.xlim(-.5, max(site_res_list)+3)
        plt.ylim((-.025, 0.15))
        plt.bar(x, y)
        for x, y in zip(x, y):
            plt.text(x, y + 0.002, '%.3f' % y, ha='center', va='bottom', fontsize=7)
            plt.text(x, 0 - 0.005, '%d' % x, ha='center', va='bottom', fontsize=7)
        plt.show()
        print(mpl.get_backend())

    def get_res_depth_plot(self):
        depth_list = []
        res_count = 0
        for site in self.site_list:
            # site_depth_list = []
            # center_depth = 0
            # for res in site:
            #     if res['relative_ca_coord'].all() == 0:
            #         center_depth = round(res['ca_depth'], 3)
            #     site_depth_list.append(round(res['ca_depth'], 3))
            # for depth in site_depth_list:
            #     depth_list.append(abs(depth - center_depth))
            for res in site:
                depth_list.append(res['ca_depth'])
        print(depth_list)
        print(len(depth_list))
        xrange = np.arange(0, 20, 0.5)
        x = []
        y = []
        for i in xrange:
            x.append(i + 0.25)
            y_count = 0
            for depth in depth_list:
                if depth >= i and depth < i + 0.5:
                    y_count += 1
            y.append(y_count / len(depth_list))

        xticks = np.linspace(0, 20, 41)
        plt.xticks(xticks)
        plt.xlim(0, 20)
        yticks = np.linspace(0, 1, 31)
        plt.yticks(yticks)
        plt.ylim(-.01, max(y) +0.05)
        plt.bar(x, y, width=0.3)
        for x, y in zip(x, y):
            plt.text(x, y + 0.005, '%.3f' % y, ha='center', va='bottom', fontsize=9)
        label = 'This is the depth statistics for all sites'
        plt.title(label, loc='center', fontdict={'fontsize': 20})
        path = r'H:\biodata\pdbtm\result\site_res_all_' + ' depth.png'
        print(path)
        plt.savefig(path)
        plt.show()
        # print(xrange)

    def get_res_dis_plot(self):
        dis_list = []
        for site in self.site_list:
            for res in site:
                dis_list.append(res['ca_dis'])
        print(max(dis_list))

        xrange = np.arange(0, 15, 0.5)
        x = []
        y = []
        for i in xrange:
            x.append(i)
            y_count = 0
            for depth in dis_list:
                if depth <= i:
                    y_count += 1
            y.append(y_count / len(dis_list))

        xticks = np.linspace(0, 15, 31)
        plt.xticks(xticks)
        plt.xlim(0, 15)
        yticks = np.linspace(0, 1, 31)
        plt.yticks(yticks)
        plt.ylim(-.01, 1)
        plt.plot(x, y)
        for x, y in zip(x, y):
            plt.text(x, y + 0.005, '%.3f' % y, ha='center', va='bottom', fontsize=9)
        plt.show()
        # print(xrange)

    def n_res_site_ca_depth_plot(self, n):
        n_site_res = []
        for site in self.site_list:
            if len(site) == n:
                n_site_res.append(site)
        print(len(n_site_res))
        depth_list = []
        for site in n_site_res:
            for res in site:
                depth_list.append(res['relative_ca_depth'])
        print(depth_list)
        xrange = np.arange(0, 10, 0.5)
        x = []
        y = []
        for i in xrange:
            x.append(i+0.25)
            y_count = 0
            for depth in depth_list:
                if depth >= i and depth < i + 0.5:
                    y_count += 1
            y.append(y_count / len(depth_list))
        plt.figure(figsize=(15, 10))
        xticks = np.linspace(0, 10, 21)
        plt.xticks(xticks)
        plt.xlim(-.1, 10)
        plt.ylim(-.01, max(y) +0.05)
        plt.bar(x, y, width=0.3)
        for x, y in zip(x, y):
            plt.text(x, y + 0.005, '%.3f' % y, ha='center', va='bottom', fontsize=9)
        label = 'This is the relative depth statistics for sites containing ' + str(n) + ' residues'
        plt.title(label, loc='center', fontdict={'fontsize': 20})
        path = r'H:\biodata\pdbtm\result\site_res' + str(n) + ' depth.png'
        print(path)
        plt.savefig(path)
        # plt.show()

    def get_cover_rate(self, site_res_num, get_num, max_depth):
        total_correct_count = 0
        total_count = 0
        filelist = []
        filelist = self.get_filelist(r'H:\biodata\pdbtm\pdb_all', filelist)
        # print(len(filelist))
        site_list = []
        site_count = 0
        for file in filelist:
            # print(file)
            site_info = Site_info(file)
            correct_count, count = site_info.test_cover_rate(site_res_num, get_num, max_depth)
            total_correct_count += correct_count
            total_count += count
        # print('total', total_count)
        # print('correct', total_correct_count)
        return total_correct_count, total_count

    def get_cover_rate_plot(self, get_num, max_depth):
        print(get_num, max_depth)
        xrange = np.arange(4, 11, 1)
        x = []
        y = []
        count = 0
        for i in xrange:
            count += 1
            x.append(count)
            total_correct_count, total_count = self.get_cover_rate(i, get_num, max_depth)
            y.append(total_correct_count / total_count)
            print('site_res', i)
            print(total_correct_count)
            print(total_count)
            print(total_correct_count / total_count)
        path = r'H:\biodata\pdbtm\res4-10_num' + str(get_num) + '_depth' + str(max_depth) + '.pickle '
        f = open(path, 'wb')
        pickle.dump((x, y), f)
        f.close()

    def positive_set_count(self, max_numres):
        filelist = []
        filelist = self.get_filelist(r'H:\biodata\pdbtm\pdb_all', filelist)
        error_list = []
        total_site_num = 0
        site_num = 0
        for file in filelist:
            try:
                site_info = Site_info(file)
                all_site_count, site_count = site_info.positive_set_count(max_numres)
                total_site_num += all_site_count
                site_num += site_count
                print(file)
            except Exception:
                error_list.append(file)
        print(total_site_num)
        print(site_num)

    def get_positive_data(self, num, max_depth, max_numres):
        filelist = []
        filelist = self.get_filelist(r'H:\biodata\pdbtm\pdb_all', filelist)
        count = 0
        error_list = []
        for file in filelist:
            # print(file)
            filepath, filename = os.path.split(file)
            path = r'H:\biodata\pdbtm\positive_dataset_pickle\\' + filename
            # if os.path.exists(path):
            #     continue
            # try:
            #     site_info = Site_info(file)
            #     len = site_info.positive_set(num, max_depth, max_numres)
            #     print(file)
            #     print(len)
            #     count += len
            # except Exception:
            #     error_list.append(file)
            site_info = Site_info(file)
            len = site_info.positive_set(num, max_depth, max_numres)
            print(file)
            print(len)
            count += len

        print(count)
        for error in error_list:
            print(error)

    def negative_data(self, file, num, max_depth):
        filepath, filename = os.path.split(file)
        path = r'H:\biodata\pdbtm\negative_dataset2_pickle\\' + filename
        if os.path.exists(path):
            pass
        else:
            try:
                site_info = Site_info(file)
                site_info.negative_set(num, max_depth)
            except Exception:
                pass

    def dataset(self, file, num, max_depth, max_numres):

        filepath, filename = os.path.split(file)
        path = r'H:\biodata\pdbtm\dataset\\' + filename
        if os.path.exists(path):
            pass
        else:
            try:
                site_info = Site_info(file)
                site_info.get_dataset(num, max_depth, max_numres)
            except Exception:
                pass


def multiP(file_queue):
    while not file_queue.empty():
        file = file_queue.get()
        analyse = Analyse()
        analyse.dataset(file, 25, 5.5, 10)


if __name__ == '__main__':
    analyse = Analyse()
    analyse.load_pickle()
    # analyse.get_res_depth_plot()

    # analyse.positive_set_count(10)
    file_list = []
    file_list = analyse.get_filelist(r'H:\biodata\pdbtm\pdb_all', file_list)
    file_queue = mp.Queue()
    print(len(file_list))
    for file in file_list:
        file_queue.put(file)
    process_list = []
    for i in range(1, 5):
        print(i)
        process = mp.Process(target=multiP, args=(file_queue, ))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()

    # analyse.get_positive_data(25, 5.5, 10)
    # analyse.get_site_pickle(r'H:\biodata\pdbtm\pdb_all')
    # analyse.load_pickle()
    # for i in range(1, 15):
    #     analyse.n_res_site_ca_depth_plot(i)

    # site_res_list = analyse.get_site_res_num()
    # analyse.get_res_num_plot(site_res_list)
    # print(len(site_res_list))
    # print(max(site_res_list))
    # analyse.get_res_depth_plot()
