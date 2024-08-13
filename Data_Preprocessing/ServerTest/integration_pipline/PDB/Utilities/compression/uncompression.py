"""
  Author:  WX.Wang
  Purpose: Offer the most common methods uncompress files
  Created: 5/2/2019
"""
import os
import gzip
import Utilities.Files.FileSysDealer as fs
import numpy

def un_gz(file_name):
    """ungz zip file"""
    f_name = file_name.replace(".gz", "")
    g_file = gzip.GzipFile(file_name)
    open(f_name, "wb+").write(g_file.read())
    g_file.close()


def get_filelist(dir,Filelist):
    """get the file list under the direction"""
    newDir = dir
    if os.path.isfile(dir):
        suffix = fs.FilSysDealer.suffix(dir)
        if suffix == "gz":
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist


def un_gz_all(dir):
    """uncompress all files under the direction"""
    filelist = []
    get_filelist(dir, filelist)
    count = 0
    print(dir)
    print(filelist)
    for file in filelist:
        un_gz(file)
        count += 1
        print(file)
    print(count)



un_gz_all(r"H:\biodata\pdbtm\pdb_all")
