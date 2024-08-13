# -*- coding:utf-8 -*-
'''
	File Name：     testDataCapture
	Description :   download , unzip all the pdb files and count how many files there are 
	Author :        Liu Zhe (Based on the work of Gong Jianting and Zhai Yuhao)
	date：          2018/11/26
'''
import logging,datetime,ConfigDealer
from sh import rsync
import sys
import time
import os
from pdbDownloader import *
from Unzipper import *
from Counter import *

if __name__ == "__main__": 
    
    cfg = ConfigDealer.ConfigDealer()   
    print(cfg.getConfigDictionary())
    download = PDBDownloader(cfg.getConfigDictionary(),'PDBparser')
    download.run()
    
    rootdir = "/home/RaidDisk/pdbfiles/pdb/pdb/"
    goaldir = "/home/RaidDisk/pdbfiles/updb/pdb/"    
    unzip = Unzipper(rootdir,goaldir)
    unzip.run()
    
    pdbcount = Counter(r'/home/RaidDisk/pdbfiles/pdb/pdb')
    pdbcount.run()
    updbcount = Counter(r'/home/RaidDisk/pdbfiles/updb/pdb')
    updbcount.run()    