# -*- coding:utf-8 -*-
'''
	File Name：     Unzipper
	Description :   unzip all the files
	Author :        Liu Zhe (Based on the work of Gong Jianting and Zhai Yuhao)
	date：          2018/11/26
'''
import os
import json
import DataStorage as DS
import time, os,datetime,logging,gzip,configparser,pickle
 
class Unzipper(object):
    def __init__(self, rootdir,goaldir):  
        self.rootdir = rootdir
        self.goaldir = goaldir
    def  UnzipFile(self,fileName,unzipFileName):
        """"""
        try:
            zipHandle = gzip.GzipFile(mode='rb',fileobj = open(fileName,'rb'))
            with open(unzipFileName, 'wb') as fileHandle:
                fileHandle.write(zipHandle.read())
    
        except IOError:
            raise("Unzip file failed!")  
        
    def mkdir(self,path):
        #Created uncompress path folder
        isExists=os.path.exists(path)
        if not isExists:
            os.makedirs(path)
            print(path + " Created folder sucessful!")
            return True
        else:  
            #print ("this path is exist")
            return False       
    
    def run(self):
        for parent,dirnames,filenames in os.walk(self.rootdir):
            for filename in filenames:
                #start unzip,get the target name and make a files
                dirname = filename[4:6]
                filename_with_rootdir = self.rootdir + str(dirname) + '/' + str(filename)
                unzipFileName = self.goaldir + str(dirname)+ '/' +str(filename)
                self.mkdir(self.goaldir + str(dirname) + '/') 
                try:
                    self.UnzipFile(filename_with_rootdir,unzipFileName[:-3])
                    pass
                except:
                    print(filename_with_rootdir)
                    continue        
                #print("unzip done")
                
if __name__ == "__main__":
    rootdir = "/home/RaidDisk/pdbfiles/pdb/pdb/"
    goaldir = "/home/RaidDisk/pdbfiles/updb/pdb/"    
    unzip = Unzipper(rootdir,goaldir)
    unzip.run()     

