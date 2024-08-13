# -*- coding:utf-8 -*-
'''
	File Name：     Counter
	Description :   count and print files&directories numbers
	Author :        Liu Zhe
	date：          2018/11/20
'''
import os
import time
import datetime
import argparse

class Counter(object):
    def __init__(self, path):  
        self.path = path      
    totalSize = 0
    fileNum = 0
    dirNum = 0
    
    def visitDir(self,path):
       
        for lists in os.listdir(path):
            sub_path = os.path.join(path, lists)
            #print(sub_path)
            #print(time.ctime(os.stat(sub_path).st_mtime)) # the time when this file was modified
            #print(time.ctime(os.stat(sub_path).st_ctime)) # the time when this file was created
            #f.write(sub_path+'\n')
            #f.write(time.ctime(os.stat(sub_path).st_mtime)+'\n')
            #f.write(time.ctime(os.stat(sub_path).st_ctime)+'\n') 
            if os.path.isfile(sub_path):
                self.fileNum = self.fileNum + 1                      # count all files
                self.totalSize = self.totalSize+os.path.getsize(sub_path)  # size of all files
            elif os.path.isdir(sub_path):
                self.dirNum = self.dirNum + 1                       # count all directories
                self.visitDir(sub_path)                           # recursively traversal of subfolders
    
    
    def sizeConvert(self,size):                                   # unit conversion
        K, M, G = 1024, 1024**2, 1024**3
        if size >= G:
            return str(size/G)+'G Bytes'
        elif size >= M:
            return str(size/M)+'M Bytes'
        elif size >= K:
            return str(size/K)+'K Bytes'
        else:
            return str(size)+'Bytes'        
    
    def run(self):
        if not os.path.isdir(self.path):
            print('Error:"', self.path, '" is not a directory or does not exist.')
            return
        self.visitDir(self.path)        
        f = open("count_doc.txt","a+")       # write appended
        print('The total size of '+self.path+' is:'+self.sizeConvert(self.totalSize))
        print('The total number of files in '+self.path+' is:',self.fileNum)
        print('The total number of directories in '+self.path+' is:',self.dirNum)
        f.write(str(datetime.datetime.now()))
        f.write('The total size of '+self.path+' is:'+self.sizeConvert(self.totalSize)+'\n')
        f.write('The total number of files in '+self.path+' is:'+str(self.fileNum)+'\n')
        f.write('The total number of directories in '+self.path+' is:'+str(self.dirNum)+'\n')     

if __name__ == "__main__":
    
    """
    pdbcount = Counter(r'/home/RaidDisk/pdbfiles/pdb/pdb')
    pdbcount.run()
    updbcount = Counter(r'/home/RaidDisk/pdbfiles/updb/pdb')
    updbcount.run() 
    
    cmd:
    python3 Counter.py -p /home/RaidDisk/pdbfiles/pdb/pdb
    python3 Counter.py -p /home/RaidDisk/pdbfiles/updb/pdb
    python3 Counter.py -p /home/RaidDisk/pdbfiles/picklefiles
    """
    #set the parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="input the path you want to know how many files in there")
   
    args = vars(ap.parse_args())
    
    #fetch the argues and print them
    print(args['path'])
    
    count = Counter(args['path'])
    count.run()

        
           