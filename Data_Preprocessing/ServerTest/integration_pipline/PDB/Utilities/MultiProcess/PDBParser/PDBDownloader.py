# -*- coding:utf-8 -*-
'''
	File Name：     pdbDownloader
	Description :   download all the pdb files from the website
	Author :        Liu Zhe (Based on the work of Gong Jianting and Zhai Yuhao)
	date：          2018/11/26
'''
import time, os,datetime,logging,gzip,configparser,pickle
import logging,ConfigDealer
from sh import rsync
import sys
import time
import os

class DownloadBase(object):

    __configDict = {}


    def __init__(self,configDict,targetDB):

        self.__configDict = configDict
        self.__targetDB = targetDB

        #----------------------------------------------------------------------
    def goNextStep(self):
        """
          1.  write back information to config file
          2. initial next step according to config file
        """
        #self.writeBackToConfig()
        nextStep = self.__configDict[self.__targetDB]['Download']['nextStep']
        moduleName,ext = os.path.splitext(os.path.basename(self.__configDict[self.__configDict][nextStep]['modulePath']))
        module = __import__(moduleName)                  
        # create an instance of downloader and invoke 'parser'
        objectInstance = getattr(module, self.__configDict[self.__targetDB][nextStep]['className'])(self.__configDict)
        getattr(objectInstance,self.__configDict[self.__targetDB][nextStep]['mainFunction'])()


    def getTargetDB(self):
        return self.__targetDB

    def getConfigDict(self):
        return self.__configDict

    def writeBackToConfig(self):
        #reset the values in config file
        self.__configDict['config'].set('DOWNLOAD', "last download",time.strftime("%d/%m/%Y"))
        self.__configDict['config'].write(self.__configDict['configFilePath'])

        #----------------------------------------------------------------------            
    def DumptoPickle(self, dictData,saveFilePath):

        try:
            with open (saveFilePath , 'wb') as fileHandle: 
                print(saveFilePath)
                pickle.dump(dictData,fileHandle)
        except IOError:
            print(saveFilePath)
            print(dictData)
            print ("open pickle file failed" )    

        #----------------------------------------------------------------------                
    def  UnzipFile(self,fileName,unzipFileName):
        """"""
        try:
            zipHandle = gzip.GzipFile(mode='rb',fileobj=open(fileName,'rb'))
            with open(unzipFileName, 'wb') as fileHandle:
                fileHandle.write(zipHandle.read())

        except IOError:
            raise("Unzip file failed!")      

    def run(self):
        pass



class PDBDownloader(DownloadBase):
    """"""

        #----------------------------------------------------------------------
    def __init__(self,paraDict, targetDB):
        """Constructor"""
        #call the cpnstructor of its parent class
        super(PDBDownloader, self).__init__(paraDict,targetDB)

        #overload the function of its parent class with a same name  
    def get_source_path(self,paraDict):
        #get pdb remote directory
        urlpath = paraDict[self.getTargetDB()]['Download']['URL']
        return urlpath

    def get_target_path(self,paraDict):
        #get local directory from config file
        target = paraDict[self.getTargetDB()]['Download']['targetPath']
        return target   
    
    def get_updb_path(self,paraDict):
        #get uncompress pdb file path from config file
        uncompresspdb =  paraDict[self.getTargetDB()]['Download']['uPDBpath']
        return uncompresspdb
    
    def sync_download(self,sourcepath,targetpath):
        #download PDBparser
        try:
            updatelog = rsync("-rlpt","-v", "-z" ,"--delete", "--port=33444","-i" ,"--log-file=pdbrsync.log" ,sourcepath,targetpath)
            logging.info("Update %s is done." %self.getTargetDB())
        except:
            logging.info("download %s is err." %self.getTargetDB())
            
        return updatelog
    
    def parse_update_path(self,targetpath,changelog):
        result = str(changelog.next())
        updatelist = []
        obsoletelist = []
        
        while True:
            if len(result) == 1:
                pass
            else:  
                if result[0:10] == 'total size':
                    #end line is total size is * speedup is *
                    break            
                elif result[0] == '>':
                    
                    updatelist.append(result[12:].replace('\n',''))
                    #update data format >f+++++++++ /path/filename
                elif result[0] == '*':
                    obsoletelist.append(result[12:].replace('\n','')) 
                    #delete data format *deleting   /path/filename
                else:
                    pass
            result = str(changelog.next())
        return updatelist,obsoletelist
    
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

    def undatepdb_uncompress_path(self,target, uplist ,updbpath):
        final_path_list = []
        if uplist == []:
            return final_path_list
        else: 
            for i in range(0,len(uplist)):
                filename = os.path.join(target, uplist[i])
                final_path =  os.path.join(updbpath, uplist[i][:-3])
                self.mkdir(final_path[:-11]) 
                try:
                    print("ready to UnzipFile")
                    self.UnzipFile(filename,final_path)
                except:
                    pass
                final_path_list.append(final_path)
            return final_path_list
        


    def run(self):
        # do sth. new
        logging.info("%s is downloading." %self.getTargetDB())
        paraDict = self.getConfigDict()
        paraDict[self.getTargetDB()]['Download']['lastDownload'] = time.strftime("%d/%m/%Y",time.localtime(time.time()))
        cfg = ConfigDealer.ConfigDealer()
        cfg.CfgDict2XML(paraDict, 'config.xml')            
        #sourcepath = "rsync.rcsb.org::ftp_data/structures/divided/pdb/01"
        #targetpath = "/home/gongjianting/"
        #updbpath = "/home/gongjianting/123"
        #targetpath = "/home/Biodata/OrignalData/Protein/PDBparser"

               
        sourcepath = self.get_source_path(paraDict)
        targetpath = self.get_target_path(paraDict)
        updbpath = self.get_updb_path(paraDict)
        
        
        changelog = self.sync_download(sourcepath,targetpath)
        updatelist,obsoletelist = self.parse_update_path(targetpath,changelog)      
        fianl_path_list = self.undatepdb_uncompress_path(targetpath, updatelist ,updbpath)
        obsolete_list = self.undatepdb_uncompress_path(targetpath,obsoletelist ,updbpath)
        
        im_up_result_pikcle = paraDict[self.getTargetDB()]['Download']['update_file']
        im_ob_result_pikcle = paraDict[self.getTargetDB()]['Download']['obsolete_file']

        logging.info("%s pickle." %im_ob_result_pikcle)
        logging.info("%s update pikle file." %im_up_result_pikcle)
        
        self.DumptoPickle(fianl_path_list, im_up_result_pikcle)
        self.DumptoPickle(obsolete_list, im_ob_result_pikcle)
        
        paraDict[self.getTargetDB()]['Download']['updatelist'] = 'True'
        paraDict[self.getTargetDB()]['Download']['obsoletelist'] = 'True'
 
        #stroge to config.xml
        #cfg.CfgDict2XML(paraDict, 'config.xml')  
        logging.info("%s download  is done." %self.getTargetDB())
        #self.getConfigDict()[self.getTargetDB()]['Download']['updatelist'] = updatelist
        #self.getConfigDict()[self.getTargetDB()]['Download']['obsoletelist'] = obsoletelist
        
        #return self.DetectNextStep()

           

if __name__=="__main__":
        logging.basicConfig(level=logging.INFO,                     
                            format='%(thread)d %(threadName)s %(asctime)s %(levelname)s: %(message)s',
                            datefmt='%a, %d %b %Y %H:%M:%S',
                            filename='test.log',
                            filemode='w')        
        # try:
            # with open('config.cfg', 'r', encoding='utf-8') as fh:

