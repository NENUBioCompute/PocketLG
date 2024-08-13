# -*- coding:utf-8 -*-
'''
	File Name：     testParserBase
	Description :   test the Parser of the PDBparser files and save the parsed dictionaries to the service
	Author :        Liu Zhe & Gong Yingli
	date：          2018/11/27
'''
from ParserBase import *
from pickleDealer import *
import os
import json
import datetime
import pprint

if __name__ == "__main__": 


    ParserBase = PDBParserBase()
    pickle = pickleDealer()
    rootdir = "/home/RaidDisk/pdbfiles/updb/pdb"
    saveFilePath = "/home/RaidDisk/pdbfiles/picklefiles/"
    
    now = datetime.datetime.now()
    f = open("/home/RaidDisk/pdbParser/error_records.txt","a+")
    f.write(str(now))
    
    try:
        ParserBase.parseSave(rootdir,saveFilePath) 
    except IndexError:
        print("IndexError:" + filename)
        f.write("IndexError:" + filename)
    except UnboundLocalError:
        print("UnboundLocalError:" + filename)
        f.write("UnboundLocalError:" + filename)
    except KeyError:
        print("KeyError:" + filename)         
        f.write("KeyError:" + filename)
    except AttributeError:
        print("AttributeError:" + filename)  
        f.write("AttributeError:" + filename)    
        
    f.close()        
    #print (end-start)    
#print("Done")
