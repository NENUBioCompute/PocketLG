#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
	File Name：     ParserBase
	Description :   Parser of the PDBparser files
	Author :        Liu Zhe & Gong Yingli
	date：          2018/11/29
'''
import os
from PDBParser.pdbDealer import *
from PDBParser.pickleDealer import *
from PDBParser.dictDealer import *
from PDBParser.ToString import *
import datetime
from os import path
from optparse import OptionParser
import argparse
import pprint


########################################################################   

def ParserDecorator(Regulate):
    '''
    implement the PParserBase decorator
    eg:
    @ParserDecorator
    Regulate(self,'pdb1soc.ent')
    
    @ParserDecorator
    Regulate(self,'pdb1soc.ent','ATOM')
    
    @ParserDecorator
    Regulate('pdb1soc.ent','ATOM','Chain_No')
    
    ParserBase.parse("pdbfiles/pdb1mof.ent")
    ParserBase.parse("pdbfiles/pdb1mof.ent",'ATOM') 
    ParserBase.parse("pdbfiles/pdb1mof.ent",'ATOM','x')  
    '''    
    def wrapper(*args):
        
        PDBfile = args[1]
        # if(not path.isfile(PDBfile) or not PDBfile.endswith("ent")):
        #     raise FileNotFoundError
        #print(PDBfile)
        PDBDealer = pdbDealer()
        fh = open(str(PDBfile)) 
        lines = fh.readlines()   # read file and get all lines
        list_dict = []
        temp_pickle = pickleDealer()
        
        if(len(args) == 2): 
            for line in lines:
                record_name = line[0:6].strip() 
                list = []
                list = PDBDealer.rebuild_list(record_name,line[6:])
                list.insert(0,record_name)
                list_dict.append(list)
                #print(list)
            pdb_mandatory = PDBDealer.list2dict(list_dict)
            #pprint.pprint(pdb_mandatory)
            picklefile = "temp.pic"
            temp_pickle.Dict2Pickle(pdb_mandatory,picklefile)                     
            
        elif(len(args) == 3):
            tag = args[2]
            flag = 0
            for line in lines:
                record_name = line[0:6].strip() 
                if(record_name == tag):
                    flag = 1
                    list = []
                    list = PDBDealer.rebuild_list(record_name,line[6:])
                    list.insert(0,record_name)
                    list_dict.append(list)
                    #print(list)
            if(flag == 1):
                pdb_mandatory = PDBDealer.list2dict(list_dict)  
                #pprint.pprint(pdb_mandatory[tag])
                picklefile = "temp.pic"
                temp_pickle.Dict2Pickle(pdb_mandatory,picklefile)               
            
        else:
            tag = args[2]
            key = args[3]
            print(key)
            flag = 0
            for line in lines:
                record_name = line[0:6].strip() 
                if(record_name == tag):
                    list = []
                    flag = 1
                    list = PDBDealer.rebuild_list(record_name,line[6:])
                    list.insert(0,record_name)
                    list_dict.append(list)                     
            if(flag == 1):
                pdb_mandatory = PDBDealer.list2dict(list_dict)
                #pprint.pprint(pdb_mandatory)
                                   
                new_mandatory = {}
                DICTDealer = dictDealer()
                DICTDealer.dict_depth(pdb_mandatory) 
                flag_path = 0
                for ki in key:
                    if(ki not in DICTDealer.key_depth.keys()):
                        flag_path = 1
                        break
                    if(DICTDealer.key_depth[ki] != DICTDealer.key_depth[key[0]]):
                        flag_path = 2
                if(flag_path == 1):
                    print('the key does not exist.')
                    return                    
                elif(flag_path == 2):
                    print('the depth of the keys are different.')
                    return
                else:                
                    DICTDealer.search_path(pdb_mandatory,key)

                #pprint.pprint(DICTDealer.table)
                        
                picklefile = "temp.pic"
                new_mandatory[str(key)] = DICTDealer.table
                temp_pickle.Dict2Pickle(new_mandatory ,picklefile)      
                      
        fh.close()
        Regulate(*args)
        
    return wrapper

########################################################################
class PDBParserBase():
    '''
    '''
    #----------------------------------------------------------------------
    def __init__(self):
        pass                           
    
    #----------------------------------------------------------------------
    @ParserDecorator
    def parse(*args):
        temp = pickleDealer()
        temp_pickle = "temp.pic"
        temp.PPrintPickle(temp_pickle)
        pass   
    
    def parseSave(self,rootdir,saveFilePath):
        Dealer = pdbDealer()
        pickle = pickleDealer()
        for parent,dirnames,filenames in os.walk(rootdir):
                for PDBfile in filenames:
                    fh = open(os.path.join(parent,PDBfile))
                    lines = fh.readlines()   # read file and get all lines
                    list_dict = []
                    for line in lines:
                        record_name = line[0:6].strip() 
                        list = []
                        if(Dealer.check_parse_way(record_name) == 1):   # space based parse
                            list = line[6:].split()
                        else:   # nonspace based parse
                            list = Dealer.rebuild_list(record_name,line[6:])
                        list.insert(0,record_name)
                        list_dict.append(list)
                        #print(list)
                    pdb_mandatory = Dealer.list2dict(list_dict) 
                    
                    picklefile = str(saveFilePath) + str(PDBfile[:-4]) + ".pic"
                    #print(picklefile)
                    pickle.Dict2Pickle(pdb_mandatory,picklefile)
                    #pickle.PPrintPickle(picklefile)
        
########################################################################   

def main(): 
    '''
    cmd:
    Python ParserBase.py -p pdbfiles/pdb1mof.ent
    Python ParserBase.py --pdb pdbfiles/pdb1mof.ent
    Python ParserBase.py -p pdbfiles/pdb1mof.ent -t HEADER
    Python ParserBase.py -p pdbfiles/pdb1mof.ent -t ATOM -k x
    Python ParserBase.py -p pdbfiles/pdb1mof.ent -t ATOM -k x,y,z
    '''
    #set the parameters
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pdb", required=True, help="input pdb filename needed to be parsered")
    ap.add_argument("-t", "--tag", required=None, help="input the tag you want to know")
    ap.add_argument("-k", "--key", required=None, help="input the key of the dictionary you want to know")
    args = vars(ap.parse_args())
    
    #fetch the argues and print them
    print(args['pdb'])
    print(args['tag'])    
    print(args['key'])

    
    try:
        if(args['pdb'] is not None and args['tag'] is None and args['key'] is None):
            ParserBase = PDBParserBase()
            ParserBase.parse(args['pdb'])
        elif(args['pdb'] is not None and args['tag'] is not None and args['key'] is None):
            ParserBase = PDBParserBase()
            ParserBase.parse(args['pdb'],args['tag'])
        else:
            ParserBase = PDBParserBase()
            keys = args['key'].split(',')
            ParserBase.parse(args['pdb'],args['tag'],keys)       
    except:
        print('key error')
        pass
                

if __name__ == "__main__":
    main()   
    '''
    ParserBase = PDBParserBase()
    
    rootdir = "pdbfiles"
    saveFilePath = "savefiles/"
    filename =  "pdbfiles/pdb1mof.ent"    
    tag = 'ATOM'
    #key = ['x']
    key = ['x','y','z']
    
    ParserBase.parse(filename)
    ParserBase.parse(filename,tag) 
    ParserBase.parse(filename,tag,key) 
    ParserBase.parse(filename,'ATOM',['x'])
    ParserBase.parse(filename,'ATOM',['x','y','z'])
    ParserBase.parseSave(rootdir,saveFilePath) 
    '''
    
    print("Done")
