#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
	File Name：     pickleDealer
	Description :   dealer of pickle files
	Author :        Liu Zhe & Gong Yingli
	date：          2018/11/29
'''
import pickle
import pprint
import json
import os
class pickleDealer:
    def __init__(self):
        pass
    
    def Dict2Pickle(self, dictData,picklefile):
        with open (picklefile , 'wb') as f: 
            pickle.dump(dictData,f,pickle.HIGHEST_PROTOCOL)     
    
    def PrintPickle(self, picklefile):
        filename = open(picklefile,'rb')    
        picklelist = pickle.load(filename)
        print(picklelist)  
        
    def PPrintPickle(self, picklefile):
        filename = open(picklefile,'rb')    
        picklelist = pickle.load(filename)
        # pprint.pprint(picklelist)
        return picklelist
        
if __name__ == "__main__": 
    
    pic = pickleDealer()
    rootdir = "picklefiles"
    for parent,dirnames,filenames in os.walk(rootdir):
        for filename in filenames:    
            picklefile = str(rootdir) +"/" + str(filename)
            pic.PPrintPickle(picklefile)