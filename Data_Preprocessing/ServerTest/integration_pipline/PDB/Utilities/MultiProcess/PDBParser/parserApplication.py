# -*- coding:utf-8 -*-
'''
	File Name：     parserApplication
	Description :   implement the query interface of PDBparser files
	Author :        Liu Zhe & Gong Yingli
	date：          2018/12/10
'''
from ParserRegulate import *
import argparse
import pprint


########################################################################
class parserApplication:
    
    def __init__(self):
        pass   
    
    def get_ATOM_3dCoordinate(self,filename):
        '''
        the result will be written into "temp.pic"
        '''
        Regulate = ParserRegulate() 
        Regulate.Regulate(filename,'ATOM',['x','y','z'])

if __name__ == "__main__":

    app = parserApplication()
    filename =  "pdbfiles/pdb1mof.ent"
    app.get_ATOM_3dCoordinate(filename)