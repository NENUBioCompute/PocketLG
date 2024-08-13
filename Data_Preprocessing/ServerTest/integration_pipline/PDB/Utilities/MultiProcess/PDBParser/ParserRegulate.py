# -*- coding:utf-8 -*-
'''
	File Name：     parserRegulate
	Description :   parse each tag of PDBparser files such as ATOM and so on
	Author :        Liu Zhe & Gong Yingli
	date：          2018/12/10
'''

from PDBParser.ParserBase import *
from PDBParser.pdbDealer import *
import datetime
from os import path
from optparse import OptionParser
import argparse
import pprint

########################################################################

class ParserRegulate():
    '''
    parse each tag of PDBparser files such as ATOM
    
    '''
    def __init__(self):
        pass
    
    @ParserDecorator
    def Regulate(*args):
        temp = pickleDealer()
        temp_pickle = "temp.pic"
        result = temp.PPrintPickle(temp_pickle)
        # pprint.pprint(result)
        return result
            
if __name__ == "__main__":
    
    '''
    eg.
    Regulate = ParserRegulate()  
    filename =  "pdbfiles/pdb1mof.ent"
    
    Regulate.Regulate(filename)
    Regulate.Regulate(filename,'HEADER')
    Regulate.Regulate(filename,'ATOM')
    Regulate.Regulate(filename,'ATOM',['x'])
    Regulate.Regulate(filename,'ATOM',['x','y','z'])
    '''
    Regulate = ParserRegulate()  
    filename =  "pdbfiles/pdb1mof.ent"
     
    #Regulate.Regulate(filename)
    #Regulate.Regulate(filename,'ATOM')
    #Regulate.Regulate(filename,'ATOM',['x'])
    Regulate.Regulate(filename,'ATOM',['x','y'])