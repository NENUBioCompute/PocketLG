#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
	File Name：     dictDealer
	Description :   calculate the depth of each key and rebuild the dictionaries
	Author :        Liu Zhe & Gong Yingli
	date：          2018/12/11
'''
import pprint
import json
import os

class dictDealer:
    def __init__(self):
        self.all_dict = {}
        self.all_dict_list = []
        self.key_depth = {}
        self.depth = 0
        self.path = []
        self.table = []

    def list_dictionary(self,dictionary):
        for key,value in dictionary.items():
            try:
                self.all_dict = {}
                self.all_dict[key] = value
                self.all_dict_list.append(self.all_dict)
                self.list_dictionary(value)
            except AttributeError:
                pass
    
    def dict_depth(self,dictionary):
        for key,value in dictionary.items():
            try:
                self.key_depth[key] = self.depth
                if(value.keys() is not None):
                    self.depth += 1
                    self.dict_depth(value)
                    self.depth -= 1
            except AttributeError:
                pass
            
    def search_path(self,dictionary,k):
        key_depth = self.key_depth
        #pprint.pprint(key_depth)
        try:
            for key,value in dictionary.items():
                #print("k = " + str(k))
                if(key in key_depth.keys() and key_depth[key] < key_depth[k[0]]):
                    self.path.append(key)
                if(k[0] == key):
                    #print(key)
                    #print(str(dictionary[key]))
                    for i in range(0,len(k)):
                        self.path.append(k[i])
                        self.path.append(dictionary[k[i]])
                        #print(self.path)
                    self.table.append(tuple(self.path))
                    self.path.pop()
                    for i in range(0,len(k)):    
                        self.path.pop()
                        self.path.pop()
                        #print(self.path)                        
                    return

                self.search_path(value,k)
                #self.path = []
                

        except AttributeError:
            pass      
            
if __name__ == "__main__":
    dictionary = {'ATOM': {'model_1': {'ATOM_1': {
                                                  'x': '46.116',
                                                  'y': '47.358',
                                                  'z': '56.720'},
                                       'ATOM_10': {
                                                   'x': '47.544',
                                                   'y': '47.591',
                                                   'z': '54.139'}}}}    
    Dealer = dictDealer()
    Dealer.dict_depth(dictionary)
    '''
    key_depth = 
    {'ATOM': 0, 
    'ATOM_1': 2, 'ATOM_10': 2, 
    'model_1': 1, 
    'x': 3, 'y': 3, 'z': 3}
    '''
    #pprint.pprint(Dealer.key_depth)
    
    #key = ['x']
    #Dealer.search_path(dictionary,key)
    '''
    [('ATOM', 'model_1', 'ATOM_1', 'x', '46.116'), 
    ('ATOM', 'model_1', 'ATOM_10', 'x', '47.544')]
    '''
    
    key1 = 'x'
    key2 = 'y'
    key = [key1,key2]
    flag = 0
    for ki in key:
        if(Dealer.key_depth[ki] != Dealer.key_depth[key[0]]):
            flag = 1
    if(flag == 1):
        print('the depth of the keys are different.')
    else:
        Dealer.search_path(dictionary,key)   
    
    
    #key1 = 'x'
    #key2 = 'y'
    #key3 = 'ATOM_10'
    #key = [key1,key2,key3]
    #Dealer.search_path(dictionary,key)   
    
    pprint.pprint(Dealer.table)
