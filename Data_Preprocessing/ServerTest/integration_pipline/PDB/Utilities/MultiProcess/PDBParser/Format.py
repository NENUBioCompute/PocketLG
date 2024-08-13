# -*- coding:utf-8 -*-
'''
	File Name：     Format
	Description :   split field by column numbers
	Author :        Gong Yingli & Liu Zhe
	date：          2018/12/10
'''
from jsonDealer import *
from pdbCrawler import *

########################################################################
class Format:
    
    def __init__(self):
        pass   
    
    def PDBFormat(self,filepath):
        Crawler = pdbCrawler()
        Dealer = jsonDealer()
        json_new = Crawler.get_column()
        json_data = Dealer.get_json_data(filepath)
        Dealer.change_json_data(json_data,json_new)
        if Dealer.flag == 1:
            Dealer.rewrite_json_file(filepath,json_data)
            
        return
    def otherFormat(self):
        pass
        


if __name__ == "__main__":

    filepath = 'location.json'
    formatPDB = Format()
    formatPDB.PDBFormat(filepath)
    