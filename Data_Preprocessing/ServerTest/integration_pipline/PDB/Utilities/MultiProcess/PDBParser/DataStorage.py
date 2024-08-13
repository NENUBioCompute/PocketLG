#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Author:Xian Tan
Aim:Storage dictionary type data to Biodata database(for testing)
data:18/1/23
"""

import pymongo
from pymongo import MongoClient

class DataStorage:
    
    def __init__(self,name):
        self.name = name
        self.path = self._login()
        
    def _login(self):
        client = pymongo.MongoClient("59.73.198.168",27017)
        db = client['Biodata']
        db.authenticate("Fangxy","123456")
        collection = client['Biodata'][self.name]
        return collection
        
    def Storage(self,dic):
        return self.path.insert(dic)

"""
You can use this class to test your code
Firstly you can do Storage = DS.DataStorage(Name) Name is your database's name
Then when you get a dic. type data,use Storage.Storage(Name2) Name2 is your data's name
Have fun
"""
