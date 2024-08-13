#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
	File Name：     ToString
	Description :   self-defined object printer
	Author :        Liu Zhe
	date：          2018/12/10
'''
import pprint
import json
import os

class ToString():
    def __init__(self,object):
        self.object = object
        pass
    def getDescription(self):
        return ",".join("{}={}".format(key,getattr(self.object,key)) for key in self.object.__dict__.keys())
    #rewrite __str__
    def __str__(self):
        return "{}->({})".format(self.object.__class__.__name__,self.getDescription())

#if __name__ == "__main__":
    #t  = ToString()
    #print(t)
    #print(globals())