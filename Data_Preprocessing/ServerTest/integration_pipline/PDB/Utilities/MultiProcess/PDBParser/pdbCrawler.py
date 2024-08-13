# -*- coding:utf-8 -*-
'''
	File Name：     pdbCrawler
	Description :   the web crawler to catch the pdbfile column numbers 
	Author :        Liu Zhe & Gong Yingli
	date：          2018/12/12
'''
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup
import pprint
import json
import re

########################################################################
class pdbCrawler:
    
    def __init__(self):
        pass   
    
    def getHTML(self,url):
        try:
            html = urlopen(url)
        except(HTTPError, URLError) as e:
            return None
        try:
            bsObj = BeautifulSoup(html.read(),"lxml")
        except AttributeError as e:   
            return None
        return bsObj 
    
    def findAll(self,url):
        try:
            html = urlopen(url)
        except(HTTPError, URLError) as e:
            return None
        try:
            bsObj = BeautifulSoup(html.read(),"lxml")
            namelist = bsObj.findAll(["h2","p","pre"])
        except AttributeError as e:   
            return None
        return namelist     
    
    def get_column(self):
        records = {"HEADER":self.get_HEADER_column(),"OBSLTE":self.get_OBSLTE_column(),
                   "TITLE":self.get_TITLE_column(),"SPLIT":self.get_SPLIT_column(),
                   "COMPND":self.get_COMPND_column(),"SOURCE":self.get_SOURCE_column(),
                   "KEYWDS":self.get_KEYWDS_column(),"EXPDTA":self.get_EXPDTA_column(),
                   "NUMMDL":self.get_NUMMDL_column(),"MDLTYP":self.get_MDLTYP_column(),
                   "AUTHOR":self.get_AUTHOR_column(),"REVDAT":self.get_REVDAT_column(),
                   "SPRSDE":self.get_SPRSDE_column(),"CAVEAT":self.get_CAVEAT_column(),
                   "DBREF":self.get_DBREF_column(),"DBREF1":self.get_DBREF1_column(),
                   "DBREF2":self.get_DBREF2_column(),"SEQADV":self.get_SEQADV_column(),
                   "SEQRES":self.get_SEQRES_column(),"MODRES":self.get_MODRES_column(),
                   "HET":self.get_HET_column(),"HETNAM":self.get_HETNAM_column(),
                   "HETSYN":self.get_HETSYN_column(),"FORMUL":self.get_FORMUL_column(),
                   "HELIX":self.get_HELIX_column(),"SHEET":self.get_SHEET_column(),
                   "SSBOND":self.get_SSBOND_column(),"LINK":self.get_LINK_column(),
                   "CISPEP":self.get_CISPEP_column(),"SITE":self.get_SITE_column(),
                   "CRYST1":self.get_CRYST1_column(),"ORIGXn":self.get_ORIGXn_column(),
                   "SCALEn":self.get_SCALEn_column(),"MTRIXn":self.get_MTRIXn_column(),
                   "MODEL":self.get_MODEL_column(),"ATOM":self.get_ATOM_column(),
                   "ANISOU":self.get_ANISOU_column(),"TER":self.get_TER_column(),
                   "HETATM":self.get_HETATM_column(),"CONECT":self.get_CONECT_column(),
                   "MASTER":self.get_MASTER_column()}   
                
        dict = {}
        for key in records.keys():
            dict[key] = records[key]
        return dict
        
    def get_HEADER_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#HEADER")
        for i in range(0, len(content)):
            if(content[i].get_text() == "HEADER"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format"):
                        HEADER_text = content[j + 1].get_text()
                        break                
        
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", HEADER_text)
        
        HEADER = {}
        HEADER["classification"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        HEADER["depDate"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]
        HEADER["idCode"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]

        return HEADER
    
    def get_OBSLTE_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#OBSLTE")
        for i in range(0, len(content)):
            if(content[i].get_text() == "OBSLTE"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        OBSLTE_text = content[j + 1].get_text()
                        break
        
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", OBSLTE_text)
       
        OBSLTE = {}
        OBSLTE["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        OBSLTE["repDate"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        OBSLTE["idCode"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        for i in range(1,10):
            OBSLTE["ridCode_" + str(i)] = [int(m[i+3].split(" - ")[0].strip()),int(m[i+3].split(" - ")[1].strip())]   
        
        return OBSLTE 
    
    def get_TITLE_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#TITLE")
        for i in range(0, len(content)):
            if(content[i].get_text() == "TITLE"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        TITLE_text = content[j + 1].get_text()
                        break            
                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", TITLE_text)
        
        TITLE = {}
        TITLE["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        TITLE["title"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        
        return TITLE
    
    def get_SPLIT_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#SPLIT")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "SPLIT"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SPLIT_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SPLIT_text)
        
        SPLIT = {}
        SPLIT["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())] 
        for i in range(1,4):
            SPLIT["idCode_" + str(i)] = [int(m[i+1].split(" - ")[0].strip()),int(m[i+1].split(" - ")[1].strip())]   
        SPLIT["idCode_4"] = [int(m[5].split(" – ")[0].strip()),int(m[5].split(" – ")[1].strip())]  
        for i in range(5,15):
            SPLIT["idCode_" + str(i)] = [int(m[i+1].split(" - ")[0].strip()),int(m[i+1].split(" - ")[1].strip())]           
        return SPLIT
    
    def get_CAVEAT_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#CAVEAT")
        for i in range(0, len(content)):
            if(content[i].get_text() == "CAVEAT"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format"):
                        CAVEAT_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", CAVEAT_text)
        CAVEAT = {}
        CAVEAT["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        CAVEAT["idCode"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]
        CAVEAT["comment"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]        

        return CAVEAT    
    
    def get_COMPND_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#COMPND")
        for i in range(0, len(content)):
            if(content[i].get_text() == "COMPND"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        COMPND_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", COMPND_text)
        COMPND = {}
        COMPND["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        COMPND["compound"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]

        return COMPND 
        
    def get_SOURCE_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#SOURCE")
        for i in range(0, len(content)):
            if(content[i].get_text() == "SOURCE"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SOURCE_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", SOURCE_text)
        SOURCE = {}
        SOURCE["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        SOURCE["srcName"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]

        return SOURCE
        
    def get_KEYWDS_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#KEYWDS")
        for i in range(0, len(content)):
            if(content[i].get_text() == "KEYWDS"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        KEYWDS_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", KEYWDS_text)
        KEYWDS = {}
        KEYWDS["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        KEYWDS["keywds"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]

        return KEYWDS
    
    def get_EXPDTA_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#EXPDTA")
        for i in range(0, len(content)):
            if(content[i].get_text() == "EXPDTA"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        EXPDTA_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", EXPDTA_text)
        EXPDTA = {}
        EXPDTA["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        EXPDTA["technique"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]

        return EXPDTA
    
    def get_NUMMDL_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#NUMMDL")
        for i in range(0, len(content)):
            if(content[i].get_text() == "NUMMDL"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        NUMMDL_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", NUMMDL_text)
        NUMMDL = {}
        NUMMDL["modelNumber"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        
        return NUMMDL
    
    def get_MDLTYP_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#MDLTYP")
        for i in range(0, len(content)):
            if(content[i].get_text() == "MDLTYP"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        MDLTYP_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", MDLTYP_text)
        MDLTYP = {}
        MDLTYP["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        MDLTYP["comment"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]

        return MDLTYP
    
        
    def get_AUTHOR_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#AUTHOR")
        for i in range(0, len(content)):
            if(content[i].get_text() == "AUTHOR"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        AUTHOR_text = content[j + 1].get_text()
                        break                
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", AUTHOR_text)
        AUTHOR = {}
        AUTHOR["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]
        AUTHOR["authorList"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]

        return AUTHOR
    
    def get_REVDAT_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#REVDAT")
        for i in range(0, len(content)):
            if(content[i].get_text() == "REVDAT"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        REVDAT_text = content[j + 1].get_text()
                        break
    
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)", REVDAT_text)
       
        REVDAT = {}
        REVDAT["modNum"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        REVDAT["continuation"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        REVDAT["modDate"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        REVDAT["modId"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        REVDAT["modType"] = [int(re.findall("[0-9]+[  a-zA-Z]+modType", REVDAT_text)[0].split()[0].strip())]  
        for i in range(1,5):
            REVDAT["record_" + str(i)] = [int(m[i+4].split(" - ")[0].strip()),int(m[i+4].split(" - ")[1].strip())]   
        
        return REVDAT
    
    def get_SPRSDE_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect2.html#SPRSDE")       
        for i in range(0, len(content)):
            if(content[i].get_text() == "SPRSDE"):     
                for j in range(i + 1, len(content)):
                    #print(content[j].get_text())
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or  "Record Format" in content[j].get_text() ):
                        SPRSDE_text = content[j + 1].get_text()
                        break   
        #print(SPRSDE_text)
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SPRSDE_text)
        
        SPRSDE = {}
        SPRSDE["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        SPRSDE["sprsdeDate"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        SPRSDE["idCode"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        for i in range(1,10):
            SPRSDE["sIdCode_" + str(i)] = [int(m[i+3].split(" - ")[0].strip()),int(m[i+3].split(" - ")[1].strip())]   

        return SPRSDE
    
    def get_DBREF_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#DBREF")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "DBREF (standard format)"):
                for j in range(i + 1, len(content)):
                    #print(content[j].get_text())
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or  "Record Format" in content[j].get_text() ):
                        DBREF_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  DBREF_text)
        
        DBREF = {}
        DBREF["idCode"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        DBREF["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", DBREF_text)[0].split()[0].strip())]  
        DBREF["seqBegin"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        DBREF["insertBegin"] = [int(re.findall("[0-9]+[  a-zA-Z]+insertBegin", DBREF_text)[0].split()[0].strip())]  
        DBREF["seqEnd"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]  
        DBREF["insertEnd"] = [int(re.findall("[0-9]+[  a-zA-Z]+insertEnd", DBREF_text)[0].split()[0].strip())]  
        DBREF["database"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())] 
        DBREF["dbAccession"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]   
        DBREF["dbIdCode"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())] 
        DBREF["dbseqBegin"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        DBREF["idbnsBeg"] =  [int(re.findall("[0-9]+[  a-zA-Z]+idbnsBeg", DBREF_text)[0].split()[0].strip())]  
        DBREF["dbseqEnd"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]  
        DBREF["dbinsEnd"] =  [int(re.findall("[0-9]+[  a-zA-Z]+dbinsEnd", DBREF_text)[0].split()[0].strip())]  
         
        return DBREF
    
    def get_DBREF1_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#DBREF")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "DBREF1 / DBREF2"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "DBREF1"):
                        DBREF_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  DBREF_text)
        
        DBREF = {}
        DBREF["idCode"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        DBREF["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", DBREF_text)[0].split()[0].strip())]  
        DBREF["seqBegin"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        DBREF["insertBegin"] = [int(re.findall("[0-9]+[  a-zA-Z]+insertBegin", DBREF_text)[0].split()[0].strip())]  
        DBREF["seqEnd"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]  
        DBREF["insertEnd"] = [int(re.findall("[0-9]+[  a-zA-Z]+insertEnd", DBREF_text)[0].split()[0].strip())]  
        DBREF["database"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())] 
        DBREF["dbIdCode"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())] 
         
        return DBREF
        
    def get_DBREF2_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#DBREF")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "DBREF1 / DBREF2"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "DBREF2"):
                        DBREF_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  DBREF_text)
        
        DBREF = {}
        DBREF["idCode"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        DBREF["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", DBREF_text)[0].split()[0].strip())]  
        DBREF["dbAccession"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        DBREF["seqBegin"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]  
        DBREF["seqEnd"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]  
        
         
        return DBREF
        
    def get_SEQADV_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#SEQADV")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "SEQADV"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SEQADV_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SEQADV_text)
        
        SEQADV = {}
        SEQADV["idCode"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        SEQADV["resName"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        SEQADV["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", SEQADV_text)[0].split()[0].strip())]   
        SEQADV["seqNum"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        SEQADV["iCode"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode", SEQADV_text)[0].split()[0].strip())]  
        SEQADV["database"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        SEQADV["dbAccession"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())] 
        SEQADV["dbRes"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]   
        SEQADV["dbSeq"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())] 
        SEQADV["conflict"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]   
         
        return SEQADV
        
    def get_SEQRES_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#SEQRES")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "SEQRES"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SEQRES_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SEQRES_text)
        
        SEQRES = {}
        SEQRES["serNum"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]  
        SEQRES["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", SEQRES_text)[0].split()[0].strip())]   
        SEQRES["numRes"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]  
        for i in range(1,14):
            SEQRES["resName_" + str(i)] = [int(m[i+2].split(" - ")[0].strip()),int(m[i+2].split(" - ")[1].strip())]   
         
        return SEQRES
        
    def get_MODRES_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect3.html#MODRES")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "MODRES"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        MODRES_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  MODRES_text)
        
        MODRES = {}
        MODRES["idCode"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        MODRES["resName"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        MODRES["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", MODRES_text)[0].split()[0].strip())]   
        MODRES["seqNum"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        MODRES["iCode"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode", MODRES_text)[0].split()[0].strip())]  
        MODRES["stdRes"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        MODRES["comment"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())] 
        
        return MODRES        
        
    def get_HET_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect4.html#HET")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "HET"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        HET_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  HET_text)
        
        HET = {}
        HET["hetID"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        HET["ChainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+ChainID", HET_text)[0].split()[0].strip())]   
        HET["seqNum"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        HET["iCode"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode", HET_text)[0].split()[0].strip())]  
        HET["numHetAtoms"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        HET["text"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())] 
        
        return HET
        
    def get_HETNAM_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect4.html#HETNAM")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "HETNAM"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        HETNAM_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  HETNAM_text)
        
        HETNAM = {}
        HETNAM["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]      
        HETNAM["hetID"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]    
        HETNAM["text"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        
        return HETNAM
        
    def get_HETSYN_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect4.html#HETSYN")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "HETSYN"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        HETSYN_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  HETSYN_text)
        
        HETSYN = {}
        HETSYN["continuation"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]      
        HETSYN["hetID"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]    
        HETSYN["hetSynonyms"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        
        return HETSYN
        
    def get_FORMUL_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect4.html#FORMUL")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "FORMUL"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        FORMUL_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  FORMUL_text)
        
        FORMUL = {}
        FORMUL["compNum"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        FORMUL["hetID"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        FORMUL["asterisk"] = [int(re.findall("[0-9]+[  a-zA-Z]+asterisk", FORMUL_text)[0].split()[0].strip())]   
        FORMUL["continuation"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        FORMUL["text"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        
        return FORMUL
        
    def get_HELIX_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect5.html#HELIX")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "HELIX"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        HELIX_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  HELIX_text)
        
        HELIX = {}
        HELIX["serNum"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        HELIX["helixID"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        HELIX["initResName"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        HELIX["initChainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+initChainID", HELIX_text)[0].split()[0].strip())]   
        HELIX["initSeqNum"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        HELIX["initICode"] = [int(re.findall("[0-9]+[  a-zA-Z]+initICode", HELIX_text)[0].split()[0].strip())]   
        HELIX["endResName"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]    
        HELIX["endChainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+endChainID", HELIX_text)[0].split()[0].strip())]   
        HELIX["endSeqNum"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]   
        HELIX["endICode"] = [int(re.findall("[0-9]+[  a-zA-Z]+endICode", HELIX_text)[0].split()[0].strip())]   
        HELIX["helixClass"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        HELIX["comment"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]   
        HELIX["length"] = [int(m[9].split(" - ")[0].strip()),int(m[9].split(" - ")[1].strip())]   
        
        return HELIX
        
    def get_SHEET_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect5.html#SHEET")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "SHEET"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SHEET_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SHEET_text)
        
        SHEET = {}
        SHEET["strand"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        SHEET["sheetID"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        SHEET["numStrands"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]    
        SHEET["initResName"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]         
        SHEET["initChainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+initChainID", SHEET_text)[0].split()[0].strip())]   
        SHEET["initSeqNum"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]   
        SHEET["initICode"] = [int(re.findall("[0-9]+[  a-zA-Z]+initICode", SHEET_text)[0].split()[0].strip())]   
        SHEET["endResName"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]    
        SHEET["endChainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+endChainID", SHEET_text)[0].split()[0].strip())]   
        SHEET["endSeqNum"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        SHEET["endICode"] = [int(re.findall("[0-9]+[  a-zA-Z]+endICode", SHEET_text)[0].split()[0].strip())]   
        SHEET["sense"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]   
        SHEET["curAtom"] = [int(m[9].split(" - ")[0].strip()),int(m[9].split(" - ")[1].strip())]   
        SHEET["curResName"] = [int(m[10].split(" - ")[0].strip()),int(m[10].split(" - ")[1].strip())]   
        SHEET["curChainId"] = [int(re.findall("[0-9]+[  a-zA-Z]+curChainId", SHEET_text)[0].split()[0].strip())]   
        SHEET["curResSeq"] = [int(m[11].split(" - ")[0].strip()),int(m[11].split(" - ")[1].strip())]   
        SHEET["curICode"] = [int(re.findall("[0-9]+[  a-zA-Z]+curICode", SHEET_text)[0].split()[0].strip())]   
        SHEET["prevAtom"] = [int(m[12].split(" - ")[0].strip()),int(m[12].split(" - ")[1].strip())]   
        SHEET["prevResName"] = [int(m[13].split(" - ")[0].strip()),int(m[13].split(" - ")[1].strip())]   
        SHEET["prevChainId"] = [int(re.findall("[0-9]+[  a-zA-Z]+prevChainId", SHEET_text)[0].split()[0].strip())]   
        SHEET["prevResSeq"] = [int(m[14].split(" - ")[0].strip()),int(m[14].split(" - ")[1].strip())]   
        SHEET["prevICode"] = [int(re.findall("[0-9]+[  a-zA-Z]+prevICode", SHEET_text)[0].split()[0].strip())]
        
        return SHEET
    
    
    def get_SSBOND_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect6.html#SSBOND")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "SSBOND"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SSBOND_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SSBOND_text)
        
        SSBOND = {}
        SSBOND["serNum"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        SSBOND["CYS_1"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        
        SSBOND["chainID1"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID1", SSBOND_text)[0].split()[0].strip())]   
        SSBOND["seqNum1"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        SSBOND["icode1"] = [int(re.findall("[0-9]+[  a-zA-Z]+icode1", SSBOND_text)[0].split()[0].strip())]   
        SSBOND["CYS_2"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]    
        SSBOND["chainID2"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID2", SSBOND_text)[0].split()[0].strip())]   
        SSBOND["seqNum2"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]   
        SSBOND["icode2"] = [int(re.findall("[0-9]+[  a-zA-Z]+icode2", SSBOND_text)[0].split()[0].strip())]   
        SSBOND["sym1"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]   
        SSBOND["sym2"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        SSBOND["Length"] = [int(m[8].split(" – ")[0].strip()),int(m[8].split(" – ")[1].strip())]   

        return SSBOND
    
    def get_LINK_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect6.html#LINK")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "LINK"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        LINK_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  LINK_text)
        
        LINK = {}
        LINK["name1"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        LINK["altLoc1"] = [int(re.findall("[0-9]+[  a-zA-Z]+altLoc1", LINK_text)[0].split()[0].strip())]   
        LINK["resName1"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        
        LINK["chainID1"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID1", LINK_text)[0].split()[0].strip())]   
        LINK["resSeq1"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        LINK["iCode1"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode1", LINK_text)[0].split()[0].strip())]   
        LINK["name2"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]  
        LINK["altLoc2"] = [int(re.findall("[0-9]+[  a-zA-Z]+altLoc2", LINK_text)[0].split()[0].strip())]   
        LINK["resName2"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]           
        LINK["chainID2"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID2", LINK_text)[0].split()[0].strip())]   
        LINK["resSeq2"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]   
        
        LINK["iCode2"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode2", LINK_text)[0].split()[0].strip())]   
        LINK["sym1"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        LINK["sym2"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]   
        LINK["Length"] = [int(m[9].split(" – ")[0].strip()),int(m[9].split(" – ")[1].strip())]   

        return LINK
    
    def get_CISPEP_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect6.html#CISPEP")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "CISPEP"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        CISPEP_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  CISPEP_text)
        
        CISPEP = {}
        CISPEP["serNum"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        CISPEP["pep1"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        
        CISPEP["chainID1"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID1", CISPEP_text)[0].split()[0].strip())]   
        CISPEP["seqNum1"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
        CISPEP["icode1"] = [int(re.findall("[0-9]+[  a-zA-Z]+icode1", CISPEP_text)[0].split()[0].strip())]   
        CISPEP["pep2"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]    
        CISPEP["chainID2"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID2", CISPEP_text)[0].split()[0].strip())]   
        CISPEP["seqNum2"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]   
        CISPEP["icode2"] = [int(re.findall("[0-9]+[  a-zA-Z]+icode2", CISPEP_text)[0].split()[0].strip())]   
        CISPEP["modNum"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]   
        CISPEP["measure"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        return CISPEP
    
    def get_SITE_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect7.html#SITE")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "SITE"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SITE_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SITE_text)
        
        SITE = {}
        SITE["seqNum"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]  
        SITE["siteID"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        SITE["numRes"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]   
          
        SITE["resName1"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())] 
        SITE["chainID1"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID1", SITE_text)[0].split()[0].strip())]   
        SITE["seq1"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]   
        SITE["iCode1"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode1", SITE_text)[0].split()[0].strip())]   
        
        SITE["resName2"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]           
        SITE["chainID2"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID2", SITE_text)[0].split()[0].strip())]   
        SITE["seq2"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        SITE["iCode2"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode2", SITE_text)[0].split()[0].strip())]  
        
        SITE["resName3"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())] 
        SITE["chainID3"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID3", SITE_text)[0].split()[0].strip())]   
        SITE["seq3"] = [int(m[9].split(" - ")[0].strip()),int(m[9].split(" - ")[1].strip())]   
        SITE["iCode3"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode3", SITE_text)[0].split()[0].strip())]   
        
        SITE["resName4"] = [int(m[10].split(" - ")[0].strip()),int(m[10].split(" - ")[1].strip())]           
        SITE["chainID4"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID4", SITE_text)[0].split()[0].strip())]   
        SITE["seq4"] = [int(m[11].split(" - ")[0].strip()),int(m[11].split(" - ")[1].strip())]   
        SITE["iCode4"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode4", SITE_text)[0].split()[0].strip())]  
        
        return SITE
    
    def get_CRYST1_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#CRYST1")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "CRYST1"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        CRYST1_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  CRYST1_text)
        
        CRYST1 = {}
        CRYST1["a"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        CRYST1["b"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        CRYST1["c"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]    
        CRYST1["alpha"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())] 
        CRYST1["beta"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]    
        CRYST1["gamma"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())] 
        CRYST1["sGroup"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]    
        CRYST1["z"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]         
           
        return CRYST1
    
    def get_ORIGXn_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#ORIGXn")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "ORIGXn"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        ORIGXn_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  ORIGXn_text)
        
        ORIGXn = {}
        ORIGXn["o[n][1]"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        ORIGXn["o[n][2]"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        ORIGXn["o[n][3]"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]    
        ORIGXn["t[n]"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]         
           
        return ORIGXn
        
    def get_SCALEn_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#SCALEn")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "SCALEn"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        SCALEn_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  SCALEn_text)
        
        SCALEn = {}
        SCALEn["s[n][1]"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        SCALEn["s[n][2]"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        SCALEn["s[n][3]"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]    
        SCALEn["u[n]"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]         
           
        return SCALEn
    
    def get_MTRIXn_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect8.html#MTRIXn")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "MTRIXn"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        MTRIXn_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  MTRIXn_text)
        
        MTRIXn = {}
        MTRIXn["serial"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]    
        MTRIXn["m[n][1]"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        MTRIXn["m[n][2]"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]    
        MTRIXn["m[n][3]"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        MTRIXn["v[n]"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())] 
        MTRIXn["iGiven"] = [int(re.findall("[0-9]+[  a-zA-Z]+iGiven", MTRIXn_text)[0].split()[0].strip())]  
           
        return MTRIXn
    
    def get_MODEL_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#MODEL")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "MODEL"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        MODEL_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  MODEL_text)
        
        MODEL = {}
        MODEL["serial"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())] 
        
        return MODEL
        
    def get_ATOM_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "ATOM"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        ATOM_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  ATOM_text)
        
        ATOM = {}
        ATOM["serial"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]  
        ATOM["name"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        ATOM["altLoc"] = [int(re.findall("[0-9]+[  a-zA-Z]+altLoc", ATOM_text)[0].split()[0].strip())]   
          
        ATOM["resName"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())] 
        ATOM["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", ATOM_text)[0].split()[0].strip())]   
        ATOM["resSeq"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        ATOM["iCode"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode", ATOM_text)[0].split()[0].strip())] 
        
        ATOM["x"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]  
        ATOM["y"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]  
        ATOM["z"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]  
        
        ATOM["occupancy"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]  
        ATOM["tempFactor"] = [int(m[9].split(" - ")[0].strip()),int(m[9].split(" - ")[1].strip())]  
        ATOM["element"] = [int(m[10].split(" - ")[0].strip()),int(m[10].split(" - ")[1].strip())]  
        ATOM["charge"] = [int(m[11].split(" - ")[0].strip()),int(m[11].split(" - ")[1].strip())]  
        
        return ATOM
        
    def get_ANISOU_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ANISOU")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "ANISOU"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        ANISOU_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  ANISOU_text)
        
        ANISOU = {}
        ANISOU["serial"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]  
        ANISOU["name"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        ANISOU["altLoc"] = [int(re.findall("[0-9]+[  a-zA-Z]+altLoc", ANISOU_text)[0].split()[0].strip())]   
          
        ANISOU["resName"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())] 
        ANISOU["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", ANISOU_text)[0].split()[0].strip())]   
        ANISOU["resSeq"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        ANISOU["iCode"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode", ANISOU_text)[0].split()[0].strip())] 
        
        ANISOU["u[0][0]"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]  
        ANISOU["u[1][1]"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]   
        ANISOU["u[2][2]"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]   
        ANISOU["u[0][1]"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]  
        ANISOU["u[0][2]"] = [int(m[9].split(" - ")[0].strip()),int(m[9].split(" - ")[1].strip())]   
        ANISOU["u[1][2]"] = [int(m[10].split(" - ")[0].strip()),int(m[10].split(" - ")[1].strip())]     
        
        ANISOU["element"] = [int(m[11].split(" - ")[0].strip()),int(m[11].split(" - ")[1].strip())]   
        ANISOU["charge"] = [int(m[12].split(" - ")[0].strip()),int(m[12].split(" - ")[1].strip())]   
        
        return ANISOU
    
    def get_TER_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#TER")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "TER"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        TER_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  TER_text)
        
        TER = {}
        TER["serial"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]  
        TER["resName"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        TER["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", TER_text)[0].split()[0].strip())]
        TER["resSeq"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())] 
        TER["iCode"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode", TER_text)[0].split()[0].strip())]   
          
        return TER
        
    def get_HETATM_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#HETATM")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "HETATM"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        ATOM_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  ATOM_text)
        
        ATOM = {}
        ATOM["serial"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]  
        ATOM["name"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())]   
        ATOM["altLoc"] = [int(re.findall("[0-9]+[  a-zA-Z]+altLoc", ATOM_text)[0].split()[0].strip())]   
          
        ATOM["resName"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())] 
        ATOM["chainID"] = [int(re.findall("[0-9]+[  a-zA-Z]+chainID", ATOM_text)[0].split()[0].strip())]   
        ATOM["resSeq"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]   
        ATOM["iCode"] = [int(re.findall("[0-9]+[  a-zA-Z]+iCode", ATOM_text)[0].split()[0].strip())] 
        
        ATOM["x"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())]  
        ATOM["y"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]  
        ATOM["z"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())]  
        
        ATOM["occupancy"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())]  
        ATOM["tempFactor"] = [int(m[9].split(" - ")[0].strip()),int(m[9].split(" - ")[1].strip())]  
        ATOM["element"] = [int(m[10].split(" - ")[0].strip()),int(m[10].split(" - ")[1].strip())]  
        ATOM["charge"] = [int(m[11].split(" - ")[0].strip()),int(m[11].split(" - ")[1].strip())]  
        
        return ATOM
    
    def get_CONECT_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect10.html#CONECT")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "CONECT"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        CONECT_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  CONECT_text)
        
        CONECT = {}
        for i in range(1,6):
            CONECT["serial_" + str(i)] = [int(m[i].split(" - ")[0].strip()),int(m[i].split(" - ")[1].strip())]  
        
        return CONECT
    
    def get_MASTER_column(self):
        content = self.findAll("http://www.wwpdb.org/documentation/file-format-content/format33/sect11.html#MASTER")
        
        for i in range(0, len(content)):
            if(content[i].get_text() == "MASTER"):
                for j in range(i + 1, len(content)):
                    if(content[j].get_text() == "Record Format" or content[j].get_text() == "Record  Format" or content[j].get_text() == "Record Format "):
                        MASTER_text = content[j + 1].get_text()
                        break               
        m = re.findall("([0-9]+[  ]+-[  ]+[0-9]+|[0-9]+[  ]+–[  ]+[0-9]+)",  MASTER_text)
        
        MASTER = {}
        MASTER["numRemark"] = [int(m[1].split(" - ")[0].strip()),int(m[1].split(" - ")[1].strip())]  
        MASTER["0"] = [int(m[2].split(" - ")[0].strip()),int(m[2].split(" - ")[1].strip())] 
        MASTER["numHet"] = [int(m[3].split(" - ")[0].strip()),int(m[3].split(" - ")[1].strip())]  
       
        MASTER["numHelix"] = [int(m[4].split(" - ")[0].strip()),int(m[4].split(" - ")[1].strip())]  
        MASTER["numSheet"] = [int(m[5].split(" - ")[0].strip()),int(m[5].split(" - ")[1].strip())] 
        MASTER["numTurn"] = [int(m[6].split(" - ")[0].strip()),int(m[6].split(" - ")[1].strip())]  
        MASTER["numSite"] = [int(m[7].split(" - ")[0].strip()),int(m[7].split(" - ")[1].strip())] 
        
        MASTER["numXform"] = [int(m[8].split(" - ")[0].strip()),int(m[8].split(" - ")[1].strip())] 
        MASTER["numCoord"] = [int(m[9].split(" - ")[0].strip()),int(m[9].split(" - ")[1].strip())]  
        MASTER["numTer"] = [int(m[10].split(" - ")[0].strip()),int(m[10].split(" - ")[1].strip())]  
        MASTER["numConect"] = [int(m[11].split(" - ")[0].strip()),int(m[11].split(" - ")[1].strip())]  
        MASTER["numSeq"] = [int(m[12].split(" - ")[0].strip()),int(m[12].split(" - ")[1].strip())]        
        
        return MASTER
    
    
if __name__ == "__main__":

    Crawler = pdbCrawler()    
    pprint.pprint(Crawler.get_column())
    

    