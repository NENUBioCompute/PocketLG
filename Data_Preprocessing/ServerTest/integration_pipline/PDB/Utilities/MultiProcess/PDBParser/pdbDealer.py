# -*- coding:utf-8 -*-
'''
	File Name：     pdbDealer
	Description :   rebuild the PDBparser files parsed by ParserBase & convert the files to dictionaries
	Author :        Liu Zhe & Gong Yingli
	date：          2018/11/30
'''
import json


class pdbDealer:
    
    def __init__(self):
        with open("PDBParser/location.json") as f:
        # with open("location.json") as f:
            self.load_dict = json.load(f) 
            
# =============================================================================
            
    def rebuild_list(self,record_name,line):        
        records = {'OBSLTE':self.rebuild_OBSLTE,'TITLE':self.rebuild_TITLE,
                   'HEADER':self.rebuild_HEADER,'COMPND':self.rebuild_COMPND,
                   'SOURCE':self.rebuild_SOURCE,'KEYWDS':self.rebuild_KEYWDS,
                   'EXPDTA':self.rebuild_EXPDTA,'MDLTYP':self.rebuild_MDLTYP,
                   'AUTHOR':self.rebuild_AUTHOR,'JRNL':self.rebuild_JRNL,
                   'HETNAM':self.rebuild_HETNAM,'HETSYN':self.rebuild_HETSYN,
                   'FORMUL':self.rebuild_FORMUL,'CAVEAT':self.rebuild_CAVEAT,
                   'MTRIX1':self.rebuild_MTRIX1,'MTRIX2':self.rebuild_MTRIX2,
                   'MTRIX3':self.rebuild_MTRIX3,'ORIGX1':self.rebuild_ORIGX1,
                   'ORIGX2':self.rebuild_ORIGX2,'ORIGX3':self.rebuild_ORIGX3,
                   'SCALE1':self.rebuild_SCALE1,'SCALE2':self.rebuild_SCALE2,
                   'SCALE3':self.rebuild_SCALE3,'ATOM':self.rebuild_ATOM,
                   'ANISOU':self.rebuild_ANISOU,'SIGUIJ':self.rebuild_SIGUIJ,
                   'HETATM':self.rebuild_HETATM,'SHEET':self.rebuild_SHEET,
                   'SSBOND':self.rebuild_SSBOND,'LINK':self.rebuild_LINK,
                   'CISPEP':self.rebuild_CISPEP,'SITE':self.rebuild_SITE,
                   'REMARK':self.rebuild_REMARK,'HET':self.rebuild_HET,
                   'CONECT':self.rebuild_CONECT,'MASTER':self.rebuild_MASTER,
                   'MODRES':self.rebuild_MODRES,'SEQADV':self.rebuild_SEQADV,
                   'DBREF':self.rebuild_DBREF,'DBREF1':self.rebuild_DBREF1,
                   'DBREF2':self.rebuild_DBREF2,'HELIX':self.rebuild_HELIX,
                   'REVDAT':self.rebuild_REVDAT,'CRYST1':self.rebuild_CRYST1,
                   'SPLIT':self.rebuild_SPLIT,'NUMMDL':self.rebuild_NUMMDL,
                   'SPRSDE':self.rebuild_SPRSDE,'SEQRES':self.rebuild_SEQRES}
        list = []
        if(record_name in records):
            list = records.get(record_name)(line)
        return list
    
# ============================================================================= 
        
    def list2dict(self,list):       
        records = {'OBSLTE':self.list2dict_OBSLTE,'TITLE':self.list2dict_TITLE,
                   'SPLIT':self.list2dict_SPLIT,'NUMMDL':self.list2dict_NUMMDL,
                   'REVDAT':self.list2dict_REVDAT,'SPRSDE':self.list2dict_SPRSDE,
                   'DBREF':self.list2dict_DBREF,'DBREF1':self.list2dict_DBREF1,
                   'DBREF2':self.list2dict_DBREF2,'SEQADV':self.list2dict_SEQADV,
                   'SEQRES':self.list2dict_SEQRES,'MODRES':self.list2dict_MODRES,
                   'HET':self.list2dict_HET,'HELIX':self.list2dict_HELIX,
                   'SHEET':self.list2dict_SHEET,'SSBOND':self.list2dict_SSBOND,
                   'LINK':self.list2dict_LINK,'CISPEP':self.list2dict_CISPEP,
                   'SITE':self.list2dict_SITE,'CRYST1':self.list2dict_CRYST1,
                   'MTRIX1':self.list2dict_MTRIX1,'MTRIX2':self.list2dict_MTRIX2,
                   'MTRIX3':self.list2dict_MTRIX3,'ORIGX1':self.list2dict_ORIGX1,
                   'ORIGX2':self.list2dict_ORIGX2,'ORIGX3':self.list2dict_ORIGX3,
                   'SCALE1':self.list2dict_SCALE1,'SCALE2':self.list2dict_SCALE2,
                   'SCALE3':self.list2dict_SCALE3,'HEADER':self.list2dict_HEADER,
                   'COMPND':self.list2dict_COMPND,'KEYWDS':self.list2dict_KEYWDS,
                   'SOURCE':self.list2dict_SOURCE,'ATOM':self.list2dict_ATOM,
                   'EXPDTA':self.list2dict_EXPDTA,'MDLTYP':self.list2dict_MDLTYP,
                   'HETATM':self.list2dict_HETATM,'CONECT':self.list2dict_CONECT,
                   'MASTER':self.list2dict_MASTER,'AUTHOR':self.list2dict_AUTHOR,
                   'JRNL':self.list2dict_JRNL,'FORMUL':self.list2dict_FORMUL,
                   'HETNAM':self.list2dict_HETNAM,'HETSYN':self.list2dict_HETSYN,
                   'CAVEAT':self.list2dict_CAVEAT,'REMARK':self.list2dict_REMARK}
        
        marks = {'OBSLTE':0,'TITLE':0,'SPLIT':0,'NUMMDL':0,'REVDAT':0,'SPRSDE':0,'DBREF':0,'DBREF1':0,
                   'DBREF2':0,'SEQADV':0,'SEQRES':0,'MODRES':0,'HET':0,'HELIX':0,'SHEET':0,'SSBOND':0,
                   'LINK':0,'CISPEP':0,'SITE':0,'CRYST1':0,'MTRIX1':0,'MTRIX2':0,'MTRIX3':0,'ORIGX1':0,
                   'ORIGX2':0,'ORIGX3':0,'SCALE1':0,'SCALE2':0,'SCALE3':0,'ATOM':0,
                   'HETATM':0,'CONECT':0,'MASTER':0,'END':0,'HEADER':0,'COMPND':0,
                   'SOURCE':0,'KEYWDS':0,'EXPDTA':0,'MDLTYP':0,'AUTHOR':0,'JRNL':0,'HETNAM':0,'HETSYN':0,
                   'FORMUL':0,'CAVEAT':0,'REMARK':0}

        dict = {}
        for row in list:
            part_dict = {}
            if(row[0] in records.keys() and marks[row[0]] == 0):
                marks[row[0]] = 1
                part_dict = records.get(row[0])(list)
                dict[row[0]] = part_dict  
        return dict   
    
# =============================================================================
        
    def rebuild_HEADER(self,line):
        list = []
        t1 = line[4:44]
        t1.replace('/',',')
        if t1.find(',') != -1:
            for i in t1.split(','):
                list.append(i.strip())
        else:
            list.append(t1.strip())
        list.append(line[44:53].strip())
        list.append(line[56:60].strip())
        return list
    
    def rebuild_TITLE(self,line):          
        list = []
        key = 'TITLE'
        keyList = ['continuation','title']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_SPRSDE(self,line):          
        list = []
        key = 'SPRSDE'
        keyList = ['continuation','sprsdeDate','idCode','sIdCode_1','sIdCode_2',
                   'sIdCode_3','sIdCode_4','sIdCode_5','sIdCode_6','sIdCode_7',
                   'sIdCode_8','sIdCode_9']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_SPLIT(self,line):          
        list = []
        key = 'SPLIT'
        keyList = ['continuation','idCode_1','idCode_2','idCode_3','idCode_4',
                   'idCode_5','idCode_6','idCode_7','idCode_8','idCode_9',
                   'idCode_10','idCode_11','idCode_12','idCode_13','idCode_14']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_NUMMDL(self,line):          
        list = []
        key = 'NUMMDL'
        keyList = ['modelNumber']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_TER(self,line):          
        list = []
        key = 'TER'
        keyList = ['serial','resName','chainID','resSeq','iCode']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_MODEL(self,line):          
        list = []
        key = 'MODEL'
        keyList = ['serial']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_SEQRES(self,line):          
        list = []
        key = 'SEQRES'
        keyList = ['serNum','chainID','numRes','resName_1','resName_2','resName_3',
                   'resName_4','resName_5','resName_6','resName_7','resName_8',
                   'resName_9','resName_10','resName_11','resName_12','resName_13']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_COMPND(self,line):
        list = []
        list.append(line[1:4].strip())
        line = line[4:]
        if line[-1] == ',':
            flag = 1
        else:
            flag = 0
        if line.strip()[-1] == ';':
            line = line.strip()[:-1]
        if line.find(':') != -1:
            temp = line.split(':')
            list.append(temp[0].strip())
            for i in temp[1].split(','):
                list.append(i.strip())
        else:
            list.append('continuation')
            for i in line.split(','):
                list.append(i.strip())
        if flag:
            list.append(',')
        return list

    def rebuild_SOURCE(self,line):
        list = []
        list.append(line[1:4].strip())
        line = line[4:]
        if line[-1] == ',':
            flag = 1
        else:
            flag = 0
        if line.strip()[-1] == ';':
            line = line.strip()[:-1]
        if line.find(':') != -1:
            temp = line.split(':')
            list.append(temp[0].strip())
            for i in temp[1].split(','):
                list.append(i.strip())
        else:
            list.append('continuation')
            for i in line.split(','):
                list.append(i.strip())
        if flag:
            list.append(',')
        return list

    def rebuild_KEYWDS(self,line):
        list = []
        line = line.rstrip()
        line.replace('/',',')
        if line[-1] == ',':
            flag = 1
        else :
            flag = 0
        list.append(line[2:4].strip())
        for i in line[4:].strip().split(','):
            if i:
                list.append(i.strip())
        if flag:
            list.append(',')
        else:
            list.append(' ')
        return list
    
    def rebuild_EXPDTA(self,line):
        list = []
        line = line.rstrip()
        line.replace('/',';')
        if line[-1] == ';':
            flag = 1
        else :
            flag = 0
        list.append(line[2:4].strip())
        for i in line[4:].strip().split(';'):
            if i:
                list.append(i.strip())
        if flag:
            list.append(',')
        else:
            list.append(' ')
        return list

    def rebuild_MDLTYP(self,line):
        list = []
        line = line.rstrip()
        letter = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P',
              'Q','R','S','T','U','V','W','X','Y','Z']   
        if line[-1] == ',' or line[-1] == ';':
            flag = 1
        else :
            flag = 0 
        if line[3] == ' ':       
            list.append('1')
        else:
            list.append(line[3])
        temp = line[4:].split(',') 
        #loc_CHAIN = temp.find('CHAIN')
        #loc_sem = temp.find(';',loc_CHAIN)
        loc_CHAIN = 80
        loc_sem = 0
        for index,i in enumerate(temp):
            i = i.strip() 
            if i.find('CHAIN') != -1:
                loc_CHAIN = index
                loc_sem = 80
                list.append(i)
            elif i.find(';') != -1:
                loc_sem = index;           
                j = i.split(';')
                if len(j[0].strip()) == 1 and index > loc_CHAIN:
                    list.append('CHAIN ' + j[0].strip())
                else:
                    list.append(j[0].strip())
                loc_CHAIN = 80;
                list.append(j[1].strip())
            else:
                if i in letter and index > loc_CHAIN and index < loc_sem:
                    list.append('CHAIN ' + i)
                elif i:
                    list.append(i)  
        if flag:
            list.append(',')
        else:
            list.append(' ')
        return list 

    def rebuild_AUTHOR(self,line):
        '''
        ['1', 'M.B.BERRY', 'P.LIANG', 'M.GLASER,']                   
        ['2', ' G.N.PHILLIPS JR.', 'T.L.ST. STEVENS,']          
        ['3', 'M.B.BERRY', 'B.MEADOR', 'G.N.PHILLIPS']
        ['4', 'JR.', 'T.L.ST. STEVENS']
        '''
        list = []
        
        if(line.strip()[-1] == ','):
            list = line.strip().strip(',').split(',')
            list[-1] = list[-1] + ',' 
        else:
            list = line[4:].strip().split(',')
        if(line[5] == ' '):
            list[0] = ' ' + list[0]        
        if(line[2:4] == '  '):
            list.insert(0,'1') 
        else:
            list.insert(0,line[2:4].strip())         

        return list

    def rebuild_JRNL(self,line):
        '''
        [' ','AUTH','1','P.A.SMITH','M.F.T.KOEHLER','H.S.GIRGIS','D.YAN','Y.CHEN,']      
        [' ','AUTH','2','J.J.CRAWFORD','M.R.DURK','R.I.HIGUCHI','J.KANG','J.MURRAY,']          
        [' ','AUTH','3','P.PARASELLI','S.PARK','W.PHUNG','J.G.QUINN','T.C.ROBERTS','L.ROUGE']   
        [' ','AUTH','4','J.B.SCHWARZ','E.SKIPPINGTON','J.WAI','M.XU','Z.YU','H.ZHANG','M.W.TAN']
        [' ','AUTH','5',' C.E.HEISE']                                                    
        [' ','TITL','1' ,'OPTIMIZED ARYLOMYCINS ARE A NEW CLASS OF GRAM-NEGATIVE']     
        [' ','TITL','2','ANTIBIOTICS.']                                                 
        ['REF', '0', '', '', '', '']                                              
        ['REF', '1', 'NATURE', '561', '189', '2018']
        ['REF', '2', 'ACTA CRYSTALLOGR.,SECT.D', '297', '1692', '2002']
        ['REFN', 'ESSN', '1476-46']
        ['PMID', '30209367']
        ['DOI', '10.1038/S41586-018-0483-6']
        ['PUBL', '1', 'jsowmd dsio,']
        ['PUBL', '2', 'dnwdhwldks']
        ['PUBL', '3', 'fwd']    
        '''
        
        list = []
        
        if(line[6:10] == 'AUTH'):
            if(line.strip()[-1] == ','):
                list = line[12:].strip().strip(',').split(',')
                list[-1] = list[-1] + ','      
            else:
                list = line[12:].strip().strip(',').split(',')
            if(line[13] == ' '):
                list[0] = ' ' + list[0]      
            if(line[10:12] == '  '):
                list.insert(0,'1')
            else:
                list.insert(0,line[10:12].strip())
            list.insert(0,'AUTH')
            list.insert(0,line[3]) 
            
        elif(line[6:10] == 'EDIT'):
            if(line.strip()[-1] == ','):
                list = line[12:].strip().strip(',').split(',')
                list[-1] = list[-1] + ','      
            else:
                list = line[12:].strip().strip(',').split(',')
            if(line[13] == ' '):
                list[0] = ' ' + list[0]      
            if(line[10:12] == '  '):
                list.insert(0,'1')
            else:
                list.insert(0,line[10:12].strip())
            list.insert(0,'EDIT')
            list.insert(0,line[3])
                
        elif(line[6:10] == 'TITL'):
            if(line[13] == ' '):
                list.append(' ' + line[12:].strip())
            else:
                list.append(line[12:].strip())
            if(line[10:12] == '  '):
                list.insert(0,'1')
            else:
                list[0] = list[0]
                list.insert(0,line[10:12].strip())
            list.insert(0,'TITL')
            list.insert(0,line[3])
                
        elif(line[6:10] == 'REF '):
            if(line[13:28] == 'TO BE PUBLISHED'):
                list = ['','','','']
                if(line[11] == ' '):
                    list.insert(0,'0')
                else:
                    list.insert(0,line[11])                             
            else:
                list.append(line[45:49].strip())
                list.append(line[50:55].strip())
                list.append(line[56:60])                
                list.insert(0,line[13:41].strip())                
                if(line[10:12] == '  '):
                    list.insert(0,'1')
                else:
                    list.insert(0,line[10:12].strip())
            list.insert(0,'REF')   
                    
        elif(line[6:10] == 'PUBL'):
            if(line[13] == ' '):
                list.append(' ' + line[12:].strip())
            else:
                list.append(line[12:].strip())
            if(line[10:12] == '  '):
                list.insert(0,'1')               
            else:
                list.insert(0,line[10:12].strip())
            list.insert(0,'PUBL') 
            
        elif(line[6:10] == 'REFN'):
            list.append(line[6:10].strip())
            list.append(line[29:33].strip())
            list.append(line[34:59].strip())
            list.insert(0,'REFN')
        
        elif(line[6:10] == 'PMID'):
            if(line[13] == ' '):
                list = [' ']
            else:
                list.append(line[13:].strip())
            list.insert(0,'PMID')
            
        elif(line[6:10] == 'DOI '):
            if(line[13] == ' '):
                list = [' ']
            else:    
                list.append(line[13:].strip())
            list.insert(0,'DOI')        
               
        return list
    
    def rebuild_HETNAM(self,line):
        '''
        ['1', 'NAG', '"N-ACETYL-D-GLUCOSAMINE"']
        ['1', 'SAD', '"BETA-METHYLENE SELENAZOLE-4-CARBOXAMIDE ADENINE"']
        ['2', 'SAD', '"DINUCLEOTIDE"']
        ['1', 'UDP', '"URIDINE-5\'-DIPHOSPHATE"']
        ['1', 'UNX', '"UNKNOWN ATOM OR ION"']
        ['1', 'UNL', '"UNKNOWN LIGAND"']
        ['1', 'B3P', '"2-[3-(2-HYDROXY-1,1-DIHYDROXYMETHYL-ETHYLAMINO)-"']
        ['2', 'B3P', '" PROPYLAMINO]-2-HYDROXYMETHYL-PROPANE-1,3-DIOL"']
        '''
        list = []
        
        list.append(line[5:8])
        if(line[9] == ' '):
            list.append('"' + ' ' + line[9:].strip() + '"')
        else:
            list.append('"' + line[9:].strip() + '"')
        if(line[2:4] == '  '):
            list.insert(0,'1')
        else:
            list.insert(0,line[2:4].strip())        
                   
        return list

    def rebuild_HETSYN(self,line):
        '''
        ['1', 'HV5', '"3-METHYL-L-VALINE"']
        ['1', 'AB1', '"ABT-378; LOPINAVIR"']
        ['1', 'CMP', '"CYCLIC AMP; CAMP"']
        ['1', 'TRS', '"TRIS  BUFFER;"']
        '''
        list = []
        list.append(line[5:8].strip())
        if(line[9] == ' '):
            list.append('"' + ' ' + line[9:].strip() + '"')
        else:
            list.append('"' + line[9:].strip() + '"')
        if(line[2:4] == '  '):
            list.insert(0,'1')
        else:
            list.insert(0,line[2:4].strip())         
                
        return list

    def rebuild_FORMUL(self,line):
        '''
        ['3', 'MG', '1',' ', '2(MG 2+)']
        ['5', 'SO4', '1',' ', '6(O4 S 2-)']
        ['13', 'HOH','1', '*', '360(H2 O)']
        ['3', 'NAP', '1',' ', '2(C21 H28 N7 O17 P3)']
        ['4', 'FOL', '1',' ', '2(C19 H19 N7 O6)']
        ['5', '1PE', '1',' ', 'C10 H22 O6']
        ['2', 'NX5', '1',' ', 'C14 H10 O2 CL2 S']  
        '''
        list = []
        list = line[2:10].strip().split()
        if(line[10:12] == '  '):
            list.append('1')
        else:
            list.append(line[10:12].strip())
        list.append(line[12])
        list.append(line[13:].strip())          
          
        return list

    def rebuild_CAVEAT(self,line):
        '''
        ['1', '2UXK', 'xxxxxpabvdklabj']
        ['2', '2UXK', 'bbbbbldkbewpab,']
        ['3', '2UXK', 'aaaaadkbewpab']
        ['4', '2UXK', ' ppppdkbewpab']
        ['1', 'PWJS', 'eeeeebewpab']
         '''        
        list = []
        if(line[2:4] == '  '):
            list.append('1')
        else:
            list.append(line[2:4].strip())
        list.append(line[5:9])    
        if(line[13] == ' '):
            list.append(' ' + line[13:].strip())
        else:
            list.append(line[13:].strip())        
                     
        return list
    
    def rebuild_MTRIX1(self,line):
        '''
        ['1', '-1.000000', '0.000000', '0.000000', '0.00000', '1']
        ['1', '-1.000000', '0.000000', '0.000000', '0.00000', '']
        '''
        list = []
        key = 'MTRIXn'
        keyList = ['serial','m[n][1]','m[n][2]','m[n][3]','v[n]','iGiven']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list  
    
    def rebuild_MTRIX2(self,line):
        '''
        ['1', '-1.000000', '0.000000', '0.000000', '0.00000', '1']
        ['1', '-1.000000', '0.000000', '0.000000', '0.00000', '']
        '''        
        list = []
        key = 'MTRIXn'
        keyList = ['serial','m[n][1]','m[n][2]','m[n][3]','v[n]','iGiven']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_MTRIX3(self,line):
        '''
        ['1', '-1.000000', '0.000000', '0.000000', '0.00000', '1']
        ['1', '-1.000000', '0.000000', '0.000000', '0.00000', '']
        '''        
        list = []
        key = 'MTRIXn'
        keyList = ['serial','m[n][1]','m[n][2]','m[n][3]','v[n]','iGiven']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list    

    def rebuild_ORIGX1(self,line):
        '''
        ['0.963457', '0.136613', '0.230424', '16.61000']
        ['-0.158977', '0.983924', '0.081383', '13.72000']
        '''
        list = []
        key = 'ORIGXn'
        keyList = ['o[n][1]','o[n][2]','o[n][3]','t[n]']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list   
    
    def rebuild_ORIGX2(self,line):
        '''
        ['0.963457', '0.136613', '0.230424', '16.61000']
        ['-0.158977', '0.983924', '0.081383', '13.72000']
        '''
        list = []
        key = 'ORIGXn'
        keyList = ['o[n][1]','o[n][2]','o[n][3]','t[n]']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list     
    
    def rebuild_ORIGX3(self,line):
        '''
        ['0.963457', '0.136613', '0.230424', '16.61000']
        ['-0.158977', '0.983924', '0.081383', '13.72000']
        '''
        list = []
        key = 'ORIGXn'
        keyList = ['o[n][1]','o[n][2]','o[n][3]','t[n]']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list         
    
    def rebuild_SCALE1(self,line):
        '''
        ['0.963457', '0.136613', '0.230424', '16.61000']
        ['-0.158977', '0.983924', '0.081383', '13.72000']
        '''
        list = []
        key = 'SCALEn'
        keyList = ['s[n][1]','s[n][2]','s[n][3]','u[n]']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list  
    
    def rebuild_SCALE2(self,line):
        '''
        ['0.963457', '0.136613', '0.230424', '16.61000']
        ['-0.158977', '0.983924', '0.081383', '13.72000']
        '''
        list = []
        key = 'SCALEn'
        keyList = ['s[n][1]','s[n][2]','s[n][3]','u[n]']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list    
    
    def rebuild_SCALE3(self,line):
        '''
        ['0.963457', '0.136613', '0.230424', '16.61000']
        ['-0.158977', '0.983924', '0.081383', '13.72000']
        '''
        list = []
        key = 'SCALEn'
        keyList = ['s[n][1]','s[n][2]','s[n][3]','u[n]']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list    
    
    def rebuild_ATOM(self,line):
        '''
        ['40', 'CZ', 'A', 'ARG', 'A', '-3', '', '13.202', '84.534', '88.850', '0.50', '40.03', 'C', '']
        ['41', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', '']
        ['42', 'NH2', 'A', 'ARG', 'A', '-3', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', '']
        '''
        list = []
        key = 'ATOM'
        keyList = ['serial','name','altLoc','resName','chainID','resSeq','iCode',
                   'x','y','z','occupancy','tempFactor','element','charge']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list          
    
    def rebuild_ANISOU(self,line):
        '''
        ['110', 'O', '', 'GLY', 'A', '13', '', '3837', '2505', '1611', '164', '-121', '189', 'O', '']
        ['111', 'N', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', '']
        '''
        list = []
        key = 'ANISOU'
        keyList = ['serial','name','altLoc','resName','chainID','resSeq','iCode',
                   'u[0][0]','u[1][1]','u[2][2]','u[0][1]','u[0][2]','u[1][2]',
                   'element','charge']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list  

    def rebuild_SIGUIJ(self,line):
        '''
        ['110', 'O', '', 'GLY', 'A', '13', '', '10', '10', '10', '10', '10', '10', 'O', '']
        ['111', 'N', '', 'ASN', 'A', '14', '', '10', '10', '10', '10', '10', '10', 'N', '']
        '''
        list = []
        key = 'ANISOU'
        keyList = ['serial','name','altLoc','resName','chainID','resSeq','iCode',
                   'u[0][0]','u[1][1]','u[2][2]','u[0][1]','u[0][2]','u[1][2]',
                   'element','charge']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_HETATM(self,line):
        '''
        ['40', 'CZ', 'A', 'ARG', 'A', '-3', '', '13.202', '84.534', '88.850', '0.50', '40.03', 'C', '']
        ['41', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', '']
        ['42', 'NH2', 'A', 'ARG', 'A', '-3', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', '']
        '''
        list = []
        key = 'HETATM'
        keyList = ['serial','name','altLoc','resName','chainID','resSeq','iCode',
                   'x','y','z','occupancy','tempFactor','element','charge']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
        
    def rebuild_CONECT(self,line):
        '''
        ['1179', '746', '1184', '1195', '1203']
        ['1179', '1211', '1222', '', '']
        '''
        list = []
        key = 'CONECT'
        keyList = ['serial_1','serial_2','serial_3','serial_4','serial_5']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list         
    
    def rebuild_MASTER(self,line):
        '''
        ['40', '0', '0', '0', '0', '0', '0', '6', '2930', '2', '0', '29']
        '''
        list = []
        key = 'MASTER'
        keyList = ['numRemark','0','numHet','numHelix','numSheet','numTurn',
                   'numSite','numXform','numCoord','numTer','numConect','numSeq']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list  
    
    def rebuild_OBSLTE(self,line):          
        list = []
        key = 'OBSLTE'
        keyList = ['continuation','repDate','idCode','ridCode_1','ridCode_2',
                   'ridCode_3','ridCode_4','ridCode_5','ridCode_6','ridCode_7',
                   'ridCode_8','ridCode_9']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
     
    def rebuild_SSBOND(self,line):   
        list = []
        key = 'SSBOND'
        keyList = ['serNum','CYS_1','chainID1','seqNum1','icode1','CYS_2','chainID2',
                   'seqNum2','icode2','sym1','sym2','Length']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_LINK(self,line):   
        list = []
        key = 'LINK'
        keyList = ['name1','altLoc1','resName1','chainID1','resSeq1','iCode1','name2',
                   'altLoc2','resName2','chainID2','resSeq2','iCode2','sym1','sym2',
                   'Length']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_CISPEP(self,line):   
        list = []
        key = 'CISPEP'
        keyList = ['serNum','pep1','chainID1','seqNum1','icode1','pep2','chainID2',
                   'seqNum2','icode2','modNum','measure']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_SITE(self,line):   
        list = []
        key = 'SITE'
        keyList = ['seqNum','siteID','numRes','resName1','chainID1','seq1','iCode1',
                   'resName2','chainID2','seq2','iCode2','resName3','chainID3',
                   'seq3','iCode3','resName4','chainID4','seq4','iCode4']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_HELIX(self,line):   
        list = []
        key = 'HELIX'
        keyList = ['serNum','helixID','initResName','initChainID','initSeqNum',
                   'initICode','endResName','endChainID','endSeqNum','endICode',
                   'helixClass','comment','length']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_SHEET(self,line):   
        list = []
        key = 'SHEET'
        keyList = ['strand','sheetID','numStrands','initResName','initChainID',
                   'initSeqNum','initICode','endResName','endChainID','endSeqNum',
                   'endICode','sense','curAtom','curResName','curChainId','curResSeq',
                   'curICode','prevAtom','prevResName','prevChainId','prevResSeq',
                   'prevICode']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
        
    def rebuild_HET(self,line):   
        list = []
        key = 'HET'
        keyList = ['hetID','ChainID','seqNum','iCode','numHetAtoms','text']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
      
    def rebuild_MODRES(self,line):   
        list = []
        key = 'MODRES'
        keyList = ['idCode','resName','chainID','seqNum','iCode','stdRes','comment']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_SEQADV(self,line):   
        list = []
        key = 'SEQADV'
        keyList = ['idCode','resName','chainID','seqNum','iCode','database',
                   'dbAccession','dbRes','dbSeq','conflict']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_DBREF(self,line):   
        list = []
        key = 'DBREF'
        keyList = ['idCode','chainID','seqBegin','insertBegin','seqEnd','insertEnd',
                   'database','dbAccession','dbIdCode','dbseqBegin','idbnsBeg',
                   'dbseqEnd','dbinsEnd']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_DBREF1(self,line):   
        list = []
        key = 'DBREF1'
        keyList = ['idCode','chainID','seqBegin','insertBegin','seqEnd','insertEnd',
                   'database','dbIdCode']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_DBREF2(self,line):   
        list = []
        key = 'DBREF2'
        keyList = ['idCode','chainID','dbAccession','seqBegin','seqEnd']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
    
    def rebuild_REVDAT(self,line):   
        list = []
        key = 'REVDAT'
        keyList = ['modNum','continuation','modDate','modId','modType','record_1',
                   'record_2','record_3','record_4']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list          

    def rebuild_CRYST1(self,line):
        list = []
        key = 'CRYST1'
        keyList = ['a','b','c','alpha','beta','gamma','sGroup','z']
        for k in keyList:
            if len(self.load_dict[key][k]) == 2:
                list.append(line[self.load_dict[key][k][0]-7:self.load_dict[key][k][1]-6].strip())
            else:
                list.append(line[self.load_dict[key][k][0]-7].strip())
        return list
        
    def rebuild_REMARK(self,line):   
        list = []
        list.append(line[1:4].strip())
        list.append(line[5:].strip())
        
        return list    
    
    #----------------------------------------------------------------------
    def list2dict_OBSLTE(self,list):
        part_dict = {}
        count = 0
        flag = 0
        for row in list:
            if row[0] == 'OBSLTE':
                count += 1
                length = len(row)
                if count == 1:
                    flag = 1
                part_dict['obslteDate'] = row[1]
                part_dict['IDcode'] = []
                for i in range(2 + flag,length):
                    part_dict['IDcode'].append(row[i])
        return part_dict

    def list2dict_TITLE(self,list):
        part_dict = {}
        count = 0
        for row in list:
            if row[0] == 'TITLE':
                count += 1
                length = len(row)
                if count == 1:
                    part_dict['title'] = row[1]
                for i in range(2,length):
                    part_dict['title'] += row[i] + ' '
        return part_dict
    
    def list2dict_SPLIT(self,list):
        part_dict = {}
        count = 0
        flag = 0
        for row in list:
            if row[0] == 'SPLIT':
                count += 1
                length = len(row)
                if count == 1:
                    flag = 1
                part_dict['IDcode'] = []
                for i in range(1 + flag,length):
                    part_dict['IDcode'].append(row[i])
            
        return part_dict
    
    def list2dict_NUMMDL(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'NUMMDL':
                part_dict['model_number'] = row[1]
        return part_dict 

    def list2dict_REVDAT(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'REVDAT':
                key = 'modification_' + row[1]
                if key not in part_dict.keys():
                    part_dict[key] = {}
                    part_dict[key]['modDate'] = row[3]
                    part_dict[key]['modID'] = row[4]
                    part_dict[key]['modType'] = row[5]
                if 'record' not in part_dict[key].keys():
                    part_dict[key]['record'] = []
                for i in row[6:]:
                    if i:
                        part_dict[key]['record'].append(i)
        return part_dict 
        
    def list2dict_SPRSDE(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'SPRSDE':
                part_dict['sprsdeDate'] = row[1]
                part_dict['idCode'] = row[2]
                if len(row) == 4:
                    part_dict['sIdCode'] = row[3]
                else:
                    part_dict['sIdCode'] = row[3:-1]
        return part_dict 
    
    def list2dict_DBREF(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'DBREF':
                key = 'idCode_' + row[1]
                if key not in part_dict.keys():
                    part_dict[key] = []
                d = {}
                d['chainID'] = row[2]
                d['seqBegin'] = row[3]
                d['insertBegin'] = row[4]
                d['seqEnd'] = row[5]
                d['insertEnd'] = row[6]
                d['database'] = row[7]
                d['dbAccession'] = row[8]
                d['dbIdCode'] = row[9]
                d['dbseqBegin'] = row[10]
                d['idbnsBeg'] = row[11]
                d['dbseqEnd'] = row[12]
                d['dbinsEnd'] = row[13]
                part_dict[key].append(d)
        return part_dict 
    
    def list2dict_DBREF1(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'DBREF1':
                key = 'idCode_' + row[1]
                if key not in part_dict.keys():
                    part_dict[key] = []
                d = {}
                d['chainID'] = row[2]
                d['seqBegin'] = row[3]
                d['insertBegin'] = row[4]
                d['seqEnd'] = row[5]
                d['insertEnd'] = row[6]
                d['database'] = row[7]
                d['dbIdCode'] = row[8]
                part_dict[key].append(d)
        return part_dict     
    
    def list2dict_DBREF2(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'DBREF':
                key = 'idCode_' + row[1]
                if key not in part_dict.keys():
                    part_dict[key] = []
                d = {}
                d['chainID'] = row[2]
                d['dbAccession'] = row[3]
                d['sqeBegin'] = row[4]
                d['sqeEnd'] = row[5]
                part_dict[key].append(d)
        return part_dict  
    
    def list2dict_SEQADV(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'SEQADV':
                key = 'idCode_' + row[1]
                if key not in part_dict.keys():
                    part_dict[key] = []
                d = {}
                d['resName'] = row[2]
                d['chainID'] = row[3]
                d['seqNum'] = row[4]
                d['iCode'] = row[5]
                d['database'] = row[6]
                d['dbAccession'] = row[7]
                d['dbRes'] = row[8]
                d['dbSeq'] = row[9]
                d['conflict'] = row[10]
                part_dict[key].append(d)
        return part_dict 
    
    def list2dict_SEQRES(self,list):
        part_dict = {}
        count = 0
        for row in list:
            if row[0] == 'SEQRES':
                if row[1] == '1':
                    count += 1
                key = 'SEQRES_' + str(count)
                if key not in part_dict.keys():
                    part_dict[key] = {}
                    part_dict[key]['chainID'] = row[2]
                    part_dict[key]['numRes'] = row[3]
                    part_dict[key]['resName'] = row[4:]
                else :
                    for i in row[4:]:
                        part_dict[key]['resName'].append(i)            
        return part_dict  
    
    def list2dict_MODRES(self,list):
        part_dict = {}
        count = 0
        for row in list:
            if row[0] == 'MODRES':
                count += 1
                key = 'MODRES_' + str(count)
                part_dict[key] = {}
                part_dict[key]['idCode'] = row[1]
                part_dict[key]['resName'] = row[1]
                part_dict[key]['chainID'] = row[1]
                part_dict[key]['seqNum'] = row[1]
                part_dict[key]['insertion_code'] = row[1]
                part_dict[key]['stdRes'] = row[1]
                part_dict[key]['comment'] = row[1]           
        return part_dict  
    
    def list2dict_HET(self,list):
        part_dict = {}
        count = 0
        for row in list:
            count += 1
            if row[0] == 'HET':
                key = 'HET_' + str(count)
                part_dict[key] = {}
                part_dict[key]['hetID'] = row[1]
                part_dict[key]['ChainID'] = row[2]
                part_dict[key]['seqNum'] = row[3]
                part_dict[key]['iCode'] = row[4]
                part_dict[key]['numHetAtoms'] = row[5]
                part_dict[key]['text'] = row[6]
        return part_dict 
    
    def list2dict_HELIX(self,list):
        part_dict = {}
        HELIX_class = {'1':'Right-handed alpha (default)','2':'Right-handed omega',
                       '3':'Right-handed pi','4':'Right-handed gamma','5':'Right-handed 3 - 10',
                       '6':'Left-handed alpha','7':'Left-handed omega','8':'Left-handed gamma',
                       '9':'2 - 7 ribbon/helix','10':'Polyproline'}
        count = 1
        for row in list:
            if row[0] == 'HELIX':
                key = 'HELIX_' + str(row[1])
                d = {}
                d['helixID'] = row[2]
                d['initResName'] = row[3]
                d['initChainID'] = row[4]
                d['initSeqNum'] = row[5]
                d['initICode'] = row[6]
                d['endResName'] = row[7]
                d['endChainID'] = row[8]
                d['endSeqNum'] = row[9]
                d['endICode'] = row[10]
                if row[11] in HELIX_class.keys():
                    d['helixClass'] = HELIX_class[row[11]]
                else:
                    d['helixClass'] = [row[11],'not defined,may be an error']
                d['comment'] = row[12]
                d['length'] = row[13]
                part_dict[key] = d      
        return part_dict 
    
    def list2dict_SHEET(self,list):
        part_dict = {}
        count = 1
        for row in list:
            if row[0] == 'SHEET':
                key = 'sheetID_' + str(count) + '_' +str(row[2])
                if key not in part_dict.keys():
                    part_dict[key] = []                    
                d = {}
                d['numStrands'] = row[3]
                d['initResName'] = row[4]
                d['initChainID'] = row[5]
                d['initSeqNum'] = row[6]
                d['initICode'] = row[7]
                d['endResName'] = row[8]
                d['endChainID'] = row[9]
                d['endSeqNum'] = row[10]
                d['endICode'] = row[11]
                d['sense'] = row[12]
                if row[12] == 0:
                    part_dict[key].append(d)
                    continue
                d['curAtom'] = row[13]
                d['curResName'] = row[14]
                d['curChainId'] = row[15]
                d['curResSeq'] = row[16]
                d['curICode'] = row[17]
                d['prevAtom'] = row[18]
                d['prevResName'] = row[19]
                d['prevChainId'] = row[20]
                d['prevResSeq'] = row[21]
                d['prevICode'] = row[22]
                part_dict[key].append(d)
                if row[1] == row[3]:
                    count += 1                
        return part_dict 

    
    def list2dict_SSBOND(self,list):
        part_dict = {}
        count = 0
        for row in list:
            if row[0] == 'SSBOND':
                count += 1
                key = 'SSBOND_' + str(count)
                part_dict[key] = {}
                part_dict[key]['CYS1'] = row[2]
                part_dict[key]['chainID1'] = row[3]
                part_dict[key]['seqNum1'] = row[4]
                part_dict[key]['iCode1'] = row[5]
                part_dict[key]['CYS2'] = row[6]
                part_dict[key]['chainID2'] = row[7]
                part_dict[key]['seqNum2'] = row[8]
                part_dict[key]['iCode2'] = row[9]
                part_dict[key]['sym1'] = row[10]
                part_dict[key]['sym2'] = row[11]
                part_dict[key]['Length'] = row[12]               
        return part_dict 
    
    def list2dict_LINK(self,list):
        part_dict = {}
        count = 0
        for row in list:
            if row[0] == 'LINK':
                count += 1
                key = 'LINK_' + str(count)
                part_dict[key] = {}
                part_dict[key]['AtomName1'] = row[2]
                part_dict[key]['altLoc1'] = row[3]
                part_dict[key]['resName1'] = row[4]
                part_dict[key]['chainID1'] = row[5]
                part_dict[key]['resSeq1'] = row[6]
                part_dict[key]['iCode1'] = row[7]
                part_dict[key]['AtomName2'] = row[8]
                part_dict[key]['altLoc2'] = row[9]
                part_dict[key]['resName2'] = row[10]
                part_dict[key]['chainID2'] = row[11]
                part_dict[key]['resSeq2'] = row[12]
                part_dict[key]['iCode2'] = row[13]
                part_dict[key]['sym1'] = row[14]
                part_dict[key]['sym2'] = row[15]
                part_dict[key]['Length'] = row[16]               
        return part_dict 
    
    def list2dict_CISPEP(self,list):
        part_dict = {}
        count = 0
        for row in list:
            if row[0] == 'CISPEP':
                count += 1
                key = 'CISPEP_' + str(count)
                part_dict[key] = {}
                part_dict[key]['pep1'] = row[2]
                part_dict[key]['chainID1'] = row[3]
                part_dict[key]['seqNum1'] = row[4]
                part_dict[key]['iCode1'] = row[5]
                part_dict[key]['pep2'] = row[6]
                part_dict[key]['chainID2'] = row[7]
                part_dict[key]['seqNum2'] = row[8]
                part_dict[key]['iCode2'] = row[9]
                part_dict[key]['modNum'] = row[10]
                part_dict[key]['measureAngle'] = row[11]               
        return part_dict 
    
    def list2dict_SITE(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'SITE':
                length = len(row)
                count = 0
                if row[3] == 1:
                    key = row[2]
                    part_dict[key] = {}
                    part_dict[key]['resName'] = row[4]
                    part_dict[key]['chainID'] = row[5]
                    part_dict[key]['seq'] = row[6]
                else:
                    if row[2] not in part_dict.keys():
                        key = row[2]
                        part_dict[key] = []
                    for i in range(4,length-1,3):
                        temp = {}
                        count += 1
                        temp['resName'+str(count)] = row[i]
                        temp['chainID'+str(count)] = row[i+1]
                        temp['seq'+str(count)] = row[i+2]
                        part_dict[key].append(temp)                
        return part_dict  
    
    def list2dict_CRYST1(self,list):
        '''
        list = [['CRYST1','52.000','58.600','61.900','90.00','90.00','90.00', 'P', '21', '21', '21',    '8'], 
               ['CRYST1','52.000','58.600','61.900','90.00','90.00','90.00', 'P', '1','8']]
        part_dict = {'CRYST1_1': {'Z_value': '8',
                                  'a(Angstroms)': '52.000',
                                  'alpha(degrees)': '90.00',
                                  'b(Angstroms)': '58.600',
                                  'beta(Angstroms)': '90.00',
                                  'c(Angstroms)': '61.900',
                                  'gamma(Angstroms)': '90.00',
                                  'spaceGroup': 'P 21 21 21'},
                    'CRYST1_2': {'Z_value': '8',
                                  'a(Angstroms)': '52.000',
                                  'alpha(degrees)': '90.00',
                                  'b(Angstroms)': '58.600',
                                  'beta(Angstroms)': '90.00',
                                  'c(Angstroms)': '61.900',
                                  'gamma(Angstroms)': '90.00',
                                  'spaceGroup': 'P 1'}}

        '''
        part_dict = {}
        
        cryst1_count = 1
        
        for row in list:
            if(row[0] == 'CRYST1'):
                cryst1_dict = {}
                cryst1_dict['a(Angstroms)'] = row[1]
                cryst1_dict['b(Angstroms)'] = row[2]
                cryst1_dict['c(Angstroms)'] = row[3]
                cryst1_dict['alpha(degrees)'] = row[4]
                cryst1_dict['beta(Angstroms)'] = row[5]
                cryst1_dict['gamma(Angstroms)'] = row[6]
                cryst1_dict['spaceGroup'] =  row[7]
                for i in range(8,len(row)-1):
                    cryst1_dict['spaceGroup'] += ' ' + row[i]
                cryst1_dict['Z_value'] = row[-1]
                part_dict['CRYST1_' + str(cryst1_count)] = cryst1_dict
                cryst1_count += 1          
        
        return part_dict   
    
    def list2dict_MTRIX1(self,list):
        '''
        list = [['MTRIX1','1','-1.000000','0.000000','0.000000','0.00000','1'],          
                ['MTRIX1','1','-1.000000','0.000000','0.000000','0.00000']]
        part_dict = {'MTRIX1_1': {'Mn1': '-1.000000',
                                 'Mn2': '0.000000',
                                 'Mn3': '0.000000',
                                 'Serial_Number': '1',
                                 'Vn': '0.00000',
                                 'iGiven': '1'},
                    'MTRIX1_2': {'Mn1': '-1.000000',
                                 'Mn2': '0.000000',
                                 'Mn3': '0.000000',
                                 'Serial_Number': '1',
                                 'Vn': '0.00000',
                                 'iGiven': '0'}}
        '''      
        part_dict = {}
        
        mtrix_count = 1
        for row in list:
            if(row[0] == 'MTRIX1'):
                mtrix_dict = {}
                mtrix_dict['Serial_Number'] = row[1]
                mtrix_dict['Mn1'] = row[2]
                mtrix_dict['Mn2'] = row[3]
                mtrix_dict['Mn3'] = row[4]
                mtrix_dict['Vn'] = row[5]
                if(row[-1] == '1'):
                    mtrix_dict['iGiven'] = row[-1]
                else:
                    mtrix_dict['iGiven'] = '0'
                part_dict['MTRIX1_' + str(mtrix_count)] = mtrix_dict
                mtrix_count += 1
                
        return part_dict  
    
    def list2dict_MTRIX2(self,list):
        '''
        list = [['MTRIX2','1','-1.000000','0.000000','0.000000','0.00000','1'],          
                ['MTRIX2','1','-1.000000','0.000000','0.000000','0.00000']]
        part_dict = {'MTRIX2_1': {'Mn1': '-1.000000',
                                 'Mn2': '0.000000',
                                 'Mn3': '0.000000',
                                 'Serial_Number': '1',
                                 'Vn': '0.00000',
                                 'iGiven': '1'},
                    'MTRIX2_2': {'Mn1': '-1.000000',
                                 'Mn2': '0.000000',
                                 'Mn3': '0.000000',
                                 'Serial_Number': '1',
                                 'Vn': '0.00000',
                                 'iGiven': '0'}}
        '''      
        part_dict = {}
        
        mtrix_count = 1
        for row in list:
            if(row[0] == 'MTRIX2'):
                mtrix_dict = {}
                mtrix_dict['Serial_Number'] = row[1]
                mtrix_dict['Mn1'] = row[2]
                mtrix_dict['Mn2'] = row[3]
                mtrix_dict['Mn3'] = row[4]
                mtrix_dict['Vn'] = row[5]
                if(row[-1] == '1'):
                    mtrix_dict['iGiven'] = row[-1]
                else:
                    mtrix_dict['iGiven'] = '0'
                part_dict['MTRIX2_' + str(mtrix_count)] = mtrix_dict
                mtrix_count += 1        

        return part_dict 
    
    def list2dict_MTRIX3(self,list):
        '''
        list = [['MTRIX3','1','-1.000000','0.000000','0.000000','0.00000','1'],          
                ['MTRIX3','1','-1.000000','0.000000','0.000000','0.00000']]
        part_dict = {{'MTRIX3_1': {'Mn1': '-1.000000',
                                 'Mn2': '0.000000',
                                 'Mn3': '0.000000',
                                 'Serial_Number': '1',
                                 'Vn': '0.00000',
                                 'iGiven': '1'},
                    'MTRIX3_2': {'Mn1': '-1.000000',
                                 'Mn2': '0.000000',
                                 'Mn3': '0.000000',
                                 'Serial_Number': '1',
                                 'Vn': '0.00000',
                                 'iGiven': '0'}}
        '''      
        part_dict = {}
        
        mtrix_count = 1
        for row in list:
            if(row[0] == 'MTRIX3'):
                mtrix_dict = {}
                mtrix_dict['Serial_Number'] = row[1]
                mtrix_dict['Mn1'] = row[2]
                mtrix_dict['Mn2'] = row[3]
                mtrix_dict['Mn3'] = row[4]
                mtrix_dict['Vn'] = row[5]
                if(row[-1] == '1'):
                    mtrix_dict['iGiven'] = row[-1]
                else:
                    mtrix_dict['iGiven'] = '0'
                part_dict['MTRIX3_' + str(mtrix_count)] = mtrix_dict
                mtrix_count += 1        
          
        return part_dict 
    
    def list2dict_ORIGX1(self,list):
        ''' 
        list = [['ORIGX1','0.963457', '0.136613', '0.230424', '16.61000'],
                ['ORIGX1','-0.158977', '0.983924', '0.081383', '13.72000']]
        part_dict = {'ORIGX1_1': {'o[n][1]': '0.963457',
                                  'o[n][2]': '0.136613',
                                  'o[n][3]': '0.230424',
                                  't[n]': '16.61000'},
                    'ORIGX1_2': {'o[n][1]': '-0.158977',
                                  'o[n][2]': '0.983924',
                                  'o[n][3]': '0.081383',
                                  't[n]': '13.72000'}}
        ''' 
        part_dict = {}
        
        origx_count = 1
        for row in list:
            if(row[0] == 'ORIGX1'):
                origx_dict = {}
                origx_dict['o[n][1]'] = row[1]
                origx_dict['o[n][2]'] = row[2]
                origx_dict['o[n][3]'] = row[3]
                origx_dict['t[n]'] = row[4]
                part_dict['ORIGX1_' + str(origx_count)] = origx_dict
                origx_count += 1      
                
        return part_dict 
    
    def list2dict_ORIGX2(self,list):
        ''' 
        list = [['ORIGX2','0.963457', '0.136613', '0.230424', '16.61000'],
                ['ORIGX2','-0.158977', '0.983924', '0.081383', '13.72000']]
        part_dict = {'ORIGX2_1': {'o[n][1]': '0.963457',
                                  'o[n][2]': '0.136613',
                                  'o[n][3]': '0.230424',
                                  't[n]': '16.61000'},
                    'ORIGX2_2': {'o[n][1]': '-0.158977',
                                  'o[n][2]': '0.983924',
                                  'o[n][3]': '0.081383',
                                  't[n]': '13.72000'}}
        ''' 
        part_dict = {}
        
        origx_count = 1
        for row in list:
            if(row[0] == 'ORIGX2'):
                origx_dict = {}
                origx_dict['o[n][1]'] = row[1]
                origx_dict['o[n][2]'] = row[2]
                origx_dict['o[n][3]'] = row[3]
                origx_dict['t[n]'] = row[4]
                part_dict['ORIGX2_' + str(origx_count)] = origx_dict                
                origx_count += 1      
                
        return part_dict         
    
    def list2dict_ORIGX3(self,list):
        ''' 
        list = [['ORIGX3','0.963457', '0.136613', '0.230424', '16.61000'],
                ['ORIGX3','-0.158977', '0.983924', '0.081383', '13.72000']]
        part_dict = {'ORIGX3_1': {'o[n][1]': '0.963457',
                                  'o[n][2]': '0.136613',
                                  'o[n][3]': '0.230424',
                                  't[n]': '16.61000'},
                    'ORIGX3_2': {'o[n][1]': '-0.158977',
                                  'o[n][2]': '0.983924',
                                  'o[n][3]': '0.081383',
                                  't[n]': '13.72000'}}
        ''' 
        part_dict = {}
        
        origx_count = 1
        for row in list:
            if(row[0] == 'ORIGX3'):
                origx_dict = {}
                origx_dict['o[n][1]'] = row[1]
                origx_dict['o[n][2]'] = row[2]
                origx_dict['o[n][3]'] = row[3]
                origx_dict['t[n]'] = row[4]
                part_dict['ORIGX3_' + str(origx_count)] = origx_dict                
                origx_count += 1      
                
        return part_dict             
    
    def list2dict_SCALE1(self,list):
        ''' 
        list = [['SCALE1','0.963457', '0.136613', '0.230424', '16.61000'],
                ['SCALE1','-0.158977', '0.983924', '0.081383', '13.72000']]
          part_dict = {'SCALE1_1': {'s[n][1]': '0.963457',
                                  's[n][2]': '0.136613',
                                  's[n][3]': '0.230424',
                                  'u[n]': '16.61000'},
                    'SCALE1_2': {'s[n][1]': '-0.158977',
                                  's[n][2]': '0.983924',
                                  's[n][3]': '0.081383',
                                  'u[n]': '13.72000'}
        ''' 
        part_dict = {}
        
        scale_count = 1
        for row in list:
            if(row[0] == 'SCALE1'):
                scale_dict = {}
                scale_dict['s[n][1]'] = row[1]
                scale_dict['s[n][2]'] = row[2]
                scale_dict['s[n][3]'] = row[3]
                scale_dict['u[n]'] = row[4]
                part_dict['SCALE1_' + str(scale_count)] = scale_dict
                scale_count += 1      
                
        return part_dict             
    
    def list2dict_SCALE2(self,list):
        ''' 
        list = [['SCALE2','0.963457', '0.136613', '0.230424', '16.61000'],
                ['SCALE2','-0.158977', '0.983924', '0.081383', '13.72000']]
          part_dict = {'SCALE2_1': {'s[n][1]': '0.963457',
                                  's[n][2]': '0.136613',
                                  's[n][3]': '0.230424',
                                  'u[n]': '16.61000'},
                    'SCALE2_2': {'s[n][1]': '-0.158977',
                                  's[n][2]': '0.983924',
                                  's[n][3]': '0.081383',
                                  'u[n]': '13.72000'}
        ''' 
        part_dict = {}
        
        scale_count = 1
        for row in list:
            if(row[0] == 'SCALE2'):
                scale_dict = {}
                scale_dict['s[n][1]'] = row[1]
                scale_dict['s[n][2]'] = row[2]
                scale_dict['s[n][3]'] = row[3]
                scale_dict['u[n]'] = row[4]
                part_dict['SCALE2_' + str(scale_count)] = scale_dict
                scale_count += 1                 
                
        return part_dict          
    
    def list2dict_SCALE3(self,list):
        ''' 
        list = [['SCALE3','0.963457', '0.136613', '0.230424', '16.61000'],
                ['SCALE3','-0.158977', '0.983924', '0.081383', '13.72000']]
        part_dict = {'SCALE3_1': {'s[n][1]': '0.963457',
                                  's[n][2]': '0.136613',
                                  's[n][3]': '0.230424',
                                  'u[n]': '16.61000'},
                    'SCALE3_2': {'s[n][1]': '-0.158977',
                                  's[n][2]': '0.983924',
                                  's[n][3]': '0.081383',
                                  'u[n]': '13.72000'}}
        ''' 
        part_dict = {}
        
        scale_count = 1
        for row in list:
            if(row[0] == 'SCALE3'):
                scale_dict = {}
                scale_dict['s[n][1]'] = row[1]
                scale_dict['s[n][2]'] = row[2]
                scale_dict['s[n][3]'] = row[3]
                scale_dict['u[n]'] = row[4]
                part_dict['SCALE3_' + str(scale_count)] = scale_dict
                scale_count += 1              
                
        return part_dict           
    
    def list2dict_ATOM(self,list):
        ''' 
        list = [['MODEL','1'],
        ['ATOM','41', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','42','GLU','A','18'], 
        ['HETATM','43', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','44','GLU','A','18'],        
        ['HETATM','110', 'O', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','110', 'O', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','110', 'O', '', 'ASN', 'A', '14', '', '1', '1', '1', '1', '1', '1', 'N', ''],        
        ['ATOM','111', 'N', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','111', 'N', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','111', 'N', '', 'ASN', 'A', '14', '', '0', '0', '0', '0', '0', '0', 'N', ''],
        ['ENDMDL'],
        ['MODEL','2'],
        ['ATOM','41', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','42','GLU','A','18'],  
        ['HETATM','43', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','44','GLU','A','18'],
        ['HETATM','110', 'O', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','110', 'O', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','110', 'O', '', 'ASN', 'A', '14', '', '11', '11', '11', '11', '11', '11', 'N', ''],        
        ['ATOM','111', 'N', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','111', 'N', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','111', 'N', '', 'ASN', 'A', '14', '', '10', '10', '10', '10', '10', '10', 'N', ''],        
        ['ENDMDL']
        ]
        
        part_dict = {
        'model_1': {'ATOM_41': 
                   {'Model_No': '1', 'Chain_No': '1', 'serial_No': '41', 'name': 'NH1', 'altLoc': 'A', 
                   'resName': 'ARG', 'chainID': 'A', 'resSeq': '-3', 'iCode': '', 'x': '12.218', 'y': '84.840', 'z': '88.007', 
                   'occupancy': '0.50', 'tempFactor': '40.76', 'element': 'N', 'charge': ''}, 
                   'ATOM_111': 
                   {'Model_No': '1', 'Chain_No': '3', 'serial_No': '111', 'name': 'N', 'altLoc': ' ', 
                   'resName': 'ASN', 'chainID': 'A', 'resSeq': '14', 'iCode': '', 'x': '14.421', 'y': '84.308', 'z': '88.373', 
                   'occupancy': '0.50', 'tempFactor': '40.45', 'element': 'N', 'charge': '', 
                   'ANISOU': {'u[0][0]': '2059', 'u[1][1]': '1674', 'u[2][2]': '1462', 'u[0][1]': '27', 'u[0][2]': '244', 'u[1][2]': '-96'}, 
                   'SIGUIJ': {'u[0][0]': '0', 'u[1][1]': '0', 'u[2][2]': '0', 'u[0][1]': '0', 'u[0][2]': '0', 'u[1][2]': '0'}}}, 
        'model_2': {'ATOM_41': 
                   {'Model_No': '2', 'Chain_No': '1', 'serial_No': '41', 'name': 'NH1', 'altLoc': 'A', 
                   'resName': 'ARG', 'chainID': 'A', 'resSeq': '-3', 'iCode': '', 'x': '12.218', 'y': '84.840', 'z': '88.007', 
                   'occupancy': '0.50', 'tempFactor': '40.76', 'element': 'N', 'charge': ''}, 
                   'ATOM_111': 
                   {'Model_No': '2', 'Chain_No': '3', 'serial_No': '111', 'name': 'N', 'altLoc': ' ', 
                   'resName': 'ASN', 'chainID': 'A', 'resSeq': '14', 'iCode': '', 'x': '14.421', 'y': '84.308', 'z': '88.373', 
                   'occupancy': '0.50', 'tempFactor': '40.45', 'element': 'N', 'charge': '', 
                   'ANISOU': {'u[0][0]': '2059', 'u[1][1]': '1674', 'u[2][2]': '1462', 'u[0][1]': '27', 'u[0][2]': '244', 'u[1][2]': '-96'}, 
                   'SIGUIJ': {'u[0][0]': '10', 'u[1][1]': '10', 'u[2][2]': '10', 'u[0][1]': '10', 'u[0][2]': '10', 'u[1][2]': '10'}}}}
           
        '''         
        part_dict = {}
        
        flag = 0
        
        for i in range(0,len(list)):
            if(list[i][0] == 'MODEL'):
                model_dict = {}
                flag = 1
                Model_No = list[i][1]
                chain_count = 1
                for end in range(i + 1,len(list)):
                    if(list[end][0] == 'ENDMDL'):
                        break
                    if(list[end][0] == 'ATOM'):
                        atom_dict = {}
                        atom_dict['Model_No'] = Model_No
                        atom_dict['Chain_No'] = '1'
                        atom_dict['serial_No'] = list[end][1]
                        atom_dict['name'] = list[end][2]
                        atom_dict['altLoc'] = list[end][3]
                        atom_dict['resName'] = list[end][4]
                        atom_dict['chainID'] = list[end][5]
                        atom_dict['resSeq'] = list[end][6]
                        atom_dict['iCode'] = list[end][7]
                        atom_dict['x'] = list[end][8]
                        atom_dict['y'] = list[end][9]
                        atom_dict['z'] = list[end][10]
                        atom_dict['occupancy'] = list[end][11]
                        atom_dict['tempFactor'] = list[end][12]
                        atom_dict['element'] = list[end][13]
                        atom_dict['charge'] = list[end][14]
                        model_dict['ATOM_' + list[end][1]] = atom_dict
                        part_dict['model_' + Model_No] = model_dict
        
                    if(list[end][0] == 'ANISOU'):
                        anisow_dict = {}
                        anisow_dict['u[0][0]'] = list[end][8]
                        anisow_dict['u[1][1]'] = list[end][9]
                        anisow_dict['u[2][2]'] = list[end][10]
                        anisow_dict['u[0][1]'] = list[end][11]
                        anisow_dict['u[0][2]'] = list[end][12]
                        anisow_dict['u[1][2]'] = list[end][13]
                        if('ATOM_' + list[end][1] in model_dict):
                            model_dict['ATOM_' + list[end][1]]['ANISOU'] = anisow_dict
        
                    if(list[end][0] == 'SIGUIJ'):
                        siguij_dict = {}
                        siguij_dict['u[0][0]'] = list[end][8]
                        siguij_dict['u[1][1]'] = list[end][9]
                        siguij_dict['u[2][2]'] = list[end][10]
                        siguij_dict['u[0][1]'] = list[end][11]
                        siguij_dict['u[0][2]'] = list[end][12]
                        siguij_dict['u[1][2]'] = list[end][13]  
                        if('ATOM_' + list[end][1] in model_dict):
                            model_dict['ATOM_' + list[end][1]]['SIGUIJ'] = siguij_dict            
        
                    if(list[end][0] == 'ATOM' and 'ATOM_' + list[end][1] in model_dict):
                        model_dict['ATOM_' + list[end][1]]['Chain_No'] = str(chain_count)
                    if(list[end][0] == 'TER'):
                        chain_count += 1
                    end += 1
        
        
        if(flag == 0):
            model_dict = {}
            for row in list:
                if(row[0] == 'ATOM'):
                    atom_dict = {}
                    atom_dict['Model_No'] = '1'
                    atom_dict['Chain_No'] = '1'
                    atom_dict['serial_No'] = row[1]
                    atom_dict['name'] = row[2]
                    atom_dict['altLoc'] = row[3]
                    atom_dict['resName'] = row[4]
                    atom_dict['chainID'] = row[5]
                    atom_dict['resSeq'] = row[6]
                    atom_dict['iCode'] = row[7]
                    atom_dict['x'] = row[8]
                    atom_dict['y'] = row[9]
                    atom_dict['z'] = row[10]
                    atom_dict['occupancy'] = row[11]
                    atom_dict['tempFactor'] = row[12]
                    atom_dict['element'] = row[13]
                    atom_dict['charge'] = row[14]
                    model_dict['ATOM_' + row[1]] = atom_dict
        
            for row in list:
                if(row[0] == 'ANISOU'):
                    anisow_dict = {}
                    anisow_dict['u[0][0]'] = row[8]
                    anisow_dict['u[1][1]'] = row[9]
                    anisow_dict['u[2][2]'] = row[10]
                    anisow_dict['u[0][1]'] = row[11]
                    anisow_dict['u[0][2]'] = row[12]
                    anisow_dict['u[1][2]'] = row[13]
                    if('ATOM_' + row[1] in model_dict):
                        model_dict['ATOM_' + row[1]]['ANISOU'] = anisow_dict
        
            for row in list:
                if(row[0] == 'SIGUIJ'):
                    siguij_dict = {}
                    siguij_dict['u[0][0]'] = row[8]
                    siguij_dict['u[1][1]'] = row[9]
                    siguij_dict['u[2][2]'] = row[10]
                    siguij_dict['u[0][1]'] = row[11]
                    siguij_dict['u[0][2]'] = row[12]
                    siguij_dict['u[1][2]'] = row[13]  
                    if('ATOM_' + row[1] in model_dict):
                        model_dict['ATOM_' + row[1]]['SIGUIJ'] = siguij_dict          
            part_dict['model_1'] = model_dict               
        
        return part_dict 
    
    def list2dict_HETATM(self,list):
        ''' 
        list = [['MODEL','1'],
        ['ATOM','41', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','42','GLU','A','18'], 
        ['HETATM','43', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','44','GLU','A','18'],        
        ['HETATM','110', 'O', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','110', 'O', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','110', 'O', '', 'ASN', 'A', '14', '', '1', '1', '1', '1', '1', '1', 'N', ''],        
        ['ATOM','111', 'N', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','111', 'N', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','111', 'N', '', 'ASN', 'A', '14', '', '0', '0', '0', '0', '0', '0', 'N', ''],
        ['ENDMDL'],
        ['MODEL','2'],
        ['ATOM','41', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','42','GLU','A','18'],  
        ['HETATM','43', 'NH1', 'A', 'ARG', 'A', '-3', '', '12.218', '84.840', '88.007', '0.50', '40.76', 'N', ''],
        ['TER','44','GLU','A','18'],
        ['HETATM','110', 'O', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','110', 'O', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','110', 'O', '', 'ASN', 'A', '14', '', '11', '11', '11', '11', '11', '11', 'N', ''],        
        ['ATOM','111', 'N', ' ', 'ASN', 'A', '14', '', '14.421', '84.308', '88.373', '0.50', '40.45', 'N', ''],
        ['ANISOU','111', 'N', '', 'ASN', 'A', '14', '', '2059', '1674', '1462', '27', '244', '-96', 'N', ''],
        ['SIGUIJ','111', 'N', '', 'ASN', 'A', '14', '', '10', '10', '10', '10', '10', '10', 'N', ''],        
        ['ENDMDL']]
        
        part_dict = {
        'model_1': {'HETATM_43': 
                    {'Model_No': '1', 'Chain_No': '2', 'serial_No': '43', 'name': 'NH1', 'altLoc': 'A', 
                    'resName': 'ARG', 'chainID': 'A', 'resSeq': '-3', 'iCode': '', 'x': '12.218', 'y': '84.840', 'z': '88.007', 
                    'occupancy': '0.50', 'tempFactor': '40.76', 'element': 'N', 'charge': ''}, 
                    'HETATM_110':
                    {'Model_No': '1', 'Chain_No': '3', 'serial_No': '110', 'name': 'O', 'altLoc': ' ', 
                    'resName': 'ASN', 'chainID': 'A', 'resSeq': '14', 'iCode': '', 'x': '14.421', 'y': '84.308', 'z': '88.373', 
                    'occupancy': '0.50', 'tempFactor': '40.45', 'element': 'N', 'charge': '', 
                    'ANISOU': {'u[0][0]': '2059', 'u[1][1]': '1674', 'u[2][2]': '1462', 'u[0][1]': '27', 'u[0][2]': '244', 'u[1][2]': '-96'}, 
                    'SIGUIJ': {'u[0][0]': '1', 'u[1][1]': '1', 'u[2][2]': '1', 'u[0][1]': '1', 'u[0][2]': '1', 'u[1][2]': '1'}}}, 
        'model_2': {'HETATM_43': 
                   {'Model_No': '2', 'Chain_No': '2', 'serial_No': '43', 'name': 'NH1', 'altLoc': 'A', 
                   'resName': 'ARG', 'chainID': 'A', 'resSeq': '-3', 'iCode': '', 'x': '12.218', 'y': '84.840', 'z': '88.007', 
                   'occupancy': '0.50', 'tempFactor': '40.76', 'element': 'N', 'charge': ''}, 
                   'HETATM_110': 
                   {'Model_No': '2', 'Chain_No': '3', 'serial_No': '110', 'name': 'O', 'altLoc': ' ', 
                   'resName': 'ASN', 'chainID': 'A', 'resSeq': '14', 'iCode': '', 'x': '14.421', 'y': '84.308', 'z': '88.373', 
                   'occupancy': '0.50', 'tempFactor': '40.45', 'element': 'N', 'charge': '', 
                   'ANISOU': {'u[0][0]': '2059', 'u[1][1]': '1674', 'u[2][2]': '1462', 'u[0][1]': '27', 'u[0][2]': '244', 'u[1][2]': '-96'}, 
                   'SIGUIJ': {'u[0][0]': '11', 'u[1][1]': '11', 'u[2][2]': '11', 'u[0][1]': '11', 'u[0][2]': '11', 'u[1][2]': '11'}}}}

        '''    
        part_dict = {}
        
        flag = 0
        
        for i in range(0,len(list)):
            if(list[i][0] == 'MODEL'):
                model_dict = {}
                flag = 1
                Model_No = list[i][1]
                chain_count = 1
                for end in range(i + 1,len(list)):
                    if(list[end][0] == 'ENDMDL'):
                        break
                    if(list[end][0] == 'HETATM'):
                        hetatm_dict = {}
                        hetatm_dict['Model_No'] = Model_No
                        hetatm_dict['Chain_No'] = '1'
                        hetatm_dict['serial_No'] = list[end][1]
                        hetatm_dict['name'] = list[end][2]
                        hetatm_dict['altLoc'] = list[end][3]
                        hetatm_dict['resName'] = list[end][4]
                        hetatm_dict['chainID'] = list[end][5]
                        hetatm_dict['resSeq'] = list[end][6]
                        hetatm_dict['iCode'] = list[end][7]
                        hetatm_dict['x'] = list[end][8]
                        hetatm_dict['y'] = list[end][9]
                        hetatm_dict['z'] = list[end][10]
                        hetatm_dict['occupancy'] = list[end][11]
                        hetatm_dict['tempFactor'] = list[end][12]
                        hetatm_dict['element'] = list[end][13]
                        hetatm_dict['charge'] = list[end][14]
                        model_dict['HETATM_' + list[end][1]] = hetatm_dict
                        part_dict['model_' + Model_No] = model_dict
        
                    if(list[end][0] == 'ANISOU'):
                        anisow_dict = {}
                        anisow_dict['u[0][0]'] = list[end][8]
                        anisow_dict['u[1][1]'] = list[end][9]
                        anisow_dict['u[2][2]'] = list[end][10]
                        anisow_dict['u[0][1]'] = list[end][11]
                        anisow_dict['u[0][2]'] = list[end][12]
                        anisow_dict['u[1][2]'] = list[end][13]
                        if('HETATM_' + list[end][1] in model_dict):
                            model_dict['HETATM_' + list[end][1]]['ANISOU'] = anisow_dict
        
                    if(list[end][0] == 'SIGUIJ'):
                        siguij_dict = {}
                        siguij_dict['u[0][0]'] = list[end][8]
                        siguij_dict['u[1][1]'] = list[end][9]
                        siguij_dict['u[2][2]'] = list[end][10]
                        siguij_dict['u[0][1]'] = list[end][11]
                        siguij_dict['u[0][2]'] = list[end][12]
                        siguij_dict['u[1][2]'] = list[end][13]  
                        if('HETATM_' + list[end][1] in model_dict):
                            model_dict['HETATM_' + list[end][1]]['SIGUIJ'] = siguij_dict            
        
                    if(list[end][0] == 'HETATM' and 'HETATM_' + list[end][1] in model_dict):
                        model_dict['HETATM_' + list[end][1]]['Chain_No'] = str(chain_count)
                    if(list[end][0] == 'TER'):
                        chain_count += 1
                    end += 1
                    
                    
        if(flag == 0):
            model_dict = {}
            for row in list:
                if(row[0] == 'HETATM'):
                    hetatm_dict = {}
                    hetatm_dict['Model_No'] = '1'
                    hetatm_dict['Chain_No'] = '1'
                    hetatm_dict['serial_No'] = row[1]
                    hetatm_dict['name'] = row[2]
                    hetatm_dict['altLoc'] = row[3]
                    hetatm_dict['resName'] = row[4]
                    hetatm_dict['chainID'] = row[5]
                    hetatm_dict['resSeq'] = row[6]
                    hetatm_dict['iCode'] = row[7]
                    hetatm_dict['x'] = row[8]
                    hetatm_dict['y'] = row[9]
                    hetatm_dict['z'] = row[10]
                    hetatm_dict['occupancy'] = row[11]
                    hetatm_dict['tempFactor'] = row[12]
                    hetatm_dict['element'] = row[13]
                    hetatm_dict['charge'] = row[14]
                    model_dict['HETATM_' + row[1]] = hetatm_dict
            
            for row in list:
                if(row[0] == 'ANISOU'):
                    anisow_dict = {}
                    anisow_dict['u[0][0]'] = row[8]
                    anisow_dict['u[1][1]'] = row[9]
                    anisow_dict['u[2][2]'] = row[10]
                    anisow_dict['u[0][1]'] = row[11]
                    anisow_dict['u[0][2]'] = row[12]
                    anisow_dict['u[1][2]'] = row[13]
                    if('HETATM_' + row[1] in model_dict):
                        model_dict['HETATM_' + row[1]]['ANISOU'] = anisow_dict
                    
            for row in list:
                if(row[0] == 'SIGUIJ'):
                    siguij_dict = {}
                    siguij_dict['u[0][0]'] = row[8]
                    siguij_dict['u[1][1]'] = row[9]
                    siguij_dict['u[2][2]'] = row[10]
                    siguij_dict['u[0][1]'] = row[11]
                    siguij_dict['u[0][2]'] = row[12]
                    siguij_dict['u[1][2]'] = row[13]  
                    if('HETATM_' + row[1] in model_dict):
                        model_dict['HETATM_' + row[1]]['SIGUIJ'] = siguij_dict          
            part_dict['model_1'] = model_dict        
        
        return part_dict
    
    def list2dict_CONECT(self,list):
        ''' 
        list = [['CONECT','1179', '746', '1184', '1195', '1203'],
        ['CONECT','1179', '1211', '1222', '', '']]
        part_dict = {'CONECT_1': {'Atom_serialNum': '1179',
                                  'bonded_atom_1': '746',
                                  'bonded_atom_2': '1184',
                                  'bonded_atom_3': '1195',
                                  'bonded_atom_4': '1203'},
                    'CONECT_2': {'Atom_serialNum': '1179',
                                 'bonded_atom_1': '1211',
                                 'bonded_atom_2': '1222',
                                 'bonded_atom_3': '',
                                 'bonded_atom_4': ''}}
        '''         
        part_dict = {}
        
        conect_count = 1
        for row in list:
            if(row[0] == 'CONECT'):
                atom_dict = {}
                atom_dict['Atom_serialNum'] = row[1]
                atom_dict['bonded_atom_1'] = row[2]
                atom_dict['bonded_atom_2'] = row[3]
                atom_dict['bonded_atom_3'] = row[4]
                atom_dict['bonded_atom_4'] = row[5]
                part_dict['CONECT_' + str(conect_count)] = atom_dict
                conect_count += 1 
                
        return part_dict  
    
    def list2dict_MASTER(self,list):
        ''' 
        list = [['MASTER','40', '0', '0', '0', '0', '0', '0', '6', '2930', '2', '0', '29']]
        '''         
        part_dict = {}
        for row in list:
            if(row[0] == 'MASTER'):
                part_dict['numRemark'] = row[1]
                part_dict['"0"'] = row[2]
                part_dict['numHet'] = row[3]
                part_dict['numHelix'] = row[4]
                part_dict['numSheet'] = row[5]
                part_dict['numTurn'] = row[6]
                part_dict['numSite'] = row[7]
                part_dict['numXform'] = row[8]
                part_dict['numCoord'] = row[9]
                part_dict['numTer'] = row[10]
                part_dict['numConect'] = row[11]
                part_dict['numSeq'] = row[12]
        
        return part_dict   
    
    def list2dict_HEADER(self,list):
        part_dict = {}
        for row in list:
            if row[0] == 'HEADER':
                length = len(row)
                if length != 4:
                    part_dict['HEADER_classification'] = []
                    for j in range(1,length - 2):
                        part_dict['HEADER_classification'].append(row[j])
                else:
                    part_dict['HEADER_classification'] = row[1]
                part_dict['HEADER_depDate'] = row[-2]
                part_dict['HEADER_idCode'] = row[-1]  
        return part_dict  
    
    def list2dict_COMPND(self,list):
        part_dict = {}
        flag = 0
        for row in list:
            if row[0] == 'COMPND':
                if row[2] != 'continuation':
                    if row[2] == 'MOL_ID':
                        key = 'MOL_ID_' + str(row[3])
                        part_dict[key] = {}
                    else:
                        key2 = row[2]
                        if row[-1] == ',':
                            flag = 1
                            part_dict[key][key2] = row[3:-1]
                        else:
                            flag = 0
                            part_dict[key][key2] = row[3:]
                else:
                    if type(part_dict[key][key2]) is str:
                        a = part_dict[key][key2]
                        part_dict[key][key2] = []
                        part_dict[key][key2].append(a)
                    
                    if flag:
                        for i in row[3:]:
                            part_dict[key][key2].append(i)
                    else:
                        part_dict[key][key2][-1] += row[3]
                        if len(row) != 4:
                            for i in row[4:]:
                                part_dict[key][key2].append(i)
        return part_dict
    
    def list2dict_SOURCE(self,list):
        part_dict = {}
        flag = 0
        for row in list:
            if row[0] == 'SOURCE':
                if row[2] != 'continuation':
                    if row[2] == 'MOL_ID':
                        key = 'MOL_ID_' + str(row[3])
                        part_dict[key] = {}
                    else:
                        key2 = row[2]
                        if row[-1] == ',':
                            flag = 1
                            part_dict[key][key2] = row[3:-1]
                        else:
                            flag = 0
                            part_dict[key][key2] = row[3:]
                else:
                    if type(part_dict[key][key2]) is str:
                        a = part_dict[key][key2]
                        part_dict[key][key2] = []
                        part_dict[key][key2].append(a)
                    
                    if flag:
                        for i in row[3:]:
                            part_dict[key][key2].append(i)
                    else:
                        part_dict[key][key2][-1] += row[3]
                        if len(row) != 4:
                            for i in row[4:]:
                                part_dict[key][key2].append(i)
        return part_dict
    
    def list2dict_KEYWDS(self,list):
        part_dict = {}
        count = 0
        flag = 1
        for row in list:
            if row[0] == 'KEYWDS':
                if row[1] != '' and flag == 0:
                    part_dict[key] += ' ' + row[2]
                    index = 3
                else:
                    index = 2
                for i in row[index:-1]:
                    count += 1
                    key = 'KEYWDS_' + str(count)
                    part_dict[key] = i
                if row[-1] == ',':
                    flag = 1
                else:
                    flag = 0
        return part_dict
    
    def list2dict_EXPDTA(self,list):
        part_dict = {}
        count = 0
        flag = 1
        for row in list:
            if row[0] == 'EXPDTA':
                if row[1] != '' and flag == 0:
                    part_dict[key] += ' ' + row[2]
                    index = 3
                else:
                    index = 2
                for i in row[index:-1]:
                    count += 1
                    key = 'technique_' + str(count)
                    part_dict[key] = i
                if row[-1] == ',':
                    flag = 1
                else:
                    flag = 0
        return part_dict
        
    def list2dict_MDLTYP(self,list):
        part_dict = {}
        count = 0
        flag = 1
        for row in list:
            if row[0] == 'MDLTYP':
                if row[1] != '1' and flag == 0:
                    part_dict[key] += ' ' + row[2]
                    index = 3
                else:
                    index = 2
                for i in row[index:-1]:
                    count += 1
                    key = 'comment_' + str(count)
                    part_dict[key] = i
                if row[-1] == ',':
                    flag = 1
                else:
                    flag = 0
        return part_dict
        
    def list2dict_AUTHOR(self,list):
        '''
        list = [['AUTHOR','1', 'M.B.BERRY', 'P.LIANG', 'M.GLASER,'],                   
        ['AUTHOR','2', ' G.N.PHILLIPS JR.', 'T.L.ST. STEVENS,'],          
        ['AUTHOR','3', 'M.B.BERRY', 'B.MEADOR', 'G.N.PHILLIPS'],
        ['AUTHOR','4', 'JR.', 'T.L.ST. STEVENS']] 
        
        part_dict = {'AUTHOR_1': 'M.B.BERRY', 'AUTHOR_2': 'P.LIANG', 'AUTHOR_3': 'M.GLASER', 
        'AUTHOR_4': 'G.N.PHILLIPS JR.', 'AUTHOR_5': 'T.L.ST. STEVENS', 
        'AUTHOR_6': 'M.B.BERRY', 'AUTHOR_7': 'B.MEADOR', 'AUTHOR_8': 'G.N.PHILLIPS JR.', 
        'AUTHOR_9': 'T.L.ST. STEVENS'}
    
        '''        
        part_dict = {}
       
        auther_count = 0
        continuation = 1
        for row in list:
            if(row[0] == 'AUTHOR'):
                if(len(row) == 2):
                    part_dict['AUTHOR_1'] = 'None'
                    break
                if(row[1] == '1'):
                    continuation += 1
                    for i in range(2,len(row)):
                        auther_count += 1
                        part_dict['AUTHOR_' + str(auther_count)] = row[i]            
                elif(row[1] == str(continuation)):    
                    continuation += 1
                    if(part_dict['AUTHOR_' + str(auther_count)][-1] == ','):
                        part_dict['AUTHOR_' + str(auther_count)] = part_dict['AUTHOR_' + str(auther_count)][:-1]
                        auther_count += 1
                        part_dict['AUTHOR_' + str(auther_count)] = row[2].strip() 
                    else:
                        if(row[2][0] != ' '):
                            part_dict['AUTHOR_' + str(auther_count)] += ' ' + row[2]
                        else:
                            part_dict['AUTHOR_' + str(auther_count)] += row[2]                    
                    for i in range(3,len(row)):
                        auther_count += 1
                        part_dict['AUTHOR_' + str(auther_count)] = row[i]               
       
        return part_dict    
    
    def list2dict_JRNL(self,list):
        '''
        list = [['JRNL',' ','AUTH','1','P.A.SMITH','M.F.T.KOEHLER','H.S.GIRGIS','D.YAN','Y.CHEN,'],      
        ['JRNL',' ','AUTH','2','J.J.CRAWFORD','M.R.DURK','R.I.HIGUCHI','J.KANG','J.MURRAY,'],           
        ['JRNL',' ','AUTH','3','P.PARASELLI','S.PARK','W.PHUNG','J.G.QUINN','T.C.ROBERTS','L.ROUGE'],    
        ['JRNL',' ','AUTH','4','J.B.SCHWARZ','E.SKIPPINGTON','J.WAI','M.XU','Z.YU','H.ZHANG','M.W.TAN'],   
        ['JRNL',' ','AUTH','5',' C.E.HEISE'],                                                    
        ['JRNL',' ','TITL','1' ,'OPTIMIZED ARYLOMYCINS ARE A NEW CLASS OF GRAM-NEGATIVE'],       
        ['JRNL',' ','TITL','2','ANTIBIOTICS.'],                                                 
        ['JRNL','REF', '0', '', '', '', ''],                                              
        ['JRNL','REF', '1', 'NATURE', '561', '189', '2018'],
        ['JRNL','REF', '2', 'ACTA CRYSTALLOGR.,SECT.D', '297', '1692', '2002'],
        ['JRNL','REFN', 'ESSN', '1476-46'],
        ['JRNL','PMID', '30209367'],
        ['JRNL','DOI', '10.1038/S41586-018-0483-6'],
        ['JRNL','PUBL', '1', 'jsowmd dsio,'],
        ['JRNL','PUBL', '2', 'dnwdhwldks'],
        ['JRNL','PUBL', '3', 'fwd']
        ]
        
        part_dict = {'AUTH': {'LString(1)': ' ', 'AUTHOR_1': 'P.A.SMITH', 'AUTHOR_2': 'M.F.T.KOEHLER', 'AUTHOR_3': 'H.S.GIRGIS', 
                                    'AUTHOR_4': 'D.YAN', 'AUTHOR_5': 'Y.CHEN', 'AUTHOR_6': 'J.J.CRAWFORD', 'AUTHOR_7': 'M.R.DURK', 
                                    'AUTHOR_8': 'R.I.HIGUCHI', 'AUTHOR_9': 'J.KANG', 'AUTHOR_10': 'J.MURRAY', 'AUTHOR_11': 'P.PARASELLI', 
                                    'AUTHOR_12': 'S.PARK', 'AUTHOR_13': 'W.PHUNG', 'AUTHOR_14': 'J.G.QUINN', 'AUTHOR_15': 'T.C.ROBERTS', 
                                    'AUTHOR_16': 'L.ROUGE J.B.SCHWARZ', 'AUTHOR_17': 'E.SKIPPINGTON', 'AUTHOR_18': 'J.WAI', 
                                    'AUTHOR_19': 'M.XU', 'AUTHOR_20': 'Z.YU', 'AUTHOR_21': 'H.ZHANG', 'AUTHOR_22': 'M.W.TAN  C.E.HEISE'}, 
        'TITL': {'LString(1)': ' ', 'TITL': 'OPTIMIZED ARYLOMYCINS ARE A NEW CLASS OF GRAM-NEGATIVE ANTIBIOTICS.'}, 
        'REF': {'pubName_0': '', 'volume _0': '', 'page_0': '', 'year_0': '', 
                'pubName_1': 'NATURE', 'volume _1': '561', 'page_1': '189', 'year_1': '2018', 
                'pubName_2': 'ACTA CRYSTALLOGR.,SECT.D', 'volume _2': '297', 'page_2': '1692', 'year_2': '2002'}, 
        'REFN': {'"ISSN" or "ESSN"': 'ESSN', 'issn': '1476-46'}, 
        'PMID': '30209367', 
        'DOI': '10.1038/S41586-018-0483-6', 
        'PUBL': 'jsowmd dsio,dnwdhwldks fwd'}}

        '''
        
        part_dict = {}
        
        AUTH_list = {}
        TITL_list = {}
        title_flag = 0
        REF_list = {}
        REFN_list = {}
        auther_count = 0
        continuation = 1
        
        for row in list:
            if(row[0] == 'JRNL'):
                if(len(row) > 3):
                    if(row[2] == 'AUTH'):
                        AUTH_list['LString(1)'] = row[1]
                        if(len(row) == 4):
                            AUTH_list['AUTH_1'] = 'None'
                            break            
                        if(row[3] == '1'):
                            continuation = int(row[3]) + 1
                            for i in range(4,len(row)):
                                auther_count += 1
                                AUTH_list['AUTHOR_' + str(auther_count)] = row[i]            
                        elif(row[3] == str(continuation)):    
                            continuation = int(row[3]) + 1
                            if(AUTH_list['AUTHOR_' + str(auther_count)][-1] == ','):
                                AUTH_list['AUTHOR_' + str(auther_count)] = AUTH_list['AUTHOR_' + str(auther_count)][:-1]
                                auther_count = auther_count + 1
                                AUTH_list['AUTHOR_' + str(auther_count)] = row[4].strip() 
                            else:
                                if(row[2][0] != ' '):
                                    AUTH_list['AUTHOR_' + str(auther_count)] += ' ' + row[4]
                                else:
                                    AUTH_list['AUTHOR_' + str(auther_count)] += row[4]
                            for i in range(5,len(row)):
                                auther_count += 1
                                AUTH_list['AUTHOR_' + str(auther_count)] = row[i]    
                        part_dict['AUTH'] = AUTH_list 
                    
                    if(row[2] == 'EDIT'):
                        AUTH_list['LString(1)'] = row[1]
                        if(len(row) == 4):
                            AUTH_list['AUTH_1'] = 'None'
                            break            
                        if(row[3] == '1'):
                            continuation = int(row[3]) + 1
                            for i in range(4,len(row)):
                                auther_count += 1
                                AUTH_list['EDIT_' + str(auther_count)] = row[i]            
                        elif(row[3] == str(continuation)):    
                            continuation = int(row[3]) + 1
                            if(AUTH_list['EDIT_' + str(auther_count)][-1] == ','):
                                AUTH_list['EDIT_' + str(auther_count)] = AUTH_list['EDIT_' + str(auther_count)][:-1]
                                auther_count = auther_count + 1
                                AUTH_list['EDIT_' + str(auther_count)] = row[4].strip() 
                            else:
                                if(row[2][0] != ' '):
                                    AUTH_list['EDIT_' + str(auther_count)] += ' ' + row[4]
                                else:
                                    AUTH_list['EDIT_' + str(auther_count)] += row[4]
                            for i in range(5,len(row)):
                                auther_count += 1
                                AUTH_list['EDIT_' + str(auther_count)] = row[i]    
                        part_dict['EDIT'] = AUTH_list
                        
                    elif(row[2] == 'TITL'):
                        TITL_list['LString(1)'] = row[1]              
                        if(row[3] == '1'):
                            continuation = int(row[3]) + 1
                            title_flag = 1
                            TITL_list['TITL'] = row[4]
                        elif(row[3] != '1' and title_flag == 0):
                            continuation = int(row[3]) + 1
                            title_flag = 1
                            TITL_list['TITL'] = row[4]                            
                        elif(row[3] == str(continuation)):
                            continuation = int(row[3]) + 1
                            if(TITL_list['TITL'][-1] == ','):
                                TITL_list['TITL'] += row[4]
                            else:
                                TITL_list['TITL'] += ' ' + row[4]
                        part_dict['TITL'] = TITL_list 
                        
                if(len(row) > 2):
                    if(row[1] == 'REF'):
                        REF_list['pubName_' + row[2]] = row[3]
                        REF_list['volume _' + row[2]] = row[4]
                        REF_list['page_' + row[2]] = row[5]
                        REF_list['year_' + row[2]] = row[6]
                        part_dict['REF'] = REF_list
                             
                    elif(row[1] == 'REFN'):
                        REFN_list['"ISSN" or "ESSN"'] = row[2]
                        REFN_list['issn'] = row[3]
                        part_dict['REFN'] = REFN_list
                        
                    elif(row[1] == 'PMID'):
                        part_dict['PMID'] = row[2]
                        
                    elif(row[1] == 'DOI'):
                        part_dict['DOI'] = row[2]
                        
                    elif(row[1] == 'PUBL'):
                        if(row[2] == '1'):
                            continuation = int(row[2]) + 1
                            part_dict['PUBL'] = row[3]
                        elif(row[2] == str(continuation)):
                            continuation = int(row[2]) + 1
                            if(part_dict['PUBL'][-1] == ','):
                                part_dict['PUBL'] += row[3]
                            else:
                                part_dict['PUBL'] += ' ' + row[3]                     
                    
        return part_dict  
    
    def list2dict_HETNAM(self,list):
        '''
        list = [['HETNAM','1', 'NAG', '"N-ACETYL-D-GLUCOSAMINE"'],
        ['HETNAM','1', 'SAD', '"BETA-METHYLENE SELENAZOLE-4-CARBOXAMIDE ADENINE"'],
        ['HETNAM','2', 'SAD', '"DINUCLEOTIDE"'],
        ['HETNAM','1', 'UDP', '"URIDINE-5\'-DIPHOSPHATE"'],
        ['HETNAM','1', 'UNX', '"UNKNOWN ATOM OR ION"'],
        ['HETNAM','1', 'UNL', '"UNKNOWN LIGAND"'],
        ['HETNAM','1', 'B3P', '"2-[3-(2-HYDROXY-1,1-DIHYDROXYMETHYL-ETHYLAMINO)-"'],
        ['HETNAM','2', 'B3P', '" PROPYLAMINO]-2-HYDROXYMETHYL-PROPANE-1,3-DIOL"']]
        
        part_dict = {'NAG': 'N-ACETYL-D-GLUCOSAMINE', 
        'SAD': 'BETA-METHYLENE SELENAZOLE-4-CARBOXAMIDE ADENINE DINUCLEOTIDE', 
        'UDP': "URIDINE-5'-DIPHOSPHATE", 
        'UNX': 'UNKNOWN ATOM OR ION', 
        'UNL': 'UNKNOWN LIGAND', 
        'B3P': '2-[3-(2-HYDROXY-1,1-DIHYDROXYMETHYL-ETHYLAMINO)- PROPYLAMINO]-2-HYDROXYMETHYL-PROPANE-1,3-DIOL'}
        '''

        part_dict = {}
        
        for row in list:
            if(row[0] == 'HETNAM'):
                if(row[1] == '1'):
                    part_dict[row[2]] = row[3].strip('"')
                else:
                    if(row[3][1] != ' '):
                        part_dict[row[2]] += ' ' + row[3].strip('"')
                    else:
                        part_dict[row[2]] += row[3].strip('"')        
        
        return part_dict  
    
    def list2dict_HETSYN(self,list):
        '''
        list = [['HETSYN','1', 'HV5', '"3-METHYL-L-VALINE"'],
        ['HETSYN','1', 'AB1', '"ABT-378; LOPINAVIR"'],
        ['HETSYN','1', 'CMP', '"CYCLIC AMP; CAMP"'],
        ['HETSYN','1', 'TRS', '"TRIS  BUFFER;"']]
        
        part_dict = {'HV5': '3-METHYL-L-VALINE', 
        'AB1': 'ABT-378; LOPINAVIR', 
        'CMP': 'CYCLIC AMP; CAMP', 
        'TRS': 'TRIS  BUFFER;'}
        '''
        
        part_dict = {}
        
        for row in list:
            if(row[0] == 'HETSYN'):
                if(row[1] == '1'):
                    part_dict[row[2]] = row[3].strip('"')
                else:
                    if(row[3][1] != ' '):
                        part_dict[row[2]] += ' ' + row[3].strip('"')
                    else:
                        part_dict[row[2]] += row[3].strip('"')              
        
        return part_dict    
    
    def list2dict_FORMUL(self,list):
        '''
        list = [['FORMUL','3', 'MG', '1',' ', '2(MG 2+)'],
        ['FORMUL','5', 'SO4', '2',' ', '2-'],
        ['FORMUL','5', 'SO4', '3',' ', ')'],
        ['FORMUL','13', 'HOH','1', '*', '360(H2 O)'],
        ['FORMUL','3', 'NAP', '1',' ', '2(C21 H28 N7 O17 P3)'],
        ['FORMUL','4', 'FOL', '1',' ', '2(C19 H19 N7 O6)'],
        ['FORMUL','5', '1PE', '1',' ', 'C10 H22 O6'],
        ['FORMUL','2', 'NX5', '1',' ', 'C14 H10 O2 CL2 S']]  
        
        part_dict = {'FORMUL_1': {'Chemical_formula': '2(MG 2+) 2-)',
                                 'compNum': '3',
                                 'hetID': 'MG',
                                 'isWater': 'No'},
                    'FORMUL_2': {'Chemical_formula': '360(H2 O)',
                                 'compNum': '13',
                                 'hetID': 'HOH',
                                 'isWater': 'Yes'},
                    'FORMUL_3': {'Chemical_formula': '2(C21 H28 N7 O17 P3)',
                                 'compNum': '3',
                                 'hetID': 'NAP',
                                 'isWater': 'No'},
                    'FORMUL_4': {'Chemical_formula': '2(C19 H19 N7 O6)',
                                 'compNum': '4',
                                 'hetID': 'FOL',
                                 'isWater': 'No'},
                    'FORMUL_5': {'Chemical_formula': 'C10 H22 O6',
                                 'compNum': '5',
                                 'hetID': '1PE',
                                 'isWater': 'No'},
                    'FORMUL_6': {'Chemical_formula': 'C14 H10 O2 CL2 S',
                                 'compNum': '2',
                                 'hetID': 'NX5',
                                 'isWater': 'No'}}

        '''
        part_dict = {}
        formul_dict = {}
        formul_count = 1
        for row in list:
            if(row[0] == 'FORMUL'):
                if(row[3] == '1'):
                    formul_dict['compNum_' + str(formul_count)] = row[1]
                    formul_dict['hetID_' + str(formul_count)] = row[2]
                    if(row[4] == '*'):
                        formul_dict['isWater_' + str(formul_count)] = 'Yes'
                    else:
                        formul_dict['isWater_' + str(formul_count)] = 'No'
                    formul_dict['Chemical_formula_' + str(formul_count)] = row[5]
                    formul_count += 1
                else:
                    if(row[5][0] != ' ' and row[5][0] != '(' and row[5][0] != ')'):
                        formul_dict['Chemical_formula_' + str(formul_count-1)] += ' ' + row[5]
                    else:
                        formul_dict['Chemical_formula_' + str(formul_count-1)] += row[5]   
                    
        for i in range(1,formul_count):
            small_dict = {}
            small_dict['compNum'] = formul_dict['compNum_' + str(i)]
            small_dict['hetID'] = formul_dict['hetID_' + str(i)]
            small_dict['isWater'] = formul_dict['isWater_' + str(i)]
            small_dict['Chemical_formula'] = formul_dict['Chemical_formula_' + str(i)]
            part_dict['FORMUL_' + str(i)] = small_dict               
        
        return part_dict         
    
    def list2dict_CAVEAT(self,list):
        '''
        list = [['CAVEAT','1', '2UXK', 'xxxxxpabvdklabj'],
        ['CAVEAT','2', '2UXK', 'bbbbbldkbewpab,'],
        ['CAVEAT','3', '2UXK', 'aaaaadkbewpab'],
        ['CAVEAT','4', '2UXK', ' ppppdkbewpab'],
        ['CAVEAT','1', 'PWJS', 'eeeeebewpab']]
        
        part_dict = {'CAVEAT_1': {'idCode': '2UXK', 'CAVEAT_reason': 'xxxxxpabvdklabj bbbbbldkbewpab, aaaaadkbewpab ppppdkbewpab'}, 
                     'CAVEAT_2': {'idCode': 'PWJS', 'CAVEAT_reason': 'eeeeebewpab'}}

        '''
        part_dict = {}
        
        caveat_dict = {}
        caveat_count = 1
        caveat_list = []
        for row in list:
            if(row[0] == 'CAVEAT'):
                if(row[1] == '1'):
                    caveat_list.append(row[2])
                    caveat_dict['idCode_' + str(caveat_count)] = row[2]
                    caveat_dict['CAVEAT_reason_' + str(caveat_count)] = row[3]
                    caveat_count += 1
                elif(row[1] != '1' and row[2] not in caveat_list):
                    caveat_list.append(row[2])
                    caveat_dict['idCode_' + str(caveat_count)] = row[2]
                    caveat_dict['CAVEAT_reason_' + str(caveat_count)] = row[3]
                    caveat_count += 1                  
                elif(row[2] in caveat_list):                          
                    if(row[3][0] != ' '):
                        caveat_dict['CAVEAT_reason_' + str(caveat_count-1)] += ' ' + row[3]
                    else:
                        caveat_dict['CAVEAT_reason_' + str(caveat_count-1)] += row[3]    
        for i in range(1,caveat_count):
            small_dict = {}
            small_dict['idCode'] = caveat_dict['idCode_' + str(i)]
            small_dict['CAVEAT_reason'] = caveat_dict['CAVEAT_reason_' + str(i)]
            part_dict['CAVEAT_' + str(i)] = small_dict        
                        
        return part_dict  
    
    def list2dict_REMARK(self,list):
        part_dict = {}
        remark_list = []
        for row in list:
            if(row[0] == 'REMARK'):
                if(row[1] not in remark_list):
                    part_list = []
                    remark_list.append(row[1])
                    part_list.append(row[2])
                    part_dict[row[1]] = part_list
                else:
                    part_dict[row[1]].append(row[2])
                
        return part_dict  
      
#if __name__ == "__main__":
