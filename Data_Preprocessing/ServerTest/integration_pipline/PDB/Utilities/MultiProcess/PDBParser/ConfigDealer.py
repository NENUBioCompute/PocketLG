import xml.etree.ElementTree as ET
from xml.dom import minidom
import datetime,os

class ConfigDealer:
    __cfg = {}
    #__tree = ET()
    
 #----------------------------------------------------------------------
    def __init__(self,fileName = 'config.xml'):
        """Constructor"""
        
        #Open the configuration file and try to parse it
        try:
            with open(fileName, 'r') as fh:
                try:
                    self.__tree = ET.parse(fh)
                    root = self.__tree.getroot()
                    self.__cfg = self.XML2Dictionary(root)
                    self.__cfg = self.__cfg[root.tag]
                except:
                    print('Error: the format of config file is illegal')
        except IOError:
            print('Error: Could not read from config file')
      
    def   XML2Dictionary(self, parentElement): 
        childrenNodes = parentElement.getchildren()
        if len(childrenNodes) != 0:
            #parentElement
            #self.__cfg.update(dict(parentElement.attrib['name'],{}))
            childDict = {}
            for node in childrenNodes:
                childDict.update(self.XML2Dictionary(node))

            if parentElement.attrib:
                if 'name' in parentElement.attrib.keys():
                    #print({parentElement.attrib['name']:childDict})
                    return {parentElement.attrib['name']:childDict}

                elif 'type' in parentElement.attrib.keys() and parentElement.attrib['type'] == 'list':
                    #handle the list type data
                    #return the list of childDict' values
                    return {parentElement.tag:list(childDict.values())}

                else:
                    #print(parentElement.attrib)
                    raise Exception('Error: the format of config file is illegal: parent elements should be with name value')
            else: 
                return  {parentElement.tag:childDict}

        else:
            #parentElement is a leaf node
            if parentElement.text is not None: 
                #print({parentElement.tag:parentElement.text})
                return {parentElement.tag:parentElement.text}
            else:
                #raise Exception('Error: the format of config file is illegal: elements with a none value')
                return {parentElement.tag:''}

      
    def getConfigDictionary(self):
        return self.__cfg
      
           
    #----------------------------------------------------------------------
    def get(self, targetDB, operator, option):
        """"""
        try:
            return self.__cfg[targetDB][operator][option]
             
        except:
            raise Exception('Error: cannot extract the config info correctly, please use correct element name')
        
    def TaskDependencyCheck(self):
        tasks = []
        interval = {'daily':1,'weekly':7,'monthly':31}
        for dbElements in self.__tree.getroot().getchildren():
            dbName = dbElements.attrib['name']
            #intervalDays = self.__cfg[dbName]['Download']['interval']
            lastDownload = datetime.datetime.strptime(self.__cfg[dbName]['Download']['lastDownload'], '%d/%m/%Y')
            nowDate = datetime.datetime.today()
            delta = nowDate - lastDownload
            try:
                intervalDay = interval[self.__cfg[dbName]['Download']['interval']]
            except:
                intervalDay = 1
            
            if delta.days >= intervalDay:
                tasks.append((dbName, 'Download', nowDate.strftime('%Y-%m-%d %H:%M:%S')))
        return tasks
    
    def LoadAndStartTask(self, targetDB, operator):
        moduleName,ext = os.path.splitext(os.path.basename(self.__cfg[targetDB][operator]['modulePath']))
        module = __import__(moduleName)                  
        # create an instance of downloader and invoke 'parser'
        objectInstance = getattr(module,self.__cfg[targetDB][operator]['className'])(self.__cfg,targetDB)
        return getattr(objectInstance,self.__cfg[targetDB][operator]['mainFunction'])()        

    #def set(self, targetDB, operator, option, value):
        
        #self.__tree.

    @staticmethod
    def CfgDict2XML(paraDict, cfgFileName):
        doc = minidom.Document()
      
        #Create the top-level node
        rootNode = doc.createElement("config")
        doc.appendChild(rootNode)
                
        #Create the targetDB level nodes
        for DBName in list(paraDict.keys()):
            DBNode = doc.createElement("targetDB")
            DBNode.setAttribute("name", DBName)
            rootNode.appendChild(DBNode)
            for operatorName in list(dict(paraDict[DBName]).keys()):
                #Create the operator level nodes
                operatorNode = doc.createElement("operator")
                operatorNode.setAttribute("name", operatorName)
                DBNode.appendChild(operatorNode)
                for infoName in list(dict(paraDict[DBName][operatorName]).keys()):
                    #Create the inforation level nodes
                    #get the information value's type
                    if isinstance(paraDict[DBName][operatorName][infoName], list):
                        if len(paraDict[DBName][operatorName][infoName]) > 0:
                            # the information value's type is list
                            listNode = doc.createElement(infoName)
                            listNode.setAttribute("type", "list")
                            operatorNode.appendChild(listNode)
                            #add the value of list to listNode as its children
                            i = 1
                            for listValue in list(paraDict[DBName][operatorName][infoName]):
                                listValueNode = doc.createElement(str(infoName) + str(i))
                                listNode.appendChild(listValueNode)
                                listValueNode.appendChild(doc.createTextNode(str(listValue)))
                                i = i + 1   
                        else:
                            #if the list has no element
                            pass
                            
                    else:
                        # the information value's type is string
                        infoNode = doc.createElement(str(infoName))
                        operatorNode.appendChild(infoNode)
                        infoNode.appendChild(doc.createTextNode(str(paraDict[DBName][operatorName][infoName])))     
           
        try: 
            with open(cfgFileName, "w+") as fileHandle:
                doc.writexml(fileHandle, addindent='  ', newl='\n',encoding='utf-8')
        except IOError: 
            print ("open config file failed" ) 
        
    
    #----------------------------------------------------------------------
    ''' 
     def  excutebyConfig(self, args):
       
        #The operations in every section shall be executed in sequence
        if self.argsValidate(args) != True:
            raise Exception("Please check the command parameters and config file")
          
        #1st : Download     
        #Load the Downloader Model        
       
        moduleName,ext = os.path.splitext(os.path.basename(self.cfg.get(args.database, 'downloader path')))
        module = __import__(moduleName)                   
        # create an instance of downloader and invoke 'downloader'
        objectInstance = getattr(module, self.cfg.get(args.database, 'downloader'))(self.cfg.getint(args.database, 'thread'),self.cfg.getint(args.database, 'interval'))
        getattr(objectInstance,self.cfg.get(args.database, 'download function') )()
       ''' 
        
        
        
       
         
if __name__=="__main__": 
    cfgDealer = ConfigDealer()
    print("before:\n\r")
    cfgDict = cfgDealer.getConfigDictionary()
    print(cfgDict)
    cfgDict["PDBparser"]["Download"]["obsoletelist"] = ['1','1']
    print("after:\n\r")
    print(cfgDict)
    #obsoletelist
    #cfgDict["PDBparser"]["Download"]["mode"] = "http"
    ConfigDealer.CfgDict2XML(cfgDict, 'config.xml')
   # print("please check the PDBparser-Download-mode in the XML file")