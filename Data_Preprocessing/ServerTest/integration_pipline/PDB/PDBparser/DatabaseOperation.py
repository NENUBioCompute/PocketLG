
import pymongo
from pymongo import MongoClient
import pymongo.errors as er
from gridfs import GridFS
import json
from bson import json_util
from bson import ObjectId

class DataStorage:

    @staticmethod    
    def Storage(name, dic, address = "39.97.240.2", port = 27017):
        client = pymongo.MongoClient(address,port)
        db=client['admin']
        db.authenticate("root","@nenu_icb_2019_2022@")
        collection = client['Biodata'][name]
        return collection.insert(dic)
    
    """
    To storage a single dic
    param name: The collection(database) name
    param dic: The dic you wanna insert
    using example: DataStorage.Storage("Test",{"Name":"Han Wang"})
    """

    # def find(self,id, address = "39.97.240.2", port = 27017):
    '''
    To find a dic in gridfs
    :param id: the id of the file
    :return str: the json format file
    '''
    #     client = pymongo.MongoClient(address, port)
    #     db = client['admin']
    #     db.authenticate("root", "@nenu_icb_2019_2022@")
    #     collection = client['Biodata']['LargePDB']
    #     # fs = GridFS(client['Biodata'], 'LargePDB')
    #     # gf = fs.get(ObjectId(id))
    #     # str = gf.read()
    #     # retun str


    def StorageIter(self,name, iter, address = "39.97.240.2", port = 27017):
        client = pymongo.MongoClient(address,port)
        db=client['admin']
        db.authenticate("root","@nenu_icb_2019_2022@")
        collection_PDB = client['Biodata'][name]
        collection_LargePDB = client['Biodata']['LargePDB']
        fs = GridFS(client['Biodata'], 'LargePDB')
        for i in iter:
            try:
                collection_PDB.insert(i)
                print(i["File"], ' success\n')
            except er.DocumentTooLarge:
                i.pop("_id")
                print(i["File"], "is too large, use gridfs to store")
                data = json_util.dumps(i).encode("UTF-8")
                ObjectId = fs.put(data, filename=i["File"])


    """
    To storage dics from a iter
    param name: The collection(database) name
    param iter: The iter to insert
    using example: DataStorage.StorageIter("database",parser_func("Database.txt"))
    """


class DataQuery:

    @staticmethod
    def QueryOne(name, key, value, address = "39.97.240.2", port = 27017):
        client = pymongo.MongoClient(address,port)
        db=client['admin']
        db.authenticate("root","@nenu_icb_2019_2022@")
        collection = client['Biodata'][name]
        dic={}
        dic[key] = value
        return collection.find_one(dic)

    """
    To find a single dic
    param name: The collection(database) name
    param key: The key's name in the database
    param value: The value matching the key
    using example: print(DataQuery.QueryOne("test","Name","Han Wang"))
    """
if __name__ == '__main__':
    db = DataStorage()
    db.find()