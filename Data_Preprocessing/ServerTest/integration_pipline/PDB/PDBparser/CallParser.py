import PDB.PDBparser.ParserBase as ppb
import PDB.PDBparser.ParserStructure as pps
import os
import PDB.Utilities.Files.FileSysDealer as fs
import PDB.PDBparser.DatabaseOperation as do
import pymongo
import pymongo.errors as er

class callParser(object):
    def parser(self, file):
        ps = pps.ParserStructure()
        result = ps.parse(file)
        return result

    def parserIter(self, dir):
        file_list = []
        file_list = self.get_filelist(dir, file_list)
        print(len(file_list))
        ps = pps.ParserStructure()
        for file in file_list:
            result = ps.parse(file)
            # print(file)
            yield result

    def storagePDB(self, dir):
        iter = self.parserIter(dir)
        database_operation = do.DataStorage()
        database_operation.StorageIter("PDBparser", iter)




    def get_filelist(self, dir, Filelist):
        """get the file list under the direction"""
        newDir = dir
        if os.path.isfile(dir):
            suffix = fs.FilSysDealer.suffix(dir)
            if suffix == "ent":
                Filelist.append(dir)
        elif os.path.isdir(dir):
            for s in os.listdir(dir):
                newDir = os.path.join(dir, s)
                self.get_filelist(newDir, Filelist)
        return Filelist


if __name__ == '__main__':
    cp = callParser()
    cp.storagePDB(r"/data/OrignalData/PDBparser/pdb")
