from . import DataFormat as df
from ..Utilities.Files import TxtFileDealer as tf
from ..Utilities.Files import FileSysDealer as fs
from . import ParserBase as ppd


def func():
    pass


class ParserStructure(object):
    '''
    This parser is an advanced parser of PDBparser file.
    Classify the result of ParserBase into a structural dict
    '''
    def __init__(self):
        self._data_format = dict()

    def parse(self, file):
        '''
        the interface of the class
        :param file:    PDBparser file name
        :return:        a structural dict
        '''
        pb = ppd.ParserBase(func)
        pdblist = pb.parser(file, target="ALL")
        result = self.parseStructure(pdblist)
        result["File"] = file
        return result

    def parseStructure(self, list):
        '''
        the interface of the class
        :param list:    the result of ParserBase into a structural dict
        :return:        a structural dict
        '''
        dict = {}
        self._data_format = df.DataFormat().txt_based_data
        for key in self._data_format.keys():
            if key not in ["MODEL", "ATOM", "HETATM", "ANISOU", "SIGUIJ", "ENDMDL", "TER", "END"]:
                dict.update({key: []})
        for line in list:
            for mark, content in line.items():
                if mark not in ["MODEL", "ATOM", "HETATM", "ANISOU", "SIGUIJ", "ENDMDL", "TER", "END"]:
                    if mark in dict.keys():
                        dict[mark].append(content)
        dict["REMARK"] = self.remark_classification(dict["REMARK"])
        dict["SEQRES"] = self.seqres_classification(dict["SEQRES"])
        dict["SITE"] = self.site_classification(dict["SITE"])
        dict["CHAINS"] = self.chain_classification(list)
        # for mark, content in dict.items():
        #     print(mark)
        #     print(content)
        return dict

    def site_classification(self, list):
        '''
        classify the 'SITE' information
        :param list: a list of dicts, each dict correspond a line start with 'site' in PDBparser file
        :return: a list of dict dicts include the site info
                 every dict include: 'siteID':      the name of this site
                                     'numRes':      the number of residues in this site
                                     'resName':     the name of the residues in this site
        '''
        site_list = []
        current_sitename = ""
        for content in list:
            if content["siteID"] != current_sitename:
                site = {"siteID": content["siteID"], "numRes": content["numRes"], "resName": []}
                site_list.append(site)
                current_sitename = content["siteID"]
            for i in range(1, 5):
                resName = "resName"+str(i)
                chainID = "chainID"+str(i)
                seq = "seq"+str(i)
                iCode = "iCode"+str(i)
                if content[resName] != "":
                    site_res = {"chainID": content[chainID], "seq": content[seq], "resName": content[resName],
                                "iCode": content[iCode]}
                    for site in site_list:
                        if site["siteID"] == current_sitename:
                            site["resName"].append(site_res)
        return site_list

    def seqres_classification(self, list):
        '''
        classify the 'SEQRES' information
        :param list: a list of dicts, each dict correspond a line start with 'seqres' in PDBparser file
        :return: a list of dict dicts include the seqres info
                 every dict include: 'chainID':     the name of this chain
                                     'numRes':      the number of residues in this chain
                                     'resName':     the name of the residues in this chain
        '''
        chain_list = []
        current_chainid = ""
        for content in list:
            if content["chainID"] != current_chainid:
                chain = {"chainID": content["chainID"], "numRes": content["numRes"], "resName": []}
                chain_list.append(chain)
                current_chainid = content["chainID"]
            for chain in chain_list:
                if chain["chainID"] == current_chainid:
                    chain["resName"] += content["resName"]
        return chain_list


    def remark_classification(self, list):
        '''
        classify the 'REMARK' information
        :param list: a list of dicts, each dict correspond a line start with 'remark' in PDBparser file
        :return:     a list of dict dicts include the remark info
                     every dict include: 'remarkNum':   the mark number of this remark
                                         'empty':       a list of str, every str correspond one line start with 'remark'
        '''
        remark_list = []
        current_remarkNum = 0
        for content in list:
            if content["remarkNum"] != current_remarkNum:
                remark = {"remarkNum": content["remarkNum"], "empty": []}
                remark_list.append(remark)
                current_remarkNum = content["remarkNum"]
            for remark in remark_list:
                if remark["remarkNum"] == current_remarkNum:
                    remark["empty"].append(content["empty"])
        return remark_list

    def chain_classification(self, list):
        '''
        classify the atomic related information. Atomic related include "MODEL", "ATOM", "HETATM", "ANISOU", "SIGUIJ",
                                                                        "ENDMDL", "TER".
        :param list: a list of dicts, each dict correspond a atomic related line in PDBparser file
        :return:     a list of dict dicts include the atomic info
                     every dict include: 'chainID':         the name of this chain
                                         'chainStructure':  a list of models
        '''
        chains = self.init_chains(list)
        for line in list:
            for mark, content in line.items():
                if mark == "MODEL" or mark == "ENDMDL":
                    for chain in chains:
                        chain["chainStructure"].append(line)
                if mark == "ATOM" or mark == "HETATM" or mark == "ANISOU" or mark == "SIGUIJ":
                    for chain in chains:
                        if content["chainID"] == chain["chainID"]:
                            chain["chainStructure"].append(line)
        chains = self.chain_to_model(chains)
        return chains

    def init_chains(self, linelist):
        '''
        iterate the list to get all the chainIDs
        :param linelist: list of atomic related info
        :return: a list of dict of chains, 'chainStructure' is an empty list to be complete with models
        '''
        idlist = []
        resultlist = []
        for line in linelist:
            for mark, content in line.items():
                if mark == "ATOM" or mark == "HETATM":
                    if content["chainID"] not in idlist:
                        idlist.append(content["chainID"])
        for chainid in idlist:
            chain ={}
            chain["chainID"] = chainid
            chain["chainStructure"] = []
            resultlist.append(chain)
        return resultlist

    def chain_to_model(self, chains):
        '''
        :param list: a list of dicts, each dict is a chain
        :return:     Complete 'chainStructure' with models
                     every dict include: 'serial':          the mark number of this remark
                                         'modelStructure':  a list of residues
        '''
        for chain in chains:
            model_open = 0
            current_model_id = 0
            model_list = []
            for line in chain["chainStructure"]:
                for mark, content in line.items():
                    if mark == "MODEL":
                        model_open = 1
                        current_model_id = content["serial"]
                        model = {"serial": current_model_id, "modelStructure": [] }
                        model_list.append(model)
                    if mark == "ATOM" or mark == "HETATM" or mark == "ANISOU" or mark == "SIGUIJ":
                        if model_open == 0:
                            current_model_id = 1
                            model_open = 1
                            model = {"serial": current_model_id, "modelStructure": []}
                            model_list.append(model)
                        for model in model_list:
                            if model["serial"] == current_model_id:
                                model["modelStructure"].append(line)
            model_list = self.model_to_residue(model_list)
            chain["chainStructure"] = {"models": model_list}
        return chains

    def model_to_residue(self, models):
        '''
        :param list: a list of dicts, each dict is a model
        :return:     Complete 'modelStructure' with residues
                     every dict include: 'resName':         the name of the residue
                                         'resSeq':          the sequence number of this residude
                                         'resStructure':    a list of 'ATOM' 'HETATM' 'ANISOU' 'SIGUIJ'
        '''
        for model in models:
            current_res_id = 0
            res_list = []
            for atoms in model["modelStructure"]:
                for mark, content in atoms.items():
                    if content["resSeq"] != current_res_id:
                        res = {"resName": content["resName"], "resSeq": content["resSeq"], "resStructure": []}
                        res_list.append(res)
                        current_res_id = content["resSeq"]
                    for res in res_list:
                        if res["resSeq"] == current_res_id:
                            res["resStructure"].append(atoms)
            model["modelStructure"] = res_list
        return models


if __name__ == '__main__':
    file = r'H:\biodata\pdbtm\pdb_all\1a0s.pdb'
    ps = ParserStructure()
    result = ps.parse(file)
    for key, content in result.items():
        print(key)
        print(content)

