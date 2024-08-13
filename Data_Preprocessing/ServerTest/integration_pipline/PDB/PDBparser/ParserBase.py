
from . import DataFormat as df
from ..Utilities.Files import TxtFileDealer as tf
from ..Utilities.Files import FileSysDealer as fs


def func():
    pass


class ParserBase(object):
    """
    Parser the protein structure files from Protein Data Bank (PDBparser), including various types of file.
    The purpose of this class is just parser the original data without any change.
    This class can be used as a decorator class or an interface, the further usages of the resolved data will depended on the caller of this class.
    """

    def __init__(self, func):
        self.__func = func
        self._data_format = dict()

    def __call__(self, *args, **kwargs):
        """
        To realize this class as a decorator.
        NOTE: Uncompleted!!!
        :param args:
        :param kwargs:
        :return:
        """
        self.__func(*args, **kwargs)

    def parser(self, file: str, target="ALL"):
        """
        The realized unique interface to be called parser a protein structure file.
        Various file types are allowed input, the process of parser will be different according to the suffix of the input file.
        NOTE: in current version, only text-based file parser is available.
        :param file:    a PDBparser file
        :param target:  the record to be parser from the file, for example: if target="ATOM", parser only  atom records, or target="ALL", parser all
                        the records in the file.
        :return:        a list
        """
        assert fs.FilSysDealer.verify_file_readable(file)
        suffix = fs.FilSysDealer.suffix(file)
        # Parser file according to the file suffix
        content = {}
        if suffix == "ent" or suffix == "pdb":
            content = self._parser_txt_based(file, target)
        if suffix == "xml":
            pass
        else:
            pass
        return content

    def _parser_txt_based(self, file: str, target="ALL"):
        """
        Inner function: Parer a text-base PDBparser file.
        :param file:    a text-based PDBparser file
        :param target:  the record to be parser from the file, for example: if target="ATOM",
                        parser only  atom records, or target="ALL", parser all the records in the file.
        :return:        a dict
        """
        data = []
        self._data_format = df.DataFormat().txt_based_data
        for line in tf.TxtFileDealer.yield_read_line(file):
            for field, content in self._parser_line(line, target).items():
                data.append({field: content})
        return data


    def _parser_line(self, line: str, target: str):
        """
        inner funciton: parser a line read from a text-based PDBparser file
        :param line:    a str of line
        :param target:  the record to be parser
        :return: a dict with one element, the key is the record name, the value is the data of the line
        """
        assert len(line) >= 2
        # assert self._data_format.isinstance(dict)
        data = {}
        mark = line[0:6].strip()
        # discard the line different with the target
        if target != "ALL"and target != mark:
            return data
        # parer the data in line according to the record name respectively
        if mark in self._data_format.keys():
            line_format = self._data_format.get(mark)
            for field in line_format.keys():
                start = line_format.get(field).get('start')
                end = line_format.get(field).get('end')
                if len(start) > 1:
                    datatype = line_format.get(field).get('datatype')
                    value = []
                    for i in range(len(start)):
                        if line[start[i]-1:end[i]].strip() != '':
                            value.append(datatype(line[start[i]-1:end[i]].strip()))
                else:
                    datatype = line_format.get(field).get('datatype')
                    if line[start[0]-1:end[0]].strip() != '':
                        value = datatype(line[start[0]-1:end[0]].strip())
                    else:
                        value = ''
                data.update({field: value})
        return {mark: data}

if __name__ == '__main__':
    file = r'H:\biodata\pdbtm\pdb_all\1a0s.pdb'
    ps = ParserBase(func)
    result = ps.parser(file, target="ALL")
    for line in result:
        print(line)
