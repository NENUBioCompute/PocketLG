from MultiThread import thread_manager
import os
from PDBParser.ParserRegulate import *


def file_open(file_name, q):
    """ file reade thread"""
    Regulate = ParserRegulate()
    result = Regulate.Regulate(file_name, 'ATOM', ['x', 'y', 'z'])
    q.put(result)
    print(file_name + " done")



@thread_manager
def file_reader(path):
    filelist = os.listdir(path)
    print(filelist)
    thread_list = []
    for i in filelist:
        i = "pdbfiles/" + i
        thread_list.append([file_open, i])
    return thread_list

def main():
    # path = input("please input file direction:\n")
    path = "pdbfiles"
    result = file_reader(path)
    pprint.pprint(result)


if __name__ == '__main__':
    main()
