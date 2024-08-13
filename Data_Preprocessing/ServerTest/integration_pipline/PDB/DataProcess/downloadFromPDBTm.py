import pickle
import os
import threading
from shutil import copyfile
from pathlib import Path

def downloadThread(cmd):
    os.system(cmd)


# f = open(r"H:\biodata\pdbtm\pdb_id.pickle", "rb")
# pdb_id = pickle.load(f)
# print(len(pdb_id))
# idlists = []
# n =0
# idid=[]
# for i in range(15):
    # if n + 50 > len(pdb_id):
    #     m = len(pdb_id) + 1
    # else:
    #     m = n+50
    # id_list = pdb_id[n:m]
    # path = r"H:\biodata\pdbtm\pdb_id" + str(i) +".pickle"
    # f = open(path, "wb")
    # pickle.dump(id_list, f)
    # idlists.append(id_list)
    # n += 50


path = r"H:\biodata\pdbtm\all_pdb_id.pickle"
f = open(path, "rb")
pdb_id = pickle.load(f)
pdb_id = sorted(set(pdb_id),key=pdb_id.index)
print(len(pdb_id))
filecount = 0
not_exist_pdb = []
not_exist_file_count = 0
for PDB_ID in pdb_id:
#     # cmd = r'wget -c --timeout=30 -N -P H:\biodata\pdbtm\pdbtest http://pdbtm.enzim.hu/data/database/' + PDB_ID[1:3] + '/' + PDB_ID[0:4]+'.pdb.gz'
#     # print(cmd)
#     # os.system(cmd)
    src = 'H:\\biodata\\PDBfile\\PDBparser\\' + PDB_ID[1:3] + '\\pdb' + PDB_ID[0:4]+'.ent.gz'
    print(src)
    dir = 'H:\\biodata\\pdbtm\\pdb_all\\' +PDB_ID[0:4] + '.pdb.gz'
    if os.path.exists(src):
        copyfile(src, dir)
        filecount += 1
    else:
        not_exist_file_count +=1
        not_exist_pdb.append(PDB_ID)
print(not_exist_pdb)
print(filecount)
print(not_exist_file_count)
# f = open(r'H:\biodata\pdbtm\not_exist.pickle', 'wb')
# pickle.dump(not_exist_pdb, f)

# path = r"H:\biodata\pdbtm\pdb_id.pickle"
# f = open(path, "rb")
# pdb_id = pickle.load(f)
# pdb_id = sorted(set(pdb_id),key=pdb_id.index)
# print(len(pdb_id))
# f.close()
# f = open(r"H:\biodata\pdbtm\pdb_id.txt", 'w')
# for id in pdb_id:
#     print(id)
#     f.write(id)