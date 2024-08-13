import pickle

def get_id(filename):
    f = open(filename, 'r')
    id_list = []
    count = 0
    for line in f.readlines():
        count += 1
        id = line[0:4]
        id_list.append(id)
    print(count)
    id_list = sorted(set(id_list), key=id_list.index)
    print(len(id_list))
    print(id_list)
    return id_list

# id_list = get_id(r"H:\biodata\pdbtm\test.fasta")
# id_list += get_id(r"H:\biodata\pdbtm\train.fasta")
# id_list += get_id(r"H:\biodata\pdbtm\valid.fasta")
id_list = get_id(r'H:\biodata\PDBfile\pdbtm_all.list.txt')
f = open(r"H:\biodata\pdbtm\all_pdb_id.pickle", "wb")
pickle.dump(id_list, f)
