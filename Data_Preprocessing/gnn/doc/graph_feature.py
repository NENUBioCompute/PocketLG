import json
from pathlib import Path
from kdbnet import pdb_graph
class ProteinDataHandler:
    def __init__(self, pdb_json_path, emb_dir=None, prot_featurize_params=None):
        self.pdb_json_path = pdb_json_path
        self.emb_dir = emb_dir
        self.prot_featurize_params = prot_featurize_params
        self._data = None

    @property
    def data(self):
        if self._data is None:
            with open(self.pdb_json_path, 'r') as json_file:
                self._data = json.load(json_file)
        return self._data

    def _format_pdb_entry(self, _data):
        _coords = _data["coords"]
        entry = {
            "name": _data["PDB_id"],
            "seq": _data["seq"],
            #"ESM": _data["ESM"],
            "coords": list(zip(_coords["N"], _coords["CA"], _coords["C"], _coords["O"])),
        }
        if self.emb_dir is not None:
            embed_file = f"{_data['PDB_id']}.pt"  # Assuming chain information is not needed for embedding file
            entry["embed"] = f"{self.emb_dir}/{embed_file}"
        return entry

    @property
    def prot2pdb(self):
        prot2pdb_data = {}
        for pdb_id, pdb_entry in self.data.items():
            prot2pdb_data[pdb_id] = self._format_pdb_entry(pdb_entry)
        return prot2pdb_data

    @property
    def pdb_graph_db(self):
        if self.prot_featurize_params is None:
            raise ValueError("Protein featurization parameters are required.")
        return pdb_graph.pdb_to_graphs(self.prot2pdb, self.prot_featurize_params)




def MainGraph():
    # 使用示例


    pdb_json_train = r'C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_outjson.json'

    emb_dir = None
    prot_featurize_params = {
        "num_pos_emb": 16,
        "num_rbf": 16,
        "contact_cutoff": 8.0
    }

    # 创建 ProteinDataHandler 实例
    protein_handler1 = ProteinDataHandler(pdb_json_train, emb_dir=emb_dir, prot_featurize_params=prot_featurize_params)

    # 获取蛋白质图表示的数据库
    pdb_graph_db = protein_handler1.pdb_graph_db


    return pdb_graph_db

MainGraph()

# if __name__ == '__main__':
#     def delX_CAno70():
#         pdb_json_train = r'C:\Users\Lenovo\Desktop\protein_out_1xo2\protein_out\protein_outjson.json'
#         # pdb_json_test = r'E:\PDBBind\Result\5000-10000_tmscore_70actxt\pdb_26230jsonv2.json'
#         pdb_graph_db = MainGraph()
#         count = 0
#         delarray = []
#         for key, value in pdb_graph_db.items():
#             if value.x.size()[0] != 70 or value.x.size()[1] != 3:
#                 print(f"Key: {key}, Value: {value}")
#                 delarray.append(key)
#                 count += 1
#
#         print(f"Total mismatches: {count}")
#         # 读取 JSON 文件
#         with open(pdb_json_train, 'r') as json_file:
#             data = json.load(json_file)
#         # 从数据中移除要删除的键值对
#         for key in delarray:
#             del data[key]
#
#         # 将更新后的数据写回到 JSON 文件中
#         with open(pdb_json_train, 'w') as json_file:
#             json.dump(data, json_file, indent=4)
#         print('done!!!!')
#     delX_CAno70()






