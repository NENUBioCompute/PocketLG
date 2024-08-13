# -*- coding:utf-8 -*-
'''
	File Name：     jsonDealer
	Description :   get json files and rewrite them
	Author :        Gong Yingli & Liu Zhe
	date：          2018/12/12
'''

import json


class jsonDealer:

    def __init__(self):
        self.flag = 0
        pass

    def get_json_data(self, filepath):

        with open(filepath, 'rb') as f:
            json_data = json.load(f)

        return json_data

    def change_json_data(self, json_data, json_new):
        # only deal with the change of col's number,no increase or decrease

        for part_dict in json_data.keys():
            # print(part_dict)
            if part_dict in json_new.keys():
                if type(json_data[part_dict]) is dict and type(json_new[part_dict]) is dict:
                    for key in json_data[part_dict].keys():
                        # print(key)
                        try:
                            if json_data[part_dict][key] != json_new[part_dict][key]:
                                self.flag = 1
                                print(part_dict, key, json_data[part_dict][key], json_new[part_dict][key])
                                json_data[part_dict][key] = json_new[part_dict][key]
                        except KeyError as e:
                            print("Error", e)
                else:
                    print("part_dict is not a dict")
            else:
                print("None", part_dict)

        return

    def rewrite_json_file(self, filepath, json_data):

        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=4)
        return


if __name__ == '__main__':
    pass
