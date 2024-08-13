##!/usr/bin/python3
"""
  Author:  H.Wang
  Purpose: Offer the common methods for file system.
  Created: 5/2/2019
"""

import os


class FilSysDealer:
    """

    """
    @staticmethod
    def verify_file_writeable(file):
        return os.access(file, os.W_OK)

    @staticmethod
    def verify_file_readable(file):
        return os.access(file, os.R_OK)

    @staticmethod
    def suffix(file):
        return os.path.splitext(file)[-1][1:]

    @staticmethod
    def verify_path(path):
        pass

    @staticmethod
    def create_file(file):
        pass

    @staticmethod
    def creat_folder(folder):
        pass

    @staticmethod
    def delete_file(file):
        pass

    @staticmethod
    def folder_tree():
        pass

    @staticmethod
    def files_in_folder():
        pass
