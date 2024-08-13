import os

rsync_dict = {
    "PDB_Format": r"rsync -rlpt -v -z --delete --port=33444 \rsync.rcsb.org::ftp_data/structures/divided/pdb/ ",
    "PDBML_Format": r"rsync -rlpt -v -z --delete --port=33444 \rsync.rcsb.org::ftp_data/structures/divided/XML/ "
}


class RsyncDownload():
    '''
    this class is to download files by rsync
    '''
    def __init__(self):
        self.rsync_dict = rsync_dict

    def download(self, target, dir):
        '''
        the interface of the class
        :param target:  what kind of file you want
        :param dir:     the dir you want to put the files
        :return:        no return yet
        '''
        cmd = self.rsync_dict[target] + dir
        os.system(cmd)


# dl = RsyncDownload()
# dl.download("PDB_Format", "./pdb")
# dl.download("PDBML_Format", "./XML")
