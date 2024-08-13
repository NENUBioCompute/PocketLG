# @Time    : 2023/12/20 14:48
# @Author  : Heng Chang

import subprocess
import sys
from pathlib import Path
import concurrent.futures  # 并发执行

sys.path.append("./ServerTest/integration_pipline")
sys.path.append("./ServerTest")


from ServerTest.integration_pipline.data_preprocessing3 import data_preprocessing3

def sampled_pdb(Protein_in_path, Pocket_path, msms_path):
    p_in = Path(Protein_in_path).resolve()

    path = Path(sys.argv[0]).parent.resolve()
    p_out = path / "tmp"
    if not p_out.exists():
        p_out.mkdir(parents=True, exist_ok=True)

    for mfile in p_out.glob("*"):
        mfile.unlink()

    Pocket_path = Path(Pocket_path).resolve()
    #print(Pocket_path)

    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        for pdb_file in p_in.glob("*.pdb"):
            a = data_preprocessing3(pdb_file, Pocket_path, MSMS=msms_path, w_dir=p_out)

            #res_num = pdb_file.stem.split("_")[-1]
            res_num = 70
            executor.submit(a.start_processing, int(res_num))


if __name__ == "__main__":
    path = Path(sys.argv[0]).parent.resolve()

    # 设置 采样后的蛋白文件夹
    Protein_out = path / ""

    Protein_out = Path(Protein_out).resolve()
    if not Protein_out.exists():
       Protein_out.mkdir(parents=True, exist_ok=True)

    # 设置 比对的蛋白pocket文件夹
    Pocket_path = path /"Pocket1"
    Pocket_path = Path(Pocket_path).resolve()
    if not Pocket_path.exists():
        Pocket_path.mkdir(parents=True, exist_ok=True)

    # 设置 采样前的蛋白文件夹
    Protein_in = path /"Protein1"
    Protein_in = Path(Protein_in).resolve()
   # if not Protein_out.exists():
   #     Protein_in.mkdir(parents=True, exist_ok=True)

    # 设置 msms路径
    if "win" in sys.platform:
        msms_path = path / "ServerTest" / "integration_pipline" / "msms.exe"
    elif "linux" in sys.platform:
        msms_path = path / "ServerTest" / "integration_pipline" / "msms"

    msms_path = Path(msms_path).resolve()

    # 以上路径请自行修改.......

    sampled_pdb(Protein_in, Pocket_path, msms_path)

    #compare_pdb(Protein_out, Pocket_path)
