import os
import pathlib

curr_dir = str(pathlib.Path(os.path.dirname(__file__)).absolute())

NAMES_CN = open(os.path.join(curr_dir, "names.cn.txt")).read().splitlines()

NAMES_US = open(os.path.join(curr_dir, "names.us.txt")).read().splitlines()
