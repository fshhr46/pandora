import torch
from sys import platform

import pandora.tools.common as common

cpu = common.get_device("cpu")
mps = None

# TODO: fix MPS randomness issue
has_mps = platform == "darwin" and False and torch.has_mps
has_mps = False

if has_mps:
    mps = common.get_device("mps")


def mask_select_on_cpu(input, mask):
    if has_mps:
        return input.to(cpu)[mask.to(cpu)].to(mps)
    else:
        return input[mask]
