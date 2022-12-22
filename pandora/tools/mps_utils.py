import os
from sys import platform

import torch
import pandora.tools.common as common

cpu = common.get_device("cpu")
mps = None

# TODO: fix MPS randomness issue
has_mps = platform == "darwin" and torch.has_mps
# has_mps = False

if has_mps:
    mps = common.get_device("mps")
    # assert os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def mask_select_on_cpu(input, mask):
    if has_mps:
        return input.to(cpu)[mask.to(cpu)].to(mps)
    else:
        return input[mask]
