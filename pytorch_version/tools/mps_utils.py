import torch

import tools.common as common

cpu = common.get_device("cpu")
mps = common.get_device("mps")


def mask_select_on_cpu(input, mask):
    if torch.has_mps:
        return input.to(cpu)[mask.to(cpu)].to(mps)
    else:
        return input[mask]
