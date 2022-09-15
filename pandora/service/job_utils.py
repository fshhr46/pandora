
import os
import glob

from pandora.models.transformers import WEIGHTS_NAME


def get_all_checkpoints(output_dir):
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    )
    return checkpoints
