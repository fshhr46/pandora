
import os
import glob
import json

from transformers import WEIGHTS_NAME


def get_all_checkpoints(output_dir):
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    )
    return checkpoints

# TODO: Unify this setup config with model config


def create_setup_config_file(
        package_dir,
        setup_config_file_name,
        training_type,
        meta_data_types,
        eval_max_seq_length,
        num_labels: str):
    setup_conf = {
        "model_name": "bert-base-chinese",
        "mode": "sequence_classification",
        "training_type": training_type,
        "meta_data_types": meta_data_types,
        "do_lower_case": True,
        "num_labels": num_labels,
        "save_mode": "pretrained",
        # TODO: This needs to be aligned with traning/eval? current set to eval's "eval_max_seq_length".
        "max_length": eval_max_seq_length,
        "embedding_name": "bert",
        "FasterTransformer": False,  # TODO: make this True
        "model_parallel": False  # Beta Feature, set to False for now.
    }
    setup_conf_path = os.path.join(package_dir, setup_config_file_name)
    with open(setup_conf_path, "w") as setup_conf_f:
        json.dump(setup_conf, setup_conf_f, indent=4)
