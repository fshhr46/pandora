from enum import Enum

import logging

import os
import glob
import json
import shutil
import traceback
from typing import Dict, Tuple, List

import torch.multiprocessing as mp
from transformers import WEIGHTS_NAME

from pandora.tools.common import logger

DATASET_FILE_NAME = "dataset.json"
TRAINING_JOB_PREFIX = "PANDORA_TRAINING"
KEYWORD_JOB_PREFIX = "PANDORA_KEYWORD"


class JobStatus(str, Enum):
    not_started = "not_started"
    running = "running"
    terminated = "terminated"
    completed = "completed"
    packaged = "packaged"


class JobType(str, Enum):
    training = "training"
    keywords = "keywords"

    @classmethod
    def get_job_prefix(cls, job_type):
        if job_type == cls.training:
            return TRAINING_JOB_PREFIX
        elif job_type == cls.keywords:
            return KEYWORD_JOB_PREFIX
        else:
            return None


def get_all_checkpoints(output_dir):
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    )
    return checkpoints


def get_job_output_dir(output_dir: str, prefix: str, job_id: str) -> str:
    folder_name = get_job_folder_name_by_id(prefix=prefix, job_id=job_id)
    return os.path.join(output_dir, folder_name)


def get_job_folder_name_by_id(prefix: str, job_id: str) -> str:
    return f"{prefix}_{job_id}"


def get_dataset_file_path(server_dir: str, prefix: str, job_id: str) -> str:
    output_dir = get_job_output_dir(server_dir, prefix=prefix, job_id=job_id)
    return os.path.join(output_dir, DATASET_FILE_NAME)


def list_running_jobs(prefix: str = None) -> Dict[str, mp.Process]:
    all_processes = mp.active_children()

    training_jobs = all_processes
    if prefix:
        training_jobs = list(filter(lambda process: process.name.startswith(
            prefix), training_jobs))
        logger.info(
            f"Found {len(all_processes)} running processes, {training_jobs} are training jobs")
    else:
        logger.info(
            f"Found {len(all_processes)} running processes.")
    output_dic = {job.name: job for job in training_jobs}
    return output_dic


def list_all_jobs(server_dir: str, prefix: str) -> Dict[str, str]:
    if prefix:
        search_key = f"{prefix}*"
    else:
        search_key = "*"
    jobs_full_path = glob.glob(os.path.join(
        server_dir, search_key))
    output_dic = {os.path.basename(path): path for path in jobs_full_path}
    return output_dic


def cleanup_artifacts(server_dir: str, prefix: str, job_id: str) -> Tuple[bool, str]:
    output_dir = get_job_output_dir(
        server_dir, prefix=prefix, job_id=job_id)
    try:
        shutil.rmtree(output_dir)
        return True, ""
    except Exception as e:
        return False, traceback.format_exc()


def get_log_path(output_dir, job_type):
    return os.path.join(output_dir, f'{job_type}_job.log')


# TODO: Unify this setup config with model config
def create_setup_config_file(
        package_dir,
        setup_config_file_name,
        bert_base_model_name,
        bert_model_type,
        training_type,
        meta_data_types,
        eval_max_seq_length,
        num_labels: str):
    setup_conf = {
        "bert_base_model_name": bert_base_model_name,
        "bert_model_type": bert_model_type,
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
