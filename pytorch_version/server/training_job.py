from enum import Enum
from genericpath import isdir, isfile
import logging
import multiprocessing
import os
import json
import shutil
import traceback
import glob

from typing import Dict, Tuple
import runner
from tools.common import logger

MAX_RUNNING_JOBS = 1
JOB_PREFIX = "PANDORA_TRAINING"
logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    not_started = "not_started"
    running = "running"
    terminated = "terminated"
    completed = "completed"


class TrainingJob(object):

    bert_base_model_name = "bert-base-chinese"
    task_name = "sentence"
    mode_type = "bert"

    def __init__(self,
                 job_id,
                 data_dir,
                 output_dir,
                 cache_dir) -> None:
        self.job_id = job_id
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir

        # Training parameters
        self.sample_size = 20

    def __call__(self, *args, **kwds) -> None:
        arg_list = runner.get_training_args(
            task_name=self.task_name,
            mode_type=self.mode_type,
            bert_base_model_name=self.bert_base_model_name,
            sample_size=self.sample_size,
        )

        dir_args = [f"--data_dir={self.data_dir}",
                    f"--output_dir={self.output_dir}/",
                    f"--cache_dir={self.cache_dir}", ]
        arg_list.extend(dir_args)

        # set actions
        arg_list.extend(
            runner.set_actions(
                do_train=True,
                do_eval=True,
                do_predict=True,
            ))
        runner.train_eval_test(arg_list)


def start_training_job(
        job_id: str,
        data_dir,
        server_dir,
        cache_dir) -> Tuple[bool, str]:
    if not _has_enough_resource():
        message = f"not enough resource to start a new training job."
        logger.info(message)
        return False, message
    else:
        logger.info("got enough resource to start a new training job.")

    output_dir = _get_job_output_dir(server_dir, job_id)
    if os.path.isdir(output_dir):
        message = f"output folder {output_dir} is not empty. failed to start a new training job."
        logger.info(message)
        return False, message

    job = TrainingJob(
        job_id=job_id,
        data_dir=data_dir,
        output_dir=output_dir,
        cache_dir=cache_dir)
    job_process = multiprocessing.Process(
        name=_get_process_name_by_id(job_id=job_id), target=job)
    job_process.daemon = True
    job_process.start()
    message = f'Started training job with ID {job_id}'
    logger.info(message)
    return True, ""


def stop_training_job(job_id: str) -> Tuple[bool, str]:
    active_training_jobs = list_training_jobs()
    job_name = _get_process_name_by_id(job_id=job_id)
    for name, job in active_training_jobs.items():
        if job_name == name:
            logger.info(f"Found active training job with ID {job_id}")
            job.terminate()
            return True, f"Successfully stopped training job with ID  {job_id}"
    return False, f"Failed to find active training job with ID {job_id}"


def get_report(server_dir: str, job_id: str, include_data: bool = True) -> str:
    report_dir = _get_report_output_dir(server_dir, job_id)

    output = {}
    if not os.path.isdir(report_dir):
        return {}
    paths = {}
    data = {}
    output["paths"] = paths
    output["data"] = data

    report_all_path = os.path.join(report_dir, "report_all.json")
    if os.path.isfile(report_all_path):
        paths["report_all_path"] = report_all_path
        if include_data == True:
            with open(report_all_path, encoding="utf-8") as f:
                data["report_all_data"] = json.load(f)

    report_by_class_path = os.path.join(report_dir, "report.json")
    if os.path.isfile(report_by_class_path):
        paths["report_by_class_path"] = report_by_class_path
        if include_data:
            with open(report_by_class_path, encoding="utf-8") as f:
                data["report_by_class_data"] = json.load(f)
            output["data"] = data
    return output


def cleanup_artifacts(server_dir: str, job_id: str) -> Tuple[bool, str]:
    if is_job_running(job_id=job_id):
        return False, f"can no delete a running job_id {job_id}."
    output_dir = _get_job_output_dir(server_dir, job_id)
    try:
        shutil.rmtree(output_dir)
        return True, ""
    except Exception as e:
        return False, traceback.format_exc()


def is_job_running(job_id: str) -> bool:
    job_name = _get_process_name_by_id(job_id=job_id)
    return job_name in list_training_jobs().keys()


def get_training_status(server_dir: str, job_id: str) -> str:
    if is_job_running(job_id=job_id):
        return JobStatus.running
    else:
        output_dir = _get_job_output_dir(server_dir, job_id=job_id)
        if os.path.exists(output_dir):
            report_dir = _get_report_output_dir(server_dir, job_id)
            if os.path.isdir(report_dir):
                return JobStatus.completed
            else:
                return JobStatus.terminated
        else:
            return JobStatus.not_started


def list_training_jobs() -> Dict[str, multiprocessing.Process]:
    all_processes = multiprocessing.active_children()
    training_jobs = list(filter(lambda process: process.name.startswith(
        JOB_PREFIX), all_processes))
    logger.info(
        f"Found {len(all_processes)} running processes, {training_jobs} are training jobs")
    output_dic = {job.name: job for job in training_jobs}
    return output_dic


def list_all_jobs(server_dir: str) -> Dict[str, str]:
    jobs_full_path = glob.glob(os.path.join(server_dir, f"{JOB_PREFIX}*"))
    output_dic = {os.path.basename(path): path for path in jobs_full_path}
    return output_dic


def _has_enough_resource() -> bool:
    active_training_jobs = list_training_jobs()
    return len(active_training_jobs) < MAX_RUNNING_JOBS


def _get_process_name_by_id(job_id: str) -> str:
    return f"{JOB_PREFIX}_{job_id}"


def _get_job_output_dir(output_dir: str, job_id: str) -> str:
    folder_name = _get_process_name_by_id(job_id)
    return os.path.join(output_dir, folder_name)


def _get_report_output_dir(server_dir: str, job_id: str) -> str:
    output_dir = _get_job_output_dir(server_dir, job_id)
    return os.path.join(output_dir, "predict")
