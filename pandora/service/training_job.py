from genericpath import isdir, isfile
import logging
import os
import json
import traceback

import torch.multiprocessing as mp
from typing import Dict, Tuple, List
from pandora.dataset import poseidon_data
import pandora.service.job_runner as job_runner
import pandora.service.job_utils as job_utils
from pandora.service.job_utils import JobStatus, JobType
from pandora.tools.common import logger
import pandora.packaging.packager as packager
import pandora.dataset.dataset_utils as dataset_utils
import pandora.tools.training_utils as training_utils
import pandora.tools.common as common

from pandora.packaging.feature import (
    TrainingType,
    MetadataType,
)
from pandora.packaging.model import BertBaseModelType
from pandora.packaging.losses import LossType


class TrainingJob(object):

    def _set_mode_type_and_name(
            self,
            training_type: TrainingType,
            meta_data_types: List[str]):

        self.bert_model_type, self.bert_base_model_name = training_utils.get_mode_type_and_name(
            training_type,
            meta_data_types
        )

    def __init__(self,
                 job_id,
                 data_dir,
                 output_dir,
                 cache_dir,
                 sample_size: int,
                 training_type: TrainingType,
                 meta_data_types: List[str],
                 loss_type: LossType,
                 num_folds: int) -> None:
        self.job_id = job_id
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir

        # Training parameters
        self.sample_size = sample_size
        self.training_type = training_type
        self.meta_data_types = meta_data_types

        # Find the right model
        self._set_mode_type_and_name(
            self.training_type,
            self.meta_data_types,
        )
        self.loss_type = loss_type

        # Whether to perform cross validation
        self.num_folds = num_folds

    def __call__(self, *args, **kwds) -> None:

        num_epochs = training_utils.get_num_epochs(
            self.training_type,
            self.meta_data_types,
            bert_model_type=self.bert_model_type,
        )

        batch_size = training_utils.get_batch_size(
            self.training_type,
            self.meta_data_types,
            bert_model_type=self.bert_model_type,
        )

        arg_list = job_runner.get_training_args(
            # model args
            bert_model_type=self.bert_model_type,
            bert_base_model_name=self.bert_base_model_name,
            sample_size=self.sample_size,
            training_type=self.training_type,
            meta_data_types=self.meta_data_types,
            loss_type=self.loss_type,
            num_folds=self.num_folds,
            # training args
            num_epochs=num_epochs,
            batch_size=batch_size,
        )

        dir_args = [f"--data_dir={self.data_dir}",
                    f"--output_dir={self.output_dir}/",
                    f"--cache_dir={self.cache_dir}", ]
        arg_list.extend(dir_args)

        # set actions
        # Only do eval when cross_validation is disabled.
        arg_list.extend(
            job_runner.set_actions(
                do_train=True,
                do_eval=True,
                do_predict=True,
            ))

        resource_dir = dataset_utils.get_partitioned_data_folder(
            self.output_dir)

        # TODO: this is hacky as we relies on os.path.join("1", "", "2")
        # returns "1/2"
        #
        # and os.path.join("1", "dataset", "2")
        # returns "1/dataset/2"
        datasets = [""]
        job_runner.run_e2e_modeling(
            arg_list, resource_dir=resource_dir, datasets=datasets)


def start_training_job(
        job_id: str,
        server_dir,
        cache_dir,
        sample_size: int,
        loss_type) -> Tuple[bool, str]:
    # partitioned data locates in job/datasets/{train|dev|test}.json
    output_dir = job_utils.get_job_output_dir(
        server_dir, prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    partition_dir = dataset_utils.get_partitioned_data_folder(
        resource_dir=output_dir)
    if not os.path.isdir(partition_dir):
        return False, f"no dataset found in {partition_dir}"

    # load partition file
    data_partition_args_file_path = dataset_utils.get_data_partition_args_file_path(
        partition_dir)
    with open(data_partition_args_file_path) as data_partition_args_f:
        data_partition_args = json.load(data_partition_args_f)
    num_folds = data_partition_args["num_folds"]

    status = get_status(server_dir=server_dir, job_id=job_id)
    if status != JobStatus.not_started:
        message = f"You can only start a job with {JobStatus.not_started} status. current status {status}."
        logger.info(message)
        return False, message

    # load dataset file
    dataset_path = job_utils.get_dataset_file_path(
        server_dir, prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    if not os.path.isfile(dataset_path):
        return False, {}, f"dataset file {dataset_path} not exists"
    _, _, training_type, _, meta_data_types = poseidon_data.load_poseidon_dataset_file(
        dataset_path)

    job = TrainingJob(
        job_id=job_id,
        data_dir=partition_dir,
        output_dir=output_dir,
        cache_dir=cache_dir,
        sample_size=sample_size,
        training_type=training_type,
        meta_data_types=meta_data_types,
        loss_type=loss_type,
        num_folds=num_folds)
    mp.set_start_method("spawn", force=True)
    job_process = mp.Process(
        name=job_utils.get_job_folder_name_by_id(
            prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id), target=job)
    job_process.daemon = True
    job_process.start()
    message = f'Started training job with ID {job_id}'
    logger.info(message)
    return True, ""


def stop_job(job_id: str) -> Tuple[bool, str]:
    active_jobs = list_training_jobs()
    job_name = job_utils.get_job_folder_name_by_id(
        prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    for name, job in active_jobs.items():
        if job_name == name:
            logger.info(f"Found active training job with ID {job_id}")
            job.terminate()
            return True, f"Successfully stopped training job with ID  {job_id}"
    return False, f"Failed to find active training job with ID {job_id}"


def partition_dataset(
        server_dir: str,
        job_id: str,
        min_samples: int,
        data_ratios: List,
        num_folds: int,
        seed: int = 42) -> Tuple[bool, Dict, str]:
    dataset_path = job_utils.get_dataset_file_path(
        server_dir, prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    if not os.path.isfile(dataset_path):
        return False, {}, f"dataset file {dataset_path} not exists"
    try:
        resource_dir = job_utils.get_job_output_dir(
            server_dir, prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
        partition_dir = dataset_utils.get_partitioned_data_folder(resource_dir)
        if not os.path.exists(partition_dir):
            os.mkdir(partition_dir)

        result = poseidon_data.partition_poseidon_dataset(
            dataset_path=dataset_path,
            output_dir=partition_dir,
            min_samples=min_samples,
            data_ratios=data_ratios,
            num_folds=num_folds,
            seed=seed)

        if not result["valid_tags"]:
            message = "No valid tags in dataset"
            logger.warn(message)
            raise ValueError(message)
    except ValueError as e:
        return False, {}, f"failed to validate and partition dataset. Error: {e}"
    return True, result, ""


def get_report(server_dir: str, job_id: str, include_data: bool = True) -> str:
    output_dir = job_utils.get_job_output_dir(
        server_dir, prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    report_dir = job_utils.get_report_output_dir(output_dir)
    if not os.path.isdir(report_dir):
        return {}
    output = job_utils.load_model_report(report_dir, include_data)
    return output


def cleanup_artifacts(server_dir: str, job_id: str) -> Tuple[bool, str]:
    if is_job_running(job_id=job_id):
        return False, f"can no delete a running job_id {job_id}."
    return job_utils.cleanup_artifacts(
        server_dir, prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)


def get_status(server_dir: str, job_id: str) -> JobStatus:
    if is_job_running(job_id=job_id):
        return JobStatus.running
    else:
        output_dir = job_utils.get_job_output_dir(server_dir,
                                                  prefix=job_utils.TRAINING_JOB_PREFIX,
                                                  job_id=job_id)
        log_path = job_utils.get_log_path(
            output_dir, job_type=JobType.training)
        if os.path.exists(output_dir) and os.path.isfile(log_path):
            # report generation marks training is at least completed
            report_dir = job_utils.get_report_output_dir(output_dir)
            if os.path.isdir(report_dir):
                # check if packing is done
                if packager.done_packaging(output_dir):
                    return JobStatus.packaged
                return JobStatus.completed
            else:
                return JobStatus.terminated
        else:
            return JobStatus.not_started


def get_job_progress(server_dir: str, job_id: str) -> float:
    if is_job_running(job_id=job_id):
        output_dir = job_utils.get_job_output_dir(server_dir,
                                                  prefix=job_utils.TRAINING_JOB_PREFIX,
                                                  job_id=job_id)
        loss_file_path = job_utils.get_loss_file_path(output_dir)
        if os.path.exists(loss_file_path):
            with open(job_utils.get_dataset_profile_path(output_dir)) as data_profile_f:
                dataset_profile = json.load(data_profile_f)
            with open(job_utils.get_training_args_file_path(output_dir)) as training_args_f:
                training_args = json.load(training_args_f)
            count = 0
            with open(loss_file_path) as loss_f:
                for _, _ in enumerate(loss_f):
                    count += 1
            total_num_entries = dataset_profile["num_examples"]["train"]
            num_batches = (total_num_entries *
                           training_args["num_train_epochs"] // training_args["per_gpu_train_batch_size"]) + 1
            progress = 1.0 * count / num_batches
            logger.info(f"job progress is {progress}")
            return progress
        else:
            return 0.0
    else:
        return None


def is_job_running(job_id: str) -> bool:
    job_name = job_utils.get_job_folder_name_by_id(
        prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    return job_name in list_training_jobs().keys()


def list_training_jobs() -> Dict[str, mp.Process]:
    return job_utils.list_running_jobs(prefix=job_utils.TRAINING_JOB_PREFIX)


def list_all_training_jobs(server_dir: str) -> Dict[str, str]:
    return job_utils.list_all_jobs(server_dir, prefix=job_utils.TRAINING_JOB_PREFIX)


def build_model_package(
        job_id: str,
        server_dir: str) -> Tuple[bool, str]:
    status = get_status(server_dir=server_dir, job_id=job_id)
    output_dir = job_utils.get_job_output_dir(server_dir,
                                              prefix=job_utils.TRAINING_JOB_PREFIX,
                                              job_id=job_id)
    if status != JobStatus.packaged:
        if status != JobStatus.completed:
            return False, "", f"Job must be completed to create model package. Current status: {status}"
        # TODO: Fix hard coded eval_max_seq_length=128
        pkger = packager.ModelPackager(
            model_dir=output_dir,
            eval_max_seq_length=128,
        )
        package_dir = pkger.build_model_package()
    else:
        package_dir = packager.get_package_dir(output_dir)
    return True, package_dir, f"Your model package is created at {package_dir}.\
                   You can copy the model package torchserve and create *.mar file by running\
                   \"sh package.sh model_name model_version\""


def download_model_package(
        job_id: str,
        server_dir: str) -> Tuple[bool, str]:
    status = get_status(server_dir=server_dir, job_id=job_id)
    output_dir = job_utils.get_job_output_dir(
        server_dir, prefix=job_utils.TRAINING_JOB_PREFIX, job_id=job_id)
    if status != JobStatus.packaged:
        raise ValueError(
            f"model is not yet packaged, Current status: {status}")
    package_dir = packager.get_package_dir(output_dir)
    package_zip_path = os.path.join(output_dir, "package.zip")
    common.zipdir(dir_to_zip=package_dir, output_path=package_zip_path)
    logger.info(f"created zipped package at {package_zip_path}")
    return package_zip_path
