from importlib.resources import path
import logging
import argparse
import os
import shutil
import json
import pathlib
import traceback

from pandora.packaging.losses import LossType
from pandora.service.job_utils import DATASET_FILE_NAME, JobType

from pandora.tools.common import logger
import pandora.service.training_job as training_job
import pandora.service.keywords_job as keywords_job
import pandora.service.server as server
import pandora.dataset.dataset_utils as dataset_utils
import pandora.service.job_utils as job_utils


from flask import (
    Flask,
    jsonify,
    request,
    send_file
)

flaskApp = Flask("Pandora")
flaskApp.config['JSON_AS_ASCII'] = False
server = server.Server(flaskApp)


def _get_job_id(args) -> str:
    job_id = args.get('id')
    if job_id:
        logger.info(f"job id is {job_id}")
        return job_id
    raise ValueError(f"invalid Job ID: {job_id}")


def _get_job_type(args) -> str:
    # TODO: remove default value
    job_type = args.get("job_type", default=JobType.training, type=str)
    if job_type and hasattr(JobType, job_type):
        logger.info(f"job_type is {job_type}")
        return job_type
    raise ValueError(f"invalid job_type {job_type}")


def _get_job_module(job_type: str):
    if job_type == JobType.training:
        return training_job
    elif job_type == JobType.keywords:
        return keywords_job
    else:
        raise ValueError(f"invalid job_type {job_type}")


# ================== shared APIs ==================


@flaskApp.route('/ingest-dataset', methods=['POST'])
def ingest_dataset():
    job_type = _get_job_type(request.args)
    job_id = _get_job_id(args=request.args)
    prefix = JobType.get_job_prefix(job_type=job_type)
    try:
        job_output_dir = job_utils.get_job_output_dir(
            server.output_dir, prefix=prefix, job_id=job_id)
        file_dir = pathlib.Path(job_output_dir)
        file_dir.mkdir(parents=True, exist_ok=True)
        dataset_json = request.get_json()
        dataset_path = os.path.join(job_output_dir, DATASET_FILE_NAME)
        if not dataset_json:
            raise ValueError("invalid input json")
        with open(dataset_path, 'w') as f:
            json.dump(dataset_json, f, ensure_ascii=False)
    except Exception as e:
        return {
            "success": False,
            "message": traceback.format_exc(),
            "dataset_path": None
        }
    return {
        "success": True,
        "message": f"ingested dataset to {dataset_path}",
        "dataset_path": dataset_path
    }


@flaskApp.route('/list', methods=['GET'])
def list_jobs():
    job_type = _get_job_type(request.args)
    jobs = {}
    job_prefix = JobType.get_job_prefix(job_type)
    running = request.args.get("running", default="", type=str)
    if running:
        if running.lower() == "true":
            jobs = job_utils.list_running_jobs(prefix=job_prefix)
        elif running.lower() == "false":
            jobs = job_utils.list_all_jobs(
                server.output_dir, prefix=job_prefix)
    return jsonify([job_name for job_name in jobs.keys()])


@flaskApp.route('/testdata', methods=['POST'])
def get_output_path():
    file_name = request.args.get('file_name')
    job_type = _get_job_type(request.args)
    job_id = _get_job_id(args=request.args)
    job_prefix = JobType.get_job_prefix(job_type)
    try:
        job_output_dir = job_utils.get_job_output_dir(
            server.output_dir, job_prefix, job_id)
        os.mkdir(job_output_dir)
        shutil.copyfile(
            os.path.join("test_data", file_name),
            os.path.join(job_output_dir, DATASET_FILE_NAME))
    except Exception as e:
        return {
            "success": False,
            "message": traceback.format_exc()
        }
    return {
        "success": True,
        "message": ""
    }


@flaskApp.route('/stop', methods=['POST'])
def stop_job():
    job_type = _get_job_type(request.args)
    job_id = _get_job_id(args=request.args)
    success, message = _get_job_module(job_type).stop_job(job_id=job_id)
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


@flaskApp.route('/cleanup', methods=['POST'])
def cleanup_artifacts():
    job_type = _get_job_type(request.args)
    job_id = _get_job_id(args=request.args)
    success, message = _get_job_module(job_type).cleanup_artifacts(
        server_dir=server.output_dir,
        job_id=job_id,
    )
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


@flaskApp.route('/status', methods=['GET'])
def get_status():
    job_type = _get_job_type(request.args)
    job_id = _get_job_id(args=request.args)
    status = _get_job_module(job_type).get_status(
        server_dir=server.output_dir,
        job_id=job_id,
    )
    progress = _get_job_module(job_type).get_job_progress(
        server_dir=server.output_dir,
        job_id=job_id,
    )
    output = {
        "status": status,
        "progress": progress,
    }
    return jsonify(output)

# ================== training job related ==================


@flaskApp.route('/start', methods=['POST'])
def start_training():
    job_id = _get_job_id(args=request.args)

    sample_size = request.args.get("sample_size", default=0, type=int)
    logging.info(f"sample_size is {sample_size}")

    loss_type = request.args.get(
        "loss_type", default=LossType.focal_loss, type=str)
    logging.info(f"loss_type is {loss_type}")

    active_training_jobs = training_job.list_training_jobs()
    has_resource, msg = server.has_enough_resource(len(active_training_jobs))
    if not has_resource:
        return {
            "success": False,
            "message": msg
        }
    success, message = training_job.start_training_job(
        job_id=job_id,
        server_dir=server.output_dir,
        cache_dir=server.cache_dir,
        sample_size=sample_size,
        loss_type=loss_type)
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


@flaskApp.route('/partition', methods=['POST'])
def partition_dataset():
    job_id = _get_job_id(args=request.args)
    if request.data:
        json_data = request.get_json()
    else:
        json_data = {}
    logging.info(f"json input is {json_data}")
    min_samples = json_data.get("min_samples", 10)
    data_ratios = json_data.get(
        "data_ratios", {"train": 0.6, "dev": 0.2, "test": 0.2})
    logging.info(f"min_samples is {min_samples}")
    logging.info(f"data_ratios is {data_ratios}")

    # Validations
    error_msg = dataset_utils.validate_ratios(data_ratios)
    if error_msg:
        return {
            "success": False,
            "result": {},
            "message": error_msg,
        }

    if type(min_samples) != int or min_samples <= 0:
        return {
            "success": False,
            "result": {},
            "message": f"min_samples must be positive integers. got: {min_samples}",
        }

    success, result, message = training_job.partition_dataset(
        server_dir=server.output_dir,
        job_id=job_id,
        min_samples=min_samples,
        data_ratios=data_ratios,
    )
    output = {
        "success": success,
        "result": result,
        "message": message,
    }
    return jsonify(output)


@flaskApp.route('/report', methods=['GET'])
def get_model_report():
    job_id = _get_job_id(args=request.args)
    include_data = request.args.get("data", default="", type=str)
    if include_data:
        if include_data.lower() == "true":
            include_data = True
        elif include_data.lower() == "false":
            include_data = False
    logging.info(f"include_data is {include_data}")
    output = training_job.get_report(
        server_dir=server.output_dir,
        job_id=job_id,
        include_data=include_data)
    return jsonify(output)


@flaskApp.route('/package', methods=['POST'])
def start_packaging():
    job_id = _get_job_id(args=request.args)
    success, package_dir, message = training_job.build_model_package(
        job_id=job_id,
        server_dir=server.output_dir,
    )
    output = {
        "success": success,
        "message": message,
        "package_dir": package_dir,
    }
    return jsonify(output)


@flaskApp.route('/download', methods=['POST'])
def download_package():
    job_id = _get_job_id(args=request.args)
    package_zip_path = training_job.download_model_package(
        job_id=job_id,
        server_dir=server.output_dir,
    )
    return send_file(package_zip_path)


# ================== Keyword extraction APIs ==================


@flaskApp.route('/extract-keywords', methods=['POST'])
def extract_keywords():
    job_id = _get_job_id(args=request.args)
    model_name = request.args.get('model_name')
    model_version = request.args.get('model_version')

    if server.torch_serve_host and server.torch_serve_port:
        logger.info(f"torch_serve_host is {server.torch_serve_host}")
        logger.info(f"torch_serve_port is {server.torch_serve_port}")
    else:
        if not server.torch_serve_host:
            logger.error(
                f"server.torch_serve_host is missing. Please restart server with --torch_serve_host")
        if not server.torch_serve_port:
            logger.error(
                f"server.torch_serve_port is missing. Please restart server with --torch_serve_port")
        raise ValueError("failed to run extract keyword.")

    success, message, path = keywords_job.start_keyword_extraction_job(
        server_dir=server.output_dir,
        job_id=job_id,
        host=server.torch_serve_host,
        port=server.torch_serve_port,
        model_name=model_name,
        model_version=model_version
    )
    return {
        "success": success,
        "message": message,
        "path": path
    }


@flaskApp.route('/get-keywords', methods=['GET'])
def get_keywords():
    job_id = _get_job_id(args=request.args)
    output = keywords_job.get_keywords(
        server_dir=server.output_dir,
        job_id=job_id)
    return jsonify(output)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--max_jobs", type=int, default=1, required=False)
    parser.add_argument("--log_dir", type=str, default=None, required=False)
    parser.add_argument("--log_level", type=str,
                        default=logging.INFO, required=False, choices=logging._nameToLevel.keys())
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str,
                        default=None, required=False)
    parser.add_argument("--cache_dir", type=str,
                        default=None, required=False)
    parser.add_argument("--torch_serve_host", type=str,
                        default=None, required=False)
    parser.add_argument("--torch_serve_port", type=str,
                        default=None, required=False)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    server.run(args)
