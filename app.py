import logging
import argparse
import os
import shutil
import json
import pathlib
import traceback

from pandora.tools.common import logger
import pandora.service.training_job as training_job
import pandora.service.server as server
import pandora.dataset.dataset_utils as dataset_utils


from flask import (
    Flask,
    jsonify,
    request,
    send_file
)

flaskApp = Flask("Pandora")
flaskApp.config['JSON_AS_ASCII'] = False
server = server.Server(flaskApp)


@flaskApp.route('/start', methods=['POST'])
def start_training():
    job_id = _get_job_id(args=request.args)
    sample_size = request.args.get("sample_size", default=0, type=int)
    logging.info(f"sample_size is {sample_size}")
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
        sample_size=sample_size)
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


def _get_job_id(args) -> str:
    job_id = args.get('id')
    if job_id:
        logger.info(f"job id is {job_id}")
        return job_id
    raise ValueError(f"invalid Job ID: {job_id}")


@flaskApp.route('/stop', methods=['POST'])
def stop_training():
    job_id = _get_job_id(args=request.args)
    success, message = training_job.stop_training_job(job_id=job_id)
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


@flaskApp.route('/cleanup', methods=['POST'])
def cleanup_artifacts():
    job_id = _get_job_id(args=request.args)
    success, message = training_job.cleanup_artifacts(
        server_dir=server.output_dir,
        job_id=job_id,
    )
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


@flaskApp.route('/status', methods=['GET'])
def get_training_status():
    job_id = _get_job_id(args=request.args)
    status = training_job.get_training_status(
        server_dir=server.output_dir,
        job_id=job_id,
    )
    output = {
        "status": status
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


@flaskApp.route('/list', methods=['GET'])
def list_training_jobs():
    running = request.args.get("running", default="", type=str)
    jobs = {}
    if running:
        if running.lower() == "true":
            jobs = training_job.list_training_jobs()
        elif running.lower() == "false":
            jobs = training_job.list_all_jobs(server.output_dir)
    return jsonify([job_name for job_name in jobs.keys()])


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


@flaskApp.route('/testdata', methods=['POST'])
def get_output_path():
    job_id = _get_job_id(args=request.args)
    try:
        job_output_dir = training_job.get_job_output_dir(
            server.output_dir, job_id)
        os.mkdir(job_output_dir)
        dataset_file_name = "dataset.json"
        shutil.copyfile(
            os.path.join("test_data", dataset_file_name),
            os.path.join(job_output_dir, dataset_file_name))
    except Exception as e:
        return {
            "success": False,
            "message": traceback.format_exc()
        }
    return {
        "success": True,
        "message": ""
    }


@flaskApp.route('/ingest-dataset', methods=['POST'])
def ingest_dataset():
    job_id = _get_job_id(args=request.args)
    try:
        job_output_dir = training_job.get_job_output_dir(
            server.output_dir, job_id)
        file_dir = pathlib.Path(job_output_dir)
        file_dir.mkdir(parents=True, exist_ok=True)
        dataset_json = request.get_json()
        dataset_file_name = "dataset.json"
        dataset_path = os.path.join(job_output_dir, dataset_file_name)
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
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    server.run(args)
