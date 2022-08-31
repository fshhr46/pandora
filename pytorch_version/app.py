import logging
import os
import argparse
from pathlib import Path

from tools.common import init_logger, logger
import server.training_job as training_job

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)


class Server(object):
    def __init__(self, flaskApp) -> None:
        self.flaskApp = flaskApp

    def run(self, args) -> None:
        # dirs
        self.output_dir = args.output_dir

        # default data dir log will be in $HOME/workspace/resource/datasets.
        if args.data_dir:
            self.data_dir = args.data_dir
        else:
            home = str(Path.home())
            self.data_dir = os.path.join(home, "workspace/resource/datasets")

        # default service log will be in the output_dir.
        if args.cache_dir:
            self.cache_dir = args.cache_dir
        else:
            home = str(Path.home())
            self.cache_dir = os.path.join(home, ".cache/torch/transformers")

        # default service log will be in the output_dir.
        if args.log_dir:
            self.log_path = os.path.join(args.log_dir, "service_log.txt")
        else:
            self.log_path = os.path.join(self.output_dir, "service_log.txt")

        # logs
        if args.log_level:
            log_level = logging.getLevelName(args.log_level)
        init_logger(log_file=self.log_path, log_file_level=log_level)

        # run
        self.flaskApp.run(host=args.host, port=args.port)


flaskApp = Flask("Pandora")
server = Server(flaskApp)


@flaskApp.route('/start', methods=['POST'])
def start_training():
    job_id = request.args.get('id')
    logging.info(f"job id is {job_id}")
    sample_size = request.args.get("sample_size", default=0, type=int)
    logging.info(f"sample_size is {sample_size}")
    success, message = training_job.start_training_job(
        job_id=job_id,
        server_dir=server.output_dir,
        data_dir=server.data_dir,
        cache_dir=server.cache_dir,
        sample_size=sample_size)
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


@flaskApp.route('/stop', methods=['POST'])
def stop_training():
    job_id = request.args.get('id')
    logging.info(f"job id is {job_id}")
    success, message = training_job.stop_training_job(job_id=job_id)
    output = {
        "success": success,
        "message": message
    }
    return jsonify(output)


@flaskApp.route('/cleanup', methods=['POST'])
def cleanup_artifacts():
    job_id = request.args.get('id')
    logging.info(f"job id is {job_id}")
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
    job_id = request.args.get('id')
    logging.info(f"job id is {job_id}")
    status = training_job.get_training_status(
        server_dir=server.output_dir,
        job_id=job_id,
    )
    output = {
        "status": status
    }
    return jsonify(output)


@flaskApp.route('/report', methods=['GET'])
def get_model_report():
    job_id = request.args.get('id')
    logging.info(f"job id is {job_id}")
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


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True)
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=None, required=False)
    parser.add_argument("--log_level", type=str,
                        default=logging.INFO, required=False, choices=logging._nameToLevel.keys())
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None, required=False)
    parser.add_argument("--cache_dir", type=str, default=None, required=False)
    return parser


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    # python3 app.py --host=0.0.0.0 --port=38888 --log_level=DEBUG --log_dir=$HOME/pandora_outputs --output_dir=$HOME/pandora_outputs --data_dir=$HOME/workspace/resource/datasets --cache_dir=$HOME/.cache/torch/transformers
    server.run(args)
