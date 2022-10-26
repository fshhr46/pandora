import multiprocessing
import os
import time
import requests
import logging

import app
import json
import psutil
import os
import argparse
import tempfile
import shutil
from pandora.packaging.feature import TrainingType
from pandora.service.job_utils import JobType

from pandora.service.training_job import JobStatus

TEST_HOST = "127.0.0.1"
TEST_PORT = "8888"

# Torchserve related configs
TORCHSERVE_HOST = "127.0.0.1"
MODEL_STORE = "/home/model-server/model-store"
MANAGEMENT_PORT = "8081"
COMMAND_PORT = "8083"


def kill_proc_tree(pid):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()


def start_test_server(host, port, output_dir):
    # home = str(Path.home())
    arg_list = [
        f"--host={host}",
        f"--port={port}",
        "--log_level=DEBUG",
        f"--log_dir={output_dir}",
        f"--output_dir={output_dir}",
        # f"--data_dir={home}/workspace/resource/datasets/sentence",
        # f"--cache_dir={home}/.cache/torch/transformers",
    ]
    parser = app.get_arg_parser()
    args = parser.parse_args(arg_list)
    app.server.run(args)


def make_request(url: str, post: bool = False, json_data=None, data=None, json_output=True):
    if post:
        if json_data:
            url_obj = requests.post(url, json=json_data)
        elif data:
            url_obj = requests.post(url, data=data)
        else:
            url_obj = requests.post(url)
    else:
        url_obj = requests.get(url)

    logging.info(f"status code: {url_obj.status_code}")
    logging.info(f"content: {url_obj.text}")
    if json_output:
        data = url_obj.text
        data = json.loads(data)
    else:
        data = url_obj
        if url_obj.status_code != 200:
            raise ValueError(url_obj)
    return data


def get_url():
    return f"http://{TEST_HOST}:{TEST_PORT}"


def get_command_url():
    return f"http://{TORCHSERVE_HOST}:{COMMAND_PORT}"


def get_management_url():
    return f"http://{TORCHSERVE_HOST}:{MANAGEMENT_PORT}"


def prepare_job_data(training_type: str, job_id, job_type, file_path=None):
    if file_path:
        with open(file_path) as f:
            dataset = json.load(f)
            assert make_request(
                f"{get_url()}/ingest-dataset?id={job_id}&job_type={job_type}", post=True, json_data=dataset)["success"]
    else:
        assert make_request(
            f"{get_url()}/testdata?id={job_id}&job_type={job_type}&file_name={training_type}.json", post=True)["success"]


def test_training_failed(training_type):

    job_type = JobType.training
    # generate job ID
    job_id = time.time_ns()
    logging.info(f"test Job ID is {job_id}")

    # start job witout data
    assert not make_request(
        f"{get_url()}/start?id={job_id}&training_type={training_type}", post=True)["success"]

    # prepare datadir
    prepare_job_data(training_type=training_type,
                     job_id=job_id, job_type=job_type)
    assert make_request(
        f"{get_url()}/partition?id={job_id}", post=True)["success"]

    # start job
    assert make_request(
        f"{get_url()}/start?id={job_id}&training_type={training_type}", post=True)["success"]

    # ensure training job is running
    assert f"PANDORA_TRAINING_{job_id}" in make_request(
        f"{get_url()}/list?running=true&job_type={job_type}")

    # delete artifacts will fail when job is running
    assert not make_request(
        f"{get_url()}/cleanup?id={job_id}&job_type={job_type}", post=True)["success"]

    # ensure job can be stated when server has enough resource.
    assert not make_request(
        f"{get_url()}/start?id={job_id}&training_type={training_type}", post=True)["success"]
    logging.info("waiting for job to be started")
    time.sleep(10)

    # check job status
    assert make_request(
        f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.running

    # stop running job
    assert make_request(
        f"{get_url()}/stop?id={job_id}&job_type={job_type}", post=True)["success"]

    # Check job status changed from running to terminated.
    fails = 0
    while make_request(
            f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.running:
        time.sleep(1)
        fails += 1
        assert fails < 10
    assert make_request(
        f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.terminated

    # Check job is successfully stopped
    assert not make_request(
        f"{get_url()}/stop?id={job_id}&job_type={job_type}", post=True)["success"]

    # list jobs, make sure the job is in the list
    assert f"PANDORA_TRAINING_{job_id}" in make_request(
        f"{get_url()}/list?running=false&job_type={job_type}")

    # start training the same job but failed.
    assert not make_request(
        f"{get_url()}/start?id={job_id}&training_type={training_type}", post=True)["success"]

    # get report should return nothing
    output = make_request(
        f"{get_url()}/report?id={job_id}&data=True", post=False)
    assert output.get("paths") == None
    assert output.get("data") == None

    # delete artifacts
    assert make_request(
        f"{get_url()}/cleanup?id={job_id}&job_type={job_type}", post=True)["success"]

    # check job status and should be changed to not_started
    assert make_request(
        f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.not_started


def test_training_success(training_type: str, test_keyword: bool = False):
    sample_size = 10
    if training_type == TrainingType.meta_data:
        sample_size = 0

    # generate job ID
    job_id = time.time_ns()
    logging.info(f"test Job ID is {job_id}")
    job_type = JobType.training

    # prepare datadir
    prepare_job_data(training_type=training_type, job_id=job_id, job_type=job_type, file_path=os.path.join(
        "test_data",  f"{training_type}.json"))
    assert make_request(
        f"{get_url()}/partition?id={job_id}", post=True)["success"]

    # start job
    assert make_request(
        f"{get_url()}/start?id={job_id}&training_type={training_type}&sample_size={sample_size}", post=True)["success"]
    logging.info("waiting for job to be started")

    # Check job status changed from running to completed.
    checks = 0
    interval = 5
    max_checks = 12 * 10
    while make_request(
            f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.running:
        time.sleep(interval)
        checks += 1
        assert checks < max_checks, "timeout, job is not completed"

    assert make_request(
        f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.completed,\
        "job did not complete successfully"

    output = make_request(
        f"{get_url()}/report?id={job_id}&data=True", post=False)
    paths = output.get("paths")
    data = output.get("data")
    assert len(paths) == 2
    assert len(data) == 2

    # run packaging
    package_output = make_request(
        f"{get_url()}/package?id={job_id}", post=True)
    assert package_output["success"]

    assert make_request(
        f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.packaged

    if test_keyword:

        # publish model
        package_path = package_output["package_dir"]
        command = f"cd {package_path} && sh register.sh {job_id} 0 {MODEL_STORE}"
        logging.info(f"command is {command}")
        make_request(
            f"{get_command_url()}/command", post=True, data=command, json_output=False)

        logging.info(f"scaling up worker")
        # assign worker
        scale_up_response = requests.put(
            f"{get_management_url()}/models/{job_id}?&min_worker=1&synchronous=true")
        if scale_up_response.status_code != 200:
            raise ValueError(scale_up_response.text)

        # Check job status changed from running to completed.
        checks = 0
        interval = 5
        max_checks = 12 * 10
        num_workers = 0
        while num_workers == 0:
            model_response = make_request(
                f"{get_management_url()}/models/{job_id}", post=False)
            num_workers = len(model_response[0]["workers"])
            logging.info(f"num_workers is {num_workers}")
            time.sleep(interval)
            checks += 1
            assert checks < max_checks, "timeout, job is not completed"

        # Run test
        test_keywords(training_type, job_id)

        # Delete
        delete_resp = requests.delete(
            f"{get_management_url()}/models/{job_id}")
        if delete_resp.status_code != 200:
            raise ValueError(delete_resp.text)

    # delete artifacts
    assert make_request(
        f"{get_url()}/cleanup?id={job_id}&job_type={job_type}", post=True)["success"]

    # check job status and should be changed to not_started
    assert make_request(
        f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.not_started


def test_keywords(training_type: str, model_name: str):
    sample_size = 10
    # generate job ID
    job_id = time.time_ns()
    logging.info(f"test Job ID is {job_id}")
    job_type = JobType.keywords

    # start job witout data
    assert not make_request(
        f"{get_url()}/extract-keywords?id={job_id}", post=True)["success"]

    # prepare datadir
    prepare_job_data(training_type=training_type, job_id=job_id, job_type=job_type, file_path=os.path.join(
        "test_data",  f"{training_type}.json"))

    # start job
    assert make_request(
        f"{get_url()}/extract-keywords?id={job_id}&model_name={model_name}", post=True)["success"]

    # Check job status changed from running to completed.
    checks = 0
    interval = 5
    max_checks = 12 * 10
    while make_request(
            f"{get_url()}/status?id={job_id}&job_type={job_type}", post=False)["status"] == JobStatus.running:
        time.sleep(interval)
        checks += 1
        assert checks < max_checks, "timeout, job is not completed"

    make_request(
        f"{get_url()}/get-keywords?id={job_id}", post=False)


# python3 app.py --host=0.0.0.0 --port=38888 --log_level=DEBUG --log_dir=$HOME/pandora_outputs --output_dir=$HOME/pandora_outputs --data_dir=$HOME/workspace/resource/datasets/sentence --cache_dir=$HOME/.cache/torch/transformers
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None, required=False)
    parser.add_argument("--port", type=str, default=None, required=False)
    parser.add_argument("--local_server", action="store_true",
                        help="Whether to start a .")

    # Torchserve related configs
    parser.add_argument("--torchserve_host", type=str,
                        default=None, required=False)
    parser.add_argument("--model_store", type=str,
                        default=None, required=False)
    parser.add_argument("--management_port", type=str,
                        default=None, required=False)
    parser.add_argument("--command_port", type=str,
                        default=None, required=False)
    args = parser.parse_args()

    if args.host:
        TEST_HOST = args.host
    if args.port:
        TEST_PORT = args.port

    # Torchserve related configs
    if args.torchserve_host:
        TORCHSERVE_HOST = args.torchserve_host
    if args.model_store:
        MODEL_STORE = args.model_store
    if args.management_port:
        MANAGEMENT_PORT = args.management_port
    if args.command_port:
        COMMAND_PORT = args.command_port

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            if args.local_server:
                output_dir = tmpdirname
                # start server
                server_process = multiprocessing.Process(
                    target=start_test_server, args=(TEST_HOST, TEST_PORT, output_dir))
                server_process.start()
                logging.info("waiting for server to be ready")
                time.sleep(3)
            # test_training_failed(training_type=TrainingType.mixed_data)
            # test_training_success(training_type=TrainingType.mixed_data)
            test_training_success(
                training_type=TrainingType.meta_data, test_keyword=True)
            # test_keywords(training_type=TrainingType.meta_data)
        finally:
            logging.info("start killing processes")
            kill_proc_tree(os.getpid())
            logging.info("done killing processes")
