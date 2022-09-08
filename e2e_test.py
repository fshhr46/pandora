import multiprocessing
import os
import time
import requests

import app
import json
import psutil
import os
import argparse
import tempfile
import shutil

from pandora.service.training_job import JobStatus, get_job_output_dir

TEST_HOST = "127.0.0.1"
TEST_PORT = "36666"


def kill_proc_tree(pid):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()


def start_test_server(host, port, temp_dir):
    # home = str(Path.home())
    output_dir = os.path.join(temp_dir, "pandora_outputs")
    os.mkdir(output_dir)
    arg_list = [
        f"--host={host}",
        f"--port={port}",
        "--log_level=DEBUG",
        f"--log_dir={temp_dir}",
        f"--output_dir={temp_dir}",
        # f"--data_dir={home}/workspace/resource/datasets/sentence",
        # f"--cache_dir={home}/.cache/torch/transformers",
    ]
    parser = app.get_arg_parser()
    args = parser.parse_args(arg_list)
    app.server.run(args)


def make_request(url: str, post: bool = False):
    if post:
        url_obj = requests.post(url)
    else:
        url_obj = requests.get(url)
    text = url_obj.text
    print(text)
    data = json.loads(text)
    return data


def get_url():
    return f"http://{TEST_HOST}:{TEST_PORT}"


def test_training_failed(tmpdirname: str):
    # generate job ID
    job_id = time.time_ns()
    print(f"test Job ID is {job_id}")

    # prepare datadir
    job_output_dir = get_job_output_dir(tmpdirname, job_id)
    os.mkdir(job_output_dir)
    dataset_file_name = "dataset.json"
    shutil.copyfile(
        os.path.join("test_data", dataset_file_name),
        os.path.join(job_output_dir, dataset_file_name))
    assert make_request(
        f"{get_url()}/partition?id={job_id}", post=True)["success"]

    # start job
    assert make_request(
        f"{get_url()}/start?id={job_id}", post=True)["success"]

    # ensure training job is running
    assert f"PANDORA_TRAINING_{job_id}" in make_request(
        f"{get_url()}/list?running=true")

    # delete artifacts will fail when job is running
    assert not make_request(
        f"{get_url()}/cleanup?id={job_id}", post=True)["success"]

    # ensure job can be stated when server has enough resource.
    assert not make_request(
        f"{get_url()}/start?id={job_id}", post=True)["success"]
    print("waiting for job to be started")
    time.sleep(10)

    # check job status
    assert make_request(
        f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.running

    # stop running job
    assert make_request(
        f"{get_url()}/stop?id={job_id}", post=True)["success"]

    # Check job status changed from running to terminated.
    fails = 0
    while make_request(
            f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.running:
        time.sleep(1)
        fails += 1
        assert fails < 10
    assert make_request(
        f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.terminated

    # Check job is successfully stopped
    assert not make_request(
        f"{get_url()}/stop?id={job_id}", post=True)["success"]

    # list jobs, make sure the job is in the list
    assert f"PANDORA_TRAINING_{job_id}" in make_request(
        f"{get_url()}/list?running=false")

    # start training the same job but failed.
    assert not make_request(
        f"{get_url()}/start?id={job_id}", post=True)["success"]

    # get report should return nothing
    output = make_request(
        f"{get_url()}/report?id={job_id}&data=True", post=False)
    assert output.get("paths") == None
    assert output.get("data") == None

    # delete artifacts
    assert make_request(
        f"{get_url()}/cleanup?id={job_id}", post=True)["success"]

    # check job status and should be changed to not_started
    assert make_request(
        f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.not_started


def test_training_success(tmpdirname: str):
    sample_size = 10
    # generate job ID
    job_id = time.time_ns()
    print(f"test Job ID is {job_id}")

    # prepare datadir
    job_output_dir = get_job_output_dir(tmpdirname, job_id)
    os.mkdir(job_output_dir)
    dataset_file_name = "dataset.json"
    shutil.copyfile(
        os.path.join("test_data", dataset_file_name),
        os.path.join(job_output_dir, dataset_file_name))
    assert make_request(
        f"{get_url()}/partition?id={job_id}", post=True)["success"]

    # start job
    assert make_request(
        f"{get_url()}/start?id={job_id}&sample_size={sample_size}", post=True)["success"]
    print("waiting for job to be started")

    # Check job status changed from running to completed.
    checks = 0
    interval = 5
    max_checks = 12 * 10
    while make_request(
            f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.running:
        time.sleep(interval)
        checks += 1
        assert checks < max_checks, "timeout, job is not completed"
    assert make_request(
        f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.completed,\
        "job did not complete successfully"
    output = make_request(
        f"{get_url()}/report?id={job_id}&data=True", post=False)
    paths = output.get("paths")
    data = output.get("data")
    assert len(paths) == 2
    assert len(data) == 2

    # run packaging
    assert make_request(
        f"{get_url()}/package?id={job_id}", post=True)["success"]
    assert make_request(
        f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.packaged

    # delete artifacts
    assert make_request(
        f"{get_url()}/cleanup?id={job_id}", post=True)["success"]

    # check job status and should be changed to not_started
    assert make_request(
        f"{get_url()}/status?id={job_id}", post=False)["status"] == JobStatus.not_started


# python3 app.py --host=0.0.0.0 --port=38888 --log_level=DEBUG --log_dir=$HOME/pandora_outputs --output_dir=$HOME/pandora_outputs --data_dir=$HOME/workspace/resource/datasets/sentence --cache_dir=$HOME/.cache/torch/transformers
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None, required=False)
    parser.add_argument("--port", type=str, default=None, required=False)
    parser.add_argument("--local_server", action="store_true",
                        help="Whether to start a .")
    args = parser.parse_args()

    if args.host:
        TEST_HOST = args.host
    if args.port:
        TEST_PORT = args.port

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            if args.local_server:
                # start server
                server_process = multiprocessing.Process(
                    target=start_test_server, args=(TEST_HOST, TEST_PORT, tmpdirname))
                server_process.start()
                print("waiting for server to be ready")
                time.sleep(3)
            test_training_failed(tmpdirname)
            test_training_success(tmpdirname)
        finally:
            print("start killing processes")
            kill_proc_tree(os.getpid())
            print("done killing processes")
