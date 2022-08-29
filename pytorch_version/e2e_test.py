import multiprocessing
import os
from pathlib import Path
import time
import unittest
import requests

import app
import json
import psutil
import os

from server.training_job import JobStatus

HOST = "127.0.0.1"
PORT = "36666"


def kill_proc_tree(pid):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()


def start_test_server():
    home = str(Path.home())
    arg_list = [
        f"--host={HOST}",
        f"--port={PORT}",
        "--log_level=DEBUG",
        f"--data_dir={home}/workspace/resource/datasets/sentence",
        f"--output_dir={home}/pandora_outputs",
        f"--cache_dir={home}/.cache/torch/transformers",
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
    return f"http://{HOST}:{PORT}"


class TestTrainingJobE2E(unittest.TestCase):
    def test_training_failed(self):
        try:
            # start server
            server_process = multiprocessing.Process(target=start_test_server)
            server_process.start()
            print("waiting for server to be ready")
            time.sleep(3)

            # generate job ID
            job_id = time.time_ns()
            print(f"test Job ID is {job_id}")

            # list jobs, make sure it is empty
            assert make_request(
                f"{get_url()}/list?running=true") == []

            # start training job
            assert make_request(
                f"{get_url()}/start?id={job_id}", post=True)["success"]

            # ensure training job is running
            assert make_request(
                f"{get_url()}/list?running=true") == [f"PANDORA_TRAINING_{job_id}"]

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

            # list jobs, make sure it is empty
            assert make_request(
                f"{get_url()}/list?running=true") == []

            # list jobs, make sure it is empty
            assert make_request(
                f"{get_url()}/list?running=false") == [f"PANDORA_TRAINING_{job_id}"]

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

            # start training the same job again and now succeeded.
            assert make_request(
                f"{get_url()}/start?id={job_id}", post=True)["success"]

            # cleanup
            make_request(f"{get_url()}/cleanup?id={job_id}",
                         post=True)["success"]

        finally:
            print("start killing processes")
            kill_proc_tree(os.getpid())
            print("done killing processes")


if __name__ == '__main__':
    TestTrainingJobE2E().test_training_failed()
