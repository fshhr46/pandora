#!/bin/bash
set -e

python3 e2e_test.py --host=5y72098x40.zicp.fun --port=38888 --management_port=38081 --command_port=38083 --model_store=/home/haoranhuang/torchserve/model-store --torchserve_host=5y72098x40.zicp.fun
