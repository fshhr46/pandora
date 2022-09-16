#!/bin/bash
set -e

# docker run -d -it --mount type=bind,source=$HOME/workspace,target=/app/workspace --gpus all -p 18080:8080 -p 18081:8081 --name torchserve pytorch/torchserve:0.5.3-gpu
# docker exec -it --user root torchserve bash

# docker restart torchserve

# ==================== torchserve config
# inference_address=http://0.0.0.0:18080
# management_address=http://0.0.0.0:18081
# metrics_address=http://0.0.0.0:18082
# number_of_netty_threads=32
# job_queue_size=1000
# model_store=/home/haoranhuang/torchserve/model-store
# workflow_store=/home/haoranhuang/torchserve/wf-store

torchserve --stop && torchserve --start --ts-config /home/haoranhuang/torchserve/model-server/config.properties

version=1
model_name=synthetic

curl "http://localhost:18081/models"

# docker exec -it --user root torchserve bash -c "cd /app/workspace/resource/outputs/bert-base-chinese/synthetic_data/torchserve_package && bash register.sh $model_name $version"
cd /home/haoranhuang/workspace/resource/outputs/bert-base-chinese/synthetic_data/torchserve_package && bash register.sh $model_name $version

curl -v -X PUT http://localhost:18081/models/$model_name/$version/set-default

curl -v -X PUT "http://localhost:18081/models/$model_name?&min_worker=1&synchronous=true"

curl -d "data=hello world" "http://localhost:18080/predictions/$model_name/$version"