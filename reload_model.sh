#!/bin/bash
set -e

# docker run -d -it --mount type=bind,source=$HOME/workspace,target=/app/workspace --gpus all -p 38080:8080 -p 38081:8081 --name torchserve pytorch/torchserve:0.5.3-gpu
# docker exec -it --user root torchserve bash

# docker restart torchserve

# ==================== torchserve config
# inference_address=http://0.0.0.0:38080
# management_address=http://0.0.0.0:38081
# metrics_address=http://0.0.0.0:38082
# number_of_netty_threads=32
# job_queue_size=1000
model_store=/home/haoranhuang/torchserve/model-store
# workflow_store=/home/haoranhuang/torchserve/wf-store

# torchserve --stop && torchserve --start --ts-config /home/haoranhuang/torchserve/model-server/config.properties

version=1
model_name=short_sentence

curl -X DELETE "http://localhost:38081/models/$model_name/$version"

curl "http://localhost:38081/models"

# docker exec -it --user root torchserve bash -c "cd /app/workspace/resource/outputs/bert-base-chinese/synthetic_data/torchserve_package && bash register.sh $model_name $version"
cd /home/haoranhuang/workspace/resource/outputs/bert-base-chinese/short_sentence/torchserve_package && bash register.sh $model_name $version $model_store

curl -v -X PUT "http://localhost:38081/models/$model_name?&min_worker=1&synchronous=true"

curl -d "data=hello world" "http://localhost:38080/predictions/$model_name/$version"
