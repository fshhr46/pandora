#!/bin/bash
set -e

# docker run -d -it --mount type=bind,source=$HOME/workspace,target=/app/workspace -p 18080:8080 -p 18081:8081 --name torchserve pytorch/torchserve:0.5.3-gpu
# docker exec -it --user root torchserve bash

version=1
model_name=synthetic

curl "http://localhost:18081/models"

# bash register.sh $model_name $version
docker exec -it --user root torchserve bash -c "cd /app/workspace/resource/outputs/bert-base-chinese/synthetic_data/torchserve_package && bash register.sh $model_name $version"


# curl -X POST  "http://localhost:18081/models?url=$model_name-$version.mar&batch_size=128"
curl -v -X PUT "http://localhost:18081/models/$model_name?&min_worker=1&synchronous=true"


curl -d "data=hello world" "http://localhost:18080/predictions/$model_name/$version"