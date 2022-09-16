#!/bin/bash
set -e

Usage() {
  echo 'build model archive with model name and model verison.

Example:
    build only:  register.sh model_name 1.0'
  exit 1
}

model_name=$1
model_version=$2

if [ "$model_name" == "" ]; then
  echo 'please provide model name'
  Usage
fi

if [ "$model_version" == "" ]; then
  echo 'please provide model version'
  Usage
fi

# create archive file
bash package.sh $model_name $model_version

# copy to model-store
versioned_name="${model_name}-${model_version}.mar"
cp $model_name.mar /home/model-server/model-store/$versioned_name

# register
curl -X POST "http://localhost:8081/models?url=${versioned_name}&batch_size=3"
