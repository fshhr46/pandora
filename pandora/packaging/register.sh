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

model_store_path=$3
if [ "$model_store_path" != "" ]; then
  echo "overriding model_store_path to ${model_store_path}"
else
  model_store_path="/home/model-server/model-store"
  echo "using default model_store_path ${model_store_path}"
fi

# create archive file
bash package.sh $model_name $model_version

# copy to model-store
versioned_name="${model_name}-${model_version}.mar"
cp $model_name.mar $model_store_path/$versioned_name

# register
curl -X POST "http://localhost:8081/models?url=${versioned_name}&batch_size=3"
