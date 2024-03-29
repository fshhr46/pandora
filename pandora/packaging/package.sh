#!/bin/bash
set -e

Usage() {
  echo 'build model archive with model name and model verison.

Example:
    build only:  package.sh model_name 1.0'
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
