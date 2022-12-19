#!/bin/bash
set -e

cd $HOME/workspace/pandora

python3 app.py --host=0.0.0.0 --port=38888 --log_level=DEBUG --log_dir=$HOME/workspace/pandora_outputs --output_dir=$HOME/workspace/pandora_outputs --data_dir=$HOME/workspace/resource/datasets --cache_dir=$HOME/.cache/torch/transformers --torch_serve_host=0.0.0.0 --torch_serve_port=38080
