set -e

torchserve --start --ts-config ./torchserve/config.properties

python3 ./torchserve/app.py --host 0.0.0.0 --port 38083

