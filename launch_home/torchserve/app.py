import argparse
import subprocess

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/command", methods=['POST'])
def run_command():
    cmd = request.data.decode("UTF-8")
    print("exec command: ", cmd)
    rst = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return jsonify({
        "code": rst.returncode,
        "stdout": str(rst.stdout, encoding="UTF-8"),
        "stderr": str(rst.stderr, encoding="UTF-8"),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, required=True)
    parser.add_argument("--host", type=str, required=True)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)

