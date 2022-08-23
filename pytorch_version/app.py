from flask import Flask, jsonify, request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/status', methods=['GET'])
def get_training_status():
    job_id = request.args.get('id')
    print(f"job id is {job_id}")
    output = {
        "running": True
    }
    return jsonify(output)


if __name__ == '__main__':
    app.run()
