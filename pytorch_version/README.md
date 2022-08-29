### 数据介绍

数据详细描述: https://www.cluebenchmarks.com/introduce.html

### 运行方式
1. 下载CLUE_NER数据集，运行以下命令：
```shell
python tools/download_clue_data.py --data_dir=./datasets --tasks=cluener
```
2. 预训练模型文件格式，比如:
```text
├── prev_trained_model　# 预训练模型
|  └── bert-base
|  | └── vocab.txt
|  | └── config.json
|  | └── pytorch_model.bin
```
3. 训练/评估/预测

直cd到pytorch_version目录下，接执行runner.py脚本，如：
```python
python runner.py
```
4. 启动Flask服务

当前默认使用最后一个checkpoint模型作为预测模型，你也可以指定--predict_checkpoints参数进行对应的checkpoint进行预测，比如：
```python
python app.py --host=0.0.0.0 --port=38888 --log_level=DEBUG --output_dir=$HOME/pandora_outputs --data_dir=$HOME/workspace/resource/datasets/sentence --cache_dir=$HOME/.cache/torch/transformers
```

5. 跑e2e测试
```python
python e2e_test.py
```

6. Flaks服务的API
    # list all running jobs
    curl -XGET http://127.0.0.1:38888/list\?running=true
     # list all jobs
    curl -XGET http://127.0.0.1:38888/list\?running=false
    # start a new job
    curl -XPOST http://127.0.0.1:38888/start\?id=1135
    # get job status
    curl -XGET http://127.0.0.1:38888/status\?id=1135
    # stop a running job
    curl -XPOST http://127.0.0.1:38888/stop\?id=1135
    # delete job artifacts
    curl -XPOST http://127.0.0.1:38888/cleanup\?id=1135

    # get job report
    curl -XGET http://127.0.0.1:38888/report\?id=1135
