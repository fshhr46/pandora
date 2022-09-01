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
```python
python3 app.py --host=0.0.0.0 --port=18888 --log_level=DEBUG --log_dir=$HOME/workspace/pandora_outputs --output_dir=$HOME/workspace/pandora_outputs --data_dir=$HOME/workspace/resource/datasets --cache_dir=$HOME/.cache/torch/transformers
```

5. 跑e2e测试
```python
python e2e_test.py
```

6. Flaks服务的API
    # list all running jobs
    curl -XGET http://127.0.0.1:18888/list\?running=true
     # list all jobs
    curl -XGET http://127.0.0.1:18888/list\?running=false
    # start a new job
    curl -XPOST http://127.0.0.1:18888/start\?id=1135
    # get job status
    curl -XGET http://127.0.0.1:18888/status\?id=1135
    # stop a running job
    curl -XPOST http://127.0.0.1:18888/stop\?id=1135
    # create model package
    curl -XPOST http://127.0.0.1:18888/package\?id=1135
    # delete job artifacts (including model package)
    curl -XPOST http://127.0.0.1:18888/cleanup\?id=1135
    # get job report
    curl -XGET http://127.0.0.1:18888/report\?id=1135

7. 模型输出文件夹结构
```text
├── PANDORA_TRAINING_training1# 预训练模型
|  └── torchserve_package # 模型package
|  | └── vocab.txt
|  | └── config.json
|  | └── pytorch_model.bin
|  | └── setup_config.json
|  | └── handler.py
|  | └── index_to_name.json
|  | └── package.sh
|  | └── done
|  └── predict # 模型报告
|  | └── report.json
|  | └── report_all.json
```
