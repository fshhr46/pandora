
# 快速Demo - 训练，注册，上线

## 启动流程
### 拉取镜像
docker pull pytorch/torchserve:0.5.3-gpu

### 启动服务
python3 app.py --host=0.0.0.0 --port=38888 --log_level=DEBUG --log_dir=$HOME/workspace/pandora_outputs --output_dir=$HOME/workspace/pandora_outputs --data_dir=$HOME/workspace/resource/datasets --cache_dir=$HOME/.cache/torch/transformers

### 创建一个torchserve container
docker run -d -it --mount type=bind,source=$HOME/workspace,target=/app/workspace --name torchserve_demo pytorch/torchserve:0.5.3-gpu

## 模型训练Demo流程
### 生成测试数据
curl -XPOST http://127.0.0.1:38888/testdata?id=demo_1&job_type=training&file_name=meta_data.json

### 或者以文件形式注入测试数据（可选，optional）
curl -XPOST -H "Content-Type: application/json;charset=UTF-8" -d @test_data/dataset.json http://127.0.0.1:38888/ingest-dataset?id=demo_1&job_type=training

### 进行数据分区
curl -XPOST http://127.0.0.1:38888/partition?id=demo_1

### 调用服务训练模型
curl -XPOST http://127.0.0.1:38888/start?id=demo_1

### 查询任务状态
curl http://127.0.0.1:38888/status?id=demo_1&job_type=training

### 查询所有正在跑的训练任务（可选，optional）
curl http://127.0.0.1:38888/list?running=true&job_type=training

### 训练完成后打包
curl -XPOST http://127.0.0.1:38888/package?id=demo_1

### 打包完成后可以下载模型（可选，optional）
curl -XPOST http://127.0.0.1:38888/download?id=test_remote_1 --output package.zip

### 打包完成后登录到docker里执行注册脚本
docker exec -it --user root torchserve_demo bash -c "cd /app/workspace/pandora_outputs/PANDORA_TRAINING_demo_1/torchserve_package && bash register.sh demo 1.0"

### 中止任务（可选，optional）
curl -XPOST http://127.0.0.1:38888/stop?id=demo_1&job_type=training

### 删除任务
curl -XPOST http://127.0.0.1:38888/cleanup?id=demo_1&job_type=training

### 上线（添加worker）
docker exec -it --user root torchserve_demo bash -c "curl -v -X PUT \"http://localhost:8081/models/demo?min_worker=1&synchronous=true\""

### 推理测试
docker exec -it --user root torchserve_demo bash -c "curl -d \"data=hello world\" \"http://localhost:8080/predictions/demo/1.0\""

## 关键词提取Demo流程
### 上传数据数据
curl -XPOST -H "Content-Type: application/json;charset=UTF-8" -d @test_data/dataset.json http://127.0.0.1:38888/ingest-dataset?id=demo_1&job_type=keywords

### 调用接口提取关键词(提供模型ID以及模型版本)
curl -XPOST http://127.0.0.1:38888/extract-keywords?id=demo_1&model_name=1&model_version=0

### 查询任务状态
curl http://127.0.0.1:38888/status?id=demo_1&job_type=keywords

### 查询所有正在跑的关键词任务（可选，optional）
curl http://127.0.0.1:38888/list?running=true&job_type=keywords

### 任务完成后获取结果
curl http://127.0.0.1:38888/get-keywords?id=demo_1

### 中止任务（可选，optional）
curl -XPOST http://127.0.0.1:38888/stop?id=demo_1&job_type=keywords

### 删除任务
curl -XPOST http://127.0.0.1:38888/cleanup?id=demo_1&job_type=keywords
