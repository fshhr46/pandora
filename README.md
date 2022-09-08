
# 快速Demo - 训练，注册，上线
### 拉取镜像
docker pull pytorch/torchserve:0.5.3-gpu

### 启动服务
python3 app.py --host=0.0.0.0 --port=18888 --log_level=DEBUG --log_dir=$HOME/workspace/pandora_outputs --output_dir=$HOME/workspace/pandora_outputs --data_dir=$HOME/workspace/resource/datasets --cache_dir=$HOME/.cache/torch/transformers

### 创建一个torchserve container
docker run -d -it --mount type=bind,source=$HOME/workspace,target=/app/workspace --name torchserve_demo pytorch/torchserve:0.5.3-gpu

### 生成测试数据
curl -XPOST http://127.0.0.1:18888/testdata\?id=demo_1

### 进行数据分区
curl -XPOST http://127.0.0.1:18888/partition\?id=demo_1

### 调用服务训练模型
curl -XPOST http://127.0.0.1:18888/start\?id=demo_1

### 训练完成后打包
curl -XPOST http://127.0.0.1:18888/package\?id=demo_1

### 打包完成后登录到docker里执行注册脚本
docker exec -it --user root torchserve_demo bash -c "cd /app/workspace/pandora_outputs/PANDORA_TRAINING_demo_1/torchserve_package && bash register.sh demo 1.0"

### 上线（添加worker）
docker exec -it --user root torchserve_demo bash -c "curl -v -X PUT \"http://localhost:8081/models/demo?min_worker=1&synchronous=true\""

### 推理测试
docker exec -it --user root torchserve_demo bash -c "curl -d \"data=hello world\" \"http://localhost:8080/predictions/demo/1.0\""
