# 一、SFT

环境变量

```shell
MODEL_NAME_OR_PATH="E:\models\Qwen\Qwen2___5-0___5B-Instruct"   # 基础模型目录
STAGE="sft"                                                     # 微调数据格式 [sft, dpo]
DO_TRAIN="True"                                                 # 训练模式
DO_EVAL="False"                                                 # 评估模式
FINETUNING_TYPE="lora"                                          # 微调算法 lora, qlora, full
LORA_RANK=8                                                     # lora 秩
LORA_DROPOUT=0.1                                                # 抛弃比例
LORA_TARGET="all"                                               # 应用LoRA的目标模块名称
DATASET="train"                                                 # 数据集
LORA_DROPOUT="0.1     "                                         # 
CUTOFF_LEN=1024                                                 # 截断长度
MAX_SAMPLES=1000000                                             # 最大样本数
OVERWRITE_CACHE=True  # 覆盖缓存
PREPROCESSING_NUM_WORKERS=1  # 预处理worker数
DATALOADER_NUM_WORKERS=1  # 数据加载worker数
OUTPUT_DIR=""  # 模型保存目录
LOGGING_STEPS=""  # 多少steps打印一次日志
SAVE_STEPS=""     # 多少steps能保存一次训练参数
WARMUP_STEPS=""  #
OVERWRITE_OUTPUT_DIR=""  # 是否覆盖已有目录
PLOT_LOSS=""  # 是否将训练结果绘图
REPORT_TO=""  # 
PER_DEVICE_TRAIN_BATCH_SIZE=2  # 训练时每个设备每批数据大小
GRADIENT_ACCUMULATION_STEPS=4  # 多少steps更新梯度
LEARNING_RATE=5e-05  # 学习率 
NUM_TRAIN_EPOCHS=3  # 训练 epoch 数
LR_SCHEDULER_TYPE=cosine  # 
MAX_GRAD_NORM=1.0  # 
WARMUP_RATIO=0.1  # 
FP16=True  # 
DDP_TIMEOUT=180000000  # 
RESUME_FROM_CHECKPOINT=None  # 
TUNED_TYPE_=lora  # 
TUNED_TYPE=lora   # 
WARMUP_RATIO=0.1  # 



# 评估参数
eval_args = {}
eval_args["stage"] = str(os.environ.get("STAGE", "dpo"))
eval_args["model_name_or_path"] = str(os.environ.get("MODEL_NAME_OR_PATH", ""))
eval_args["adapter_name_or_path"] = str(os.environ.get("ADAPTER_NAME_OR_PATH", ""))
eval_args["finetuning_type"] = str(os.environ.get("FINETUNING_TYPE", "lora"))
eval_args["template"] = str(os.environ.get("TEMPLATE", "qwen"))
eval_args["dataset_dir"] = str(os.environ.get("DATASET_DIR", ""))
eval_args["eval_dataset"] = "test"
eval_args["cutoff_len"] = int(os.environ.get("CUTOFF_LEN", 128))
eval_args["max_samples"] = int(os.environ.get("MAX_SAMPLES", 10000000))
eval_args["per_device_eval_batch_size"] = int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", 1))
eval_args["max_new_tokens"] = int(os.environ.get("MAX_NEW_TOKENS", 256))
eval_args["top_p"] = float(os.environ.get("TOP_P", 0.7))
eval_args["temperature"] = float(os.environ.get("TEMPERATURE", 0.95))
eval_args["do_predict"] = bool(int(os.environ.get("DO_PREDICT", "1")))
eval_args["do_eval"] = True
eval_args["do_train"] = False
eval_args["predict_with_generate"] = False     # sft(True), dpo(False)
eval_args["eval_strategy"] = str(os.environ.get("EVAL_STRATEGY", "steps"))
eval_args["eval_steps"] = int(os.environ.get("EVAL_STEPS", 500))
# eval_args["output_dir"] = str(os.environ.get("EVAL_OUTPUT_DIR", ""))
eval_args["output_dir"] = f"./tmp/eval/{eval_args['stage']}"
# eval_args["output_dir"] = "./tmp/eval"
```

运行

```shell

```



# 二、DPO

## 环境变量说明

```shell
# 基础大模型目录
BASE_MODEL_PATH=/nfs_data/models/Qwen/Qwen2.5-7B-Instruct   # （平台默认传）
TEMPLATE=qwen                                               # 模型模板（平台默认传）
TRAIN_TEST_TYPE=1   # 自定义数据比例：0(只用TRAIN_DATASET_PATH)；使用划分好的训练测试数据：1（使用TRAIN_DATASET_PATH、TEST_DATASET_PATH）
TRAIN_DATASET_PATH=/X-llama-factory/tmp/data/dpo_zh_demo.json        # 训练数据（平台默认传）
TEST_DATASET_PATH=/X-llama-factory/tmp/data/dpo_zh_demo.json         # 测试数据（平台默认传）
TRAIN_TEST_RATIO=1                                       # 训练集比例 0~1（平台默认传）
DATA_OUTPUT_DIR=/X-llama-factory/tmp/train               # 数据预处理后保存目录
SAVEMODELPATH=E:\Inspur\GPU\mass\easyai-llamafactory\output\dpo         # 训练后模型保存目录（平台默认传）
RESULTPATH=E:\Inspur\GPU\mass\easyai-llamafactory\output\dpo\result     # 训练指标结果输出（平台默认传）
FINETUNING_TYPE=lora                          # 微调算法  full/lora/qlora
NUM_TRAIN_EPOCHS=3                            # 训练 epoch 数
LEARNING_RATE=5e-05                           # 学习率
PER_DEVICE_TRAIN_BATCH_SIZE=5e-05             # 训练batch size
```

Dockerfile

```Dockerfile
FROM harbor.inspur.local/ai-group/llamafactory:v0.9.1-py311
LABEL authors="xaw"

COPY /easyai-llamafactory /easyai-llamafactory

# 设置工作目录
WORKDIR /easyai-llamafactory/scripts

ENTRYPOINT ["python", "main.py"]
```

## 运行

### docker run启动方式

```shell
docker run -it -u root \
--gpus all \
--ipc=host \
--network=bridge \
-p 19201:8018 \
-w /X-llama-factory \
-v /nfs_data:/nfs_data \
-e CUDA_VISIBLE_DEVICES=1 \
-e BASE_MODEL_PATH=/nfs_data/models/Qwen/Qwen2.5-7B-Instruct \
-e TEMPLATE=qwen \
-e TRAIN_TEST_TYPE=1 \
-e TRAIN_DATASET_PATH=/X-llama-factory/tmp/data/dpo_zh_demo.json \
-e TEST_DATASET_PATH=/X-llama-factory/tmp/data/dpo_zh_demo.json \
-e TRAIN_TEST_RATIO=1 \
-e RESULTPATH=/X-llama-factory/tmp/result \
--entrypoint python \
--name llama-factory-test \
x-llama-factory:v0.9.3 \
main.py


docker run -it -u root \
--gpus all \
--ipc=host \
--network=bridge \
--entrypoint bash \
-v /nfs_data:/nfs_data \
--name llama-factory-test \
harbor.inspur.local/ai-group/llamafactory:v0.9.1-py311-v0.1 

CUDA_VISIBLE_DEVICES=1 DO_TRAIN=true BASE_MODEL_PATH=/nfs_data/models/Qwen/Qwen2.5-7B-Instruct TEMPLATE=qwen TRAIN_TEST_TYPE=1 TRAIN_DATASET_PATH=/easyai-llamafactory/data/dpo_zh_demo.json TEST_DATASET_PATH=/easyai-llamafactory/data/dpo_zh_demo.json TRAIN_TEST_RATIO=1 SAVEMODELPATH=/nfs_data/xaw/deploy/llamafactory/output/dpo RESULTPATH=/nfs_data/xaw/deploy/llamafactory/output/dpo/result python main.py
```

### docker-compose启动方式

```yaml
services:

  llamafactory-service:
    container_name: llamafactory-service
    image: harbor.inspur.local/ai-group/llamafactory:v0.9.1-py311-v0.1
    restart: "no"
    working_dir: /easyai-llamafactory
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - BASE_MODEL_PATH=/nfs_data/models/Qwen/Qwen2.5-7B-Instruct
      - TEMPLATE=qwen
      - TRAIN_TEST_TYPE=1
      - TRAIN_DATASET_PATH=/easyai-llamafactory/tmp/data/dpo_zh_demo.json
      - TEST_DATASET_PATH=/easyai-llamafactory/tmp/data/dpo_zh_demo.json
      - TRAIN_TEST_RATIO=1
      - DATASET_DIR=/nfs_data/xaw/deploy/llamafactory/output/tmp/data/tmp
      - OUTPUT_DIR=/nfs_data/xaw/deploy/llamafactory/output/tmp/train
      - RESULTPATH=/nfs_data/xaw/deploy/llamafactory/output/tmp/result
    volumes:
      - /nfs_data:/nfs_data
    entrypoint: >
      python main.py
```



# 三、模型量化

## 环境变量

```yaml
DO_TRAIN=false                                              # 是否训练
DO_QUANTIZATION=true                                        # 是否量化
MODEL_NAME_OR_PATH=E:\models\Qwen\Qwen2.5-0.5B-Instruct     # 基础大模型目录
TEMPLATE=qwen                                               # 模型模板
TRUST_REMOTE_CODE=true                                          
EXPORT_DIR=E:\models\Qwen\Qwen2.5-0.5B-Instruct-gptq-int2   # 量化后模型保存目录
EXPORT_QUANTIZATION_BIT=2                   # 量化位数[8, 4, 3, 2]
EXPORT_QUANTIZATION_DATASET=tmp/data/c4_demo.jsonl          # 量化校准数据集
EXPORT_SIZE=1                               # 最大导出模型文件大小
EXPORT_DEVICE=cpu                           # 导出设备，还可以为: [cpu, auto]
EXPORT_LEGACY_FORMAT=false                                  # 是否使用旧格式导出
```

## docker-compose.yaml

```yaml
services:

  llamafactory-service:
    container_name: llamafactory-service
    image: harbor.inspur.local/ai-group/llamafactory:v0.9.1-py311
    ports:
      - 18018:8018
    restart: "no"
    working_dir: /easyai-llamafactory
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - DO_TRAIN=false
      - DO_QUANTIZATION=true
      - MODEL_NAME_OR_PATH=/data/models/Qwen/Qwen2.5-0.5B-Instruct
      - TEMPLATE=qwen
      - TRUST_REMOTE_CODE=true
      - EXPORT_DIR=/data/models/Qwen/Qwen2.5-0.5B-Instruct-gptq-int2
      - EXPORT_QUANTIZATION_BIT=2
      - EXPORT_QUANTIZATION_DATASET=tmp/data/c4_demo.jsonl
      - EXPORT_SIZE=1
      - EXPORT_DEVICE=cpu
      - EXPORT_LEGACY_FORMAT=false
    volumes:
      - /data:/data
      - ./easyai-llamafactory:/easyai-llamafactory
    entrypoint: >
      bash /easyai-llamafactory/scripts/quantization.sh
```

