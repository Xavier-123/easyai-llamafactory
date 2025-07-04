import os

if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# 训练参数
train_args = dict()
# 模型
train_args["model_name_or_path"] = os.environ.get("MODEL_NAME_OR_PATH", "")
# 微调方法
train_args["stage"] = os.environ.get("STAGE", "dpo")  # sft, dpo   # 微调数据格式
train_args["do_train"] = bool(os.environ.get("DO_TRAIN", True))  # 训练模式
train_args["do_eval"] = bool(os.environ.get("DO_EVAL", False))  # 评估模式
train_args["finetuning_type"] = os.environ.get("FINETUNING_TYPE", "lora")  # 微调算法 lora, qlora, full
train_args["lora_rank"] = int(os.environ.get("LORA_RANK", 8))
train_args["lora_dropout"] = float(os.environ.get("LORA_DROPOUT", 0.1))
train_args["lora_target"] = os.environ.get("LORA_TARGET", "all")  # 应用LoRA的目标模块名称
# 数据集参数
train_args["dataset"] = str(os.environ.get("DATASET", "train"))
train_args["template"] = str(os.environ.get("TEMPLATE", "qwen"))
# train_args["cutoff_len"] = int(os.environ.get("CUTOFF_LEN", 2048))         # 截断长度
train_args["cutoff_len"] = int(os.environ.get("CUTOFF_LEN", 128))
train_args["max_samples"] = int(os.environ.get("MAX_SAMPLES", 1000000))
train_args["overwrite_cache"] = bool(os.environ.get("OVERWRITE_CACHE", True))
train_args["preprocessing_num_workers"] = int(os.environ.get("PREPROCESSING_NUM_WORKERS", 1))
train_args["dataloader_num_workers"] = int(os.environ.get("DATALOADER_NUM_WORKERS", 1))
# 输出
train_args["output_dir"] = str(os.environ.get("OUTPUT_DIR", ""))
# train_args["output_dir"] = f"./tmp/train/{train_args['stage']}"
train_args["logging_steps"] = int(os.environ.get("LOGGING_STEPS", 10))
train_args["save_steps"] = int(os.environ.get("SAVE_STEPS", "1000000000"))
train_args["warmup_steps"] = int(os.environ.get("WARMUP_STEPS", 0))
train_args["overwrite_output_dir"] = bool(os.environ.get("OVERWRITE_OUTPUT_DIR", True))
train_args["plot_loss"] = bool(os.environ.get("PLOT_LOSS", True))
train_args["save_only_model"] = bool(os.environ.get("SAVE_ONLY_MODEL", True))
train_args["report_to"] = os.environ.get("REPORT_TO", None)
# 训练
# train_args["per_device_train_batch_size"] = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 4))  # 训练时每个设备每批数据大小
# train_args["gradient_accumulation_steps"] = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 4))
train_args["per_device_train_batch_size"] = int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", 1))
train_args["gradient_accumulation_steps"] = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))  # 训练时每个设备每批数据大小
train_args["learning_rate"] = float(os.environ.get("LEARNING_RATE", 5e-05))  # 学习率
# train_args["num_train_epochs"] = float(os.environ.get("NUM_TRAIN_EPOCHS", 3.0))  # 训练 epoch 数
train_args["num_train_epochs"] = float(os.environ.get("NUM_TRAIN_EPOCHS", 0.01))
train_args["lr_scheduler_type"] = os.environ.get("LR_SCHEDULER_TYPE", "cosine")
train_args["max_grad_norm"] = float(os.environ.get("MAX_GRAD_NORM", 1.0))
train_args["warmup_ratio"] = float(os.environ.get("WARMUP_RATIO", 0.1))
train_args["fp16"] = bool(os.environ.get("FP16", True))
train_args["ddp_timeout"] = int(os.environ.get("DDP_TIMEOUT", 180000000))
train_args["resume_from_checkpoint"] = os.environ.get("DDP_TIMEOUT", None)

if len(str(os.environ.get("cuda_visible_devices", 0))) > 1 and os.environ.get("USE_DEEPSPEED", "") in ('true', 'True'):
    train_args["deepspeed"] = os.environ.get("DEEPSPEED", "ds_z0")
os.environ["TUNED_TYPE"] = str(os.environ.get("TUNED_TYPE_", "lora")).lower()

# # 传入的环境变量
# os.environ["BASE_MODEL_PATH"] = os.environ.get("BASE_MODEL_PATH", r"F:\inspur\LLM_MODEL\Qwen\Qwen2-0___5B-Instruct-GPTQ-Int4").lower()    # 基础大模型
# os.environ["TRAIN_DATASET_PATH"] = os.environ.get("TRAIN_DATASET_PATH",
#                                                   r"F:\inspur\GPU\code\X-llama-factory\tmp\data\dpo_zh_demo.json").lower()    # 初始数据目录
# os.environ["TEST_DATASET_PATH"] = os.environ.get("TRAIN_DATASET_PATH", "")               # 独立数据集-测试集
# # os.environ["TRAIN_TEST_TYPE"] = os.environ.get("TEST_DATASET_PATH", 0)                 # 是否自定义数据集
# os.environ["TRAIN_TEST_RATIO"] = os.environ.get("TRAIN_TEST_RATIO", "0.99")              # 训练集比例
# os.environ["TEST_DATASET_PATH"] = os.environ.get("TEST_DATASET_PATH", "")                # 测试集目录
# os.environ["DATASET_DIR"] = r"F:\inspur\GPU\code\X-llama-factory\tmp\data\tmp"           # 转换后数据目录
# os.environ["RESULTPATH"] = r"F:\inspur\GPU\code\X-llama-factory\tmp\result"              # 训练结果保存目录
# os.environ["OUTPUT_DIR"] = r"F:\inspur\GPU\code\X-llama-factory\tmp\train_output"   # 模型保存目录


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
eval_args["predict_with_generate"] = False  # sft(True), dpo(False)
eval_args["eval_strategy"] = str(os.environ.get("EVAL_STRATEGY", "steps"))
eval_args["eval_steps"] = int(os.environ.get("EVAL_STEPS", 500))
# eval_args["output_dir"] = str(os.environ.get("EVAL_OUTPUT_DIR", ""))
eval_args["output_dir"] = f"./tmp/eval/{eval_args['stage']}"
# eval_args["output_dir"] = "./tmp/eval"


# 合并参数
merge_args = {}
merge_args["model_name_or_path"] = str(os.environ.get("MODEL_NAME_OR_PATH", ""))  # 基础模型目录
merge_args["adapter_name_or_path"] = str(os.environ.get("ADAPTER_NAME_OR_PATH", ""))  # 微调模型目录
merge_args["template"] = str(os.environ.get("TEMPLATE", ""))  # 模型模板
merge_args["finetuning_type"] = str(os.environ.get("FINETUNING_TYPE", "lora"))  # 微调方式
merge_args["export_dir"] = str(os.environ.get("FINETUNING_TYPE", ""))  # 合并后模型导出目录
merge_args["export_size"] = int(os.environ.get("EXPORT_SIZE", 2))  #
merge_args["export_device"] = str(os.environ.get("EXPORT_DEVICE", "cpu"))  #
merge_args["export_legacy_format"] = bool(os.environ.get("EXPORT_LEGACY_FORMAT", False))  # 保留旧格式

# 量化
# os.environ.setdefault("DO_TRAIN", "False")
# os.environ.setdefault("DO_QUANTIZATION", "True")
# os.environ.setdefault("MODEL_NAME_OR_PATH", "E:\models\Qwen\Qwen2___5-0___5B-Instruct")
# os.environ.setdefault("TEMPLATE", "qwen")
# os.environ.setdefault("TRUST_REMOTE_CODE", "True")
# os.environ.setdefault("EXPORT_DIR", "tmp/quantization/qwen2.5_gptq_int2")
# os.environ.setdefault("EXPORT_QUANTIZATION_BIT", "2")
# os.environ.setdefault("EXPORT_QUANTIZATION_DATASET", "tmp/data/c4_demo.jsonl")
# os.environ.setdefault("EXPORT_SIZE", "1")
# os.environ.setdefault("EXPORT_DEVICE", "cpu")
# os.environ.setdefault("EXPORT_LEGACY_FORMAT", "False")

quantization_args = {}
quantization_args["do_train"] = bool(os.environ.get("DO_TRAIN", False))  # 训练模式
quantization_args["do_quantization"] = bool(os.environ.get("DO_QUANTIZATION", True))  # 量化模型
quantization_args["model_name_or_path"] = str(os.environ.get("MODEL_NAME_OR_PATH", ""))  # 模型目录
quantization_args["template"] = str(os.environ.get("TEMPLATE", ""))  # 模型模板
quantization_args["trust_remote_code"] = bool(os.environ.get("TRUST_REMOTE_CODE", True))
quantization_args["export_dir"] = str(os.environ.get("EXPORT_DIR", ""))  # 模型保存目录
quantization_args["export_quantization_bit"] = str(os.environ.get("EXPORT_QUANTIZATION_BIT", ""))  # 量化位数[8, 4, 3, 2]
quantization_args["export_quantization_dataset"] = str(os.environ.get("EXPORT_QUANTIZATION_DATASET", ""))  # 量化校准数据集
quantization_args["export_size"] = str(os.environ.get("EXPORT_SIZE", ""))  # 最大导出模型文件大小
quantization_args["export_device"] = str(os.environ.get("EXPORT_DEVICE", ""))  # 导出设备，还可以为: [cpu, auto]
# quantization_args["export_legacy_format"] = bool(os.environ.get("EXPORT_LEGACY_FORMAT", False))                # 是否使用旧格式导出
quantization_args["export_legacy_format"] = False
