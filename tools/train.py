import os
import pathlib
from tools.env_vars import train_args
from tools.log import logger
from llamafactory.train.tuner import run_exp


def update_args(dataset_dir, args=None) -> dict:
    """匹配大模型模板
    --lora_target LORA_TARGET
                    应用LoRA的目标模块名称。使用逗号分隔多个模块。使用"all"来指定所有可用的模块。
                    LLaMA 选择: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    BLOOM & Falcon & ChatGLM 选择: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
                    Baichuan 选择: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    Qwen 选择: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"],
                    InternLM2 选择: ["wqkv", "wo", "w1", "w2", "w3"],
                    Others 选择: 和 LLaMA一样。 (default: None)
    """

    args = args if isinstance(args, dict) else dict()

    # 1.根据模型家族,自动匹配<提示词模板> template
    template = os.environ.get("MODEL_TEMPLATE")
    model_family = os.environ.get("MODEL_FAMILY")
    model_template_type = {
        "baichuan": "baichuan",
        "baichuan2": "baichuan2",
        "gemma": "gemma",
        "deepseek": "deepseek",
        "deepseekcoder": "deepseekcoder",
        "chatglm2": "chatglm2",
        "chatglm3": "chatglm3",
        "glm4": "glm4",
        "codegeex4": "codegeex4",
        "intern": "intern",
        "intern2": "intern2",
        "intern2.5": "intern2",
        "llama": "llama",
        # "llama": "default",
        "llama2": "llama2",
        "llama3": "llama3",
        "llama3.1": "llama3",
        "llama3.2": "llama3",
        "chinese-alpaca-2": "llama2_zh",
        "mistral": "mistral",
        "qwen": "qwen",
        "qwen1.5": "qwen",
        "qwen2": "qwen",
        "qwen2-VL": "qwen2_vl",
        "qwen2.5": "qwen",
        "qwen3": "qwen3",
        "yi": "yi",
        "yi_vl": "yi_vl",
        "xuanyuan": "xuanyuan",
        "xverse": "xverse",
        "yuan2": "yuan"
    }
    if template:
        args["template"] = template
    elif model_family in model_template_type:
        args["template"] = model_template_type[model_family]
    else:
        args["template"] = "default"

    args["model_name_or_path"] = os.environ.get("BASE_MODEL_PATH")
    args["dataset_dir"] = dataset_dir
    args["dataset"] = "train"
    args["dataloader_num_workers"] = 0

    try:
        # 模型输出目录
        model_output_dir = os.environ.get("SAVEMODELPATH")
        pathlib.Path(model_output_dir).mkdir(parents=True, exist_ok=True)
        args["output_dir"] = model_output_dir
    except Exception as e:
        logger.error(e)

    # 训练参数日志
    logger.info(args)
    logger.info("=" * 100)

    return args


def check_args(args):
    if args["model_name_or_path"] is None:
        raise "model is Null"
    else:
        import glob
        model_file_list = glob.glob(args["model_name_or_path"] + "/**/*", recursive=True)
        logger.info(model_file_list)
        if len(model_file_list) <= 2:
            raise f"model is not found"


    if args["stage"] not in ["sft", "dpo", "ppo"]:
        raise f"{args['stage']} is not exist"


def train(dataset_dir, is_enable_train):
    """
    训练配置
    :param dataset_dir:
    :param is_enable_train:
    :return:
    """
    _train_args = update_args(dataset_dir, train_args)
    check_args(_train_args)

    logger.info("train_args:")
    logger.info(_train_args)

    # 开始训练
    if is_enable_train:
        logger.info("run_exp: ")
        run_exp(_train_args)

    return _train_args


if __name__ == '__main__':
    dataset_dir = r"E:\Inspur\GPU\mass\easyai-llamafactory\data\dpo_zh_demo.json"
    is_enable_train = True
    res = train(dataset_dir, is_enable_train)
    print(res)
