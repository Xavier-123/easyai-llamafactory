import os
import pathlib
from tools.log import logger
from tools.env_vars import train_args, eval_args
from llamafactory.train.tuner import run_exp

def evaluation(train_args, is_enable_train=True, is_enable_eval=True, eval_dir="./tmp/eval"):
    """
    评估配置
    :param train_args:
    :param is_enable_train:
    :param is_enable_eval:
    :param eval_dir:
    :return:
    """

    pathlib.Path(eval_args["output_dir"]).mkdir(parents=True, exist_ok=True)
    eval_args["stage"] = train_args["stage"]
    eval_args["finetuning_type"] = train_args["finetuning_type"]
    eval_args["template"] = train_args["template"]
    eval_args["model_name_or_path"] = train_args["model_name_or_path"]
    eval_args["dataset_dir"] = train_args["dataset_dir"]
    # eval_args["adapter_name_or_path"] = os.environ.get("OUTPUT_DIR")
    eval_args["adapter_name_or_path"] = os.environ.get("SAVEMODELPATH")      # 平台的模型保存目录

    # eval_args = dict()
    # if is_enable_eval:
    #     # 评估参数
    #     eval_args["stage"] = args["stage"]
    #     eval_args["model_name_or_path"] = args["model_name_or_path"]
    #     if is_enable_train:
    #         eval_args["adapter_name_or_path"] = args["output_dir"]
    #         eval_args["finetuning_type"] = args["finetuning_type"]
    #     eval_args["template"] = args["template"]
    #     eval_args["dataset_dir"] = args["dataset_dir"]
    #     # eval_args["dataset"] = "test"
    #     eval_args["eval_dataset"] = "test"
    #     eval_args["cutoff_len"] = args["cutoff_len"]
    #     eval_args["max_samples"] = args["max_samples"]
    #     eval_args["per_device_eval_batch_size"] = args["per_device_train_batch_size"]
    #     eval_args["predict_with_generate"] = bool(int(os.environ.get("PREDICT_WITH_GENERATE", "1")))
    #     eval_args["max_new_tokens"] = int(os.environ.get("MAX_NEW_TOKENS", "128"))
    #     eval_args["top_p"] = float(os.environ.get("TOP_P", "0.7"))
    #     eval_args["temperature"] = float(os.environ.get("TEMPERATURE", "0.95"))
    #     eval_args["do_predict"] = bool(int(os.environ.get("DO_PREDICT", "1")))
    #     eval_args["output_dir"] = eval_dir
    #     eval_args["do_eval"] = True
    #     eval_args["do_train"] = False
    #     eval_args["predict_with_generate"] = False
    #
    #     # eval_args["eval_dataset"] = "test"
    #     # eval_args["val_size"] = 0.1
    #     eval_args["per_device_eval_batch_size"] = 1
    #     eval_args["eval_strategy"] = "steps"
    #     eval_args["eval_steps"] = 500
    #
    #     # 评估参数日志
    #     logger.info(eval_args)
    #     logger.info("=" * 100)
    #
    #     # 开始评估
    #     try:
    #         run_exp(eval_args)
    #     except Exception as e:
    #         raise e
    #     logger.info("*" * 100)

    # 开始评估
    try:
        logger.info("************************************************** eval_args **************************************************")
        logger.info(eval_args)
        run_exp(eval_args)
    except Exception as e:
        raise e
    logger.info("*" * 100)

    return eval_args