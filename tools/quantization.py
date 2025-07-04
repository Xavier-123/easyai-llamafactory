import os.path
import yaml
from llamafactory.train.tuner import export_model
from tools.env_vars import quantization_args
from tools.log import curr_path, logger
from tools.error import FieldIsEmptyError


def quantization():
    "llamafactory-cli export **.yaml"
    # E:\Inspur\GPU\code\llama-factory\v0.9.3\LLaMA-Factory\examples\merge_lora\qwen3_gptq.yaml

    # 获取环境变量
    try:
        del quantization_args["do_quantization"]
        del quantization_args["do_train"]

        logger.info("quantization_args: ")
        logger.info(quantization_args)
        for arg in quantization_args:
            if isinstance(quantization_args[arg], bool):
                if quantization_args[arg] not in [True, False]:
                    logger.error(FieldIsEmptyError(arg))
                    raise FieldIsEmptyError(arg)
            else:
                if len(quantization_args[arg]) == 0:
                    logger.error(FieldIsEmptyError(arg))
                    raise FieldIsEmptyError(arg)

        quantization_args["export_size"] = int(quantization_args["export_size"])
        quantization_args["export_quantization_bit"] = int(quantization_args["export_quantization_bit"])
        logger.info("quantization_args:")
        logger.info(quantization_args)

    except Exception as e:
        logger.error(e)
        raise e

    # 写进文件
    file_path = os.path.join(os.path.dirname(curr_path), "tmp/quantization/quantization.yaml")
    with open(os.path.join(file_path), 'w') as f:
        yaml.dump(quantization_args, f)

    # 执行脚本
    export_model()

    logger.info("export done.")

    pass


if __name__ == '__main__':
    quantization()