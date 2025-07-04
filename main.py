import os
from tools.processor import dataset_conversion
from tools.train import train
from tools.eval import evaluation
from tools.log import logger
from tools.env_vars import train_args, eval_args, merge_args, quantization_args
from tools.result import result_analysis
from tools.quantization import quantization


def main():
    do_train = os.environ.get("DO_TRAIN")
    do_quantization = os.environ.get("DO_QUANTIZATION")
    if do_train.lower() == "false" or do_train == False:
        if do_quantization.lower() == "true" or do_quantization == True:
            # 量化模型
            quantization()
    else:
        # 数据集转换
        dataset_dir, is_enable_train, is_enable_eval = dataset_conversion()

        # 训练
        print("-" * 50 + " 开始训练 " + "-" * 50)
        train_args = train(dataset_dir, is_enable_train)

        # 评估
        if 0 < float(os.environ.get("TRAIN_TEST_RATIO")) < 1:
            print("-" * 50 + " 开始评估 " + "-" * 50)
            eval_args = evaluation(train_args, is_enable_train, is_enable_eval)

        # 结果解析
        print("-" * 50 + " 解析结果 " + "-" * 50)
        if 0 < float(os.environ.get("TRAIN_TEST_RATIO")) < 1:
            result_analysis(train_args, eval_args, is_enable_eval)
        else:
            result_analysis(train_args, is_enable_eval=False)

        # 合并权重
        if os.environ.get("MERGE_LORA", False):
            print("-" * 50 + " 合并权重 " + "-" * 50)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(os.environ.get("TRAIN_TEST_TYPE"))
    if os.environ.get("DO_TRAIN"):
        logger.info("train_args:")
        logger.info(train_args)
    if os.environ.get("DO_EVAL"):
        logger.info("eval_args:")
        logger.info(eval_args)
    if os.environ.get("DO_MERGE"):
        logger.info("merge_args:")
        logger.info(merge_args)
    if os.environ.get("DO_QUANTIZATION"):
        logger.info("quantization_args:")
        logger.info(quantization_args)
    main()
