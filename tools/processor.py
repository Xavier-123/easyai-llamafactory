import os
import random
import pathlib
import json
import hashlib
from tools.env_vars import train_args
from tools.log import logger
from datasets import load_dataset, load_from_disk


def recursive_jsonl(root_path, jsonl_list=None):
    """
    递归筛选jsonl文件
    :param root_path: 递归根路径
    :param jsonl_list: 已存在的文件列表
    :return: 递归出的jsonl文件路径列表
    """

    jsonl_list = jsonl_list if isinstance(jsonl_list, list) else []

    path_obj = pathlib.Path(root_path)
    if path_obj.is_dir():
        for item_path in pathlib.Path(root_path).iterdir():
            recursive_jsonl(item_path, jsonl_list)
    elif path_obj.suffix.lower() == ".jsonl" or path_obj.suffix.lower() == ".json":
        jsonl_list.append(path_obj.as_posix())

    return jsonl_list


def convert(jsonl_filepath):
    """
    数据集转换 -->  dataset_info.json
    :param jsonl_filepath: jsonl数据集文件路径
    :return: alpaca/sharegpt 格式的数据集列表
    """

    with open(jsonl_filepath, 'r', encoding="utf-8") as file:
        data_json = json.load(file)

    return data_json


def read_all_jsonl(jsonl_list):
    """读取并转换所有jsonl文件"""

    alpaca_json_list = []
    for jsonl_filepath in jsonl_list:
        alpaca_json_list += convert(jsonl_filepath)

    return alpaca_json_list


def dataset_conversion():
    # data_output_dir, is_enable_train, is_enable_eval = os.environ.get("DATA_OUTPUT_DIR"), True, True
    # data_output_dir, is_enable_train, is_enable_eval = r"/easyai-llamafactory/tmp/data/tmp", True, True
    data_output_dir, is_enable_train, is_enable_eval = os.path.join(r"E:\Inspur\GPU\mass\easyai-llamafactory\tmp\train", os.environ.get("STAGE")), True, True
    stage = train_args["stage"]

    # 判断数据格式

    # # 判断是不是蒸馏数据 DISTILL，如果是将格式转换为llamafactory格式
    # if os.environ.get("DISTILL"):
    #
    #     try:
    #         dataset = load_from_disk(dataset_path=os.environ.get("TRAIN_DATASET_PATH"))
    #         # logger.info(f"load_from_disk success.")
    #     except Exception as e:
    #         print(f"{e}")
    #         dataset = load_dataset(os.environ.get("TRAIN_DATASET_PATH"))
    #         # logger.info(f"load_dataset success.")

    # 格式1 - 不需要处理
    # train_test_type=0, 按比例划分数据集; train_test_type=1,独立测试集;
    # env_train_test_type = float(os.environ.get("TRAIN_TEST_TYPE"))
    env_train_test_type = float(os.environ.get("TRAINTESTTYPE"))
    if not env_train_test_type:
        """
        处理按比例分的情况
        读取 TRAINDATASETPATH 目录下的数据，按照 TRAINTESTRATIO 的比例划分训练集和测试集
        """
        env_train_dataset_path = os.environ.get("TRAINDATASETPATH")
        jsonl_list = recursive_jsonl(env_train_dataset_path)
        json_item_list = read_all_jsonl(jsonl_list)
        random.shuffle(json_item_list)

        # 读取训练集测试集比例
        env_trainTestRatio = float(os.environ.get("TRAINTESTRATIO"))
        train_Ratio = min(env_trainTestRatio, 1)
        test_Ratio = max(1 - train_Ratio, 0)

        if test_Ratio == 0:
            # 无测试集
            is_enable_eval = False
            train_list, test_list = json_item_list, []
        elif test_Ratio == 1:
            is_enable_train = False
            train_list, test_list = [], json_item_list
        else:
            test_num = round(len(json_item_list) * test_Ratio)
            test_idx_list = random.sample([i for i in range(len(json_item_list))], test_num)

            train_list, test_list = [], []
            for idx, item in enumerate(json_item_list):
                if idx in test_idx_list:
                    test_list.append(item)
                else:
                    train_list.append(item)

    else:
        """
        处理固定的训练集、测试集
        TRAINDATASETPATH - 训练集
        TESTDATASETPATH  - 测试集
        """
        # 读取trainDataSetPath下的所有jsonl数据集, 训练集
        env_train_dataset_path = os.environ.get("TRAINDATASETPATH")
        train_jsonl_list = recursive_jsonl(env_train_dataset_path)
        train_json_item_list = read_all_jsonl(train_jsonl_list)
        random.shuffle(train_json_item_list)

        # 读取testDataSetPath下的所有jsonl数据集, 测试集
        env_test_dataset_path = os.environ.get("TESTDATASETPATH")
        test_jsonl_list = recursive_jsonl(env_test_dataset_path)
        test_json_item_list = read_all_jsonl(test_jsonl_list)
        random.shuffle(test_json_item_list)
        train_list, test_list = train_json_item_list, test_json_item_list


    # 格式2 - 需要处理
    pass

    # 数据集保存路径
    pathlib.Path(data_output_dir).mkdir(parents=True, exist_ok=True)
    dataset_info_path = pathlib.Path(data_output_dir, "dataset_info.json").as_posix()
    train_dataset_path = pathlib.Path(data_output_dir, "train.json").as_posix()
    test_dataset_path = pathlib.Path(data_output_dir, "test.json").as_posix()

    # 保存数据集
    with open(train_dataset_path, "w", encoding="utf8") as f_train:
        json.dump(train_list, f_train, ensure_ascii=False, indent=4)
    with open(test_dataset_path, "w", encoding="utf8") as f_test:
        json.dump(test_list, f_test, ensure_ascii=False, indent=4)

    # 计算数据集sha1
    with open(train_dataset_path, "rb") as f1:
        train_sha1 = hashlib.sha1(f1.read()).hexdigest()
    with open(test_dataset_path, "rb") as f2:
        test_sha1 = hashlib.sha1(f2.read()).hexdigest()

    # 保存配置
    if stage == "dpo":
        dataset_info_dict = {
            "train": {
                "file_name": pathlib.Path(train_dataset_path).name,
                "ranking": True,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "chosen": "chosen",
                    "rejected": "rejected"
                },
                "file_sha1": train_sha1
            },
            "test": {
                "file_name": pathlib.Path(test_dataset_path).name,
                "ranking": True,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "chosen": "chosen",
                    "rejected": "rejected"
                },
                "file_sha1": test_sha1
            }
        }
        logger.info(dataset_info_dict)
    elif stage == "sft":
        dataset_info_dict = {}
    else:
        dataset_info_dict = {}

    with open(dataset_info_path, "w", encoding="utf8") as f_info:
        json.dump(dataset_info_dict, f_info, ensure_ascii=False, indent=4)

    return data_output_dir, is_enable_train, is_enable_eval


if __name__ == '__main__':
    jsonl_filepath = "/data/dpo_zh_demo.json"
    dataset_conversion()
    # convert(jsonl_filepath)
