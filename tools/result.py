import os
import pathlib
import json
from tools.log import logger


def save_result(result_file_path, resultPath):
    results_dict = json.load(open(result_file_path, "r", encoding="utf8"))
    json_list = [{"key": k, "value": v} for k, v in results_dict.items()]
    json_path = pathlib.Path(resultPath, "result.json").as_posix()
    with open(json_path, "w", encoding="utf8") as jf:
        json.dump(json_list, jf, ensure_ascii=False, indent=4)

    return json_list

def result_analysis(train_args, eval_args=None, is_enable_eval=False):
    # 完成后保存评估结果
    if eval_args and not os.path.exists(eval_args["output_dir"]):
        os.makedirs(eval_args["output_dir"])

    if is_enable_eval and pathlib.Path(eval_args["output_dir"], "all_results.json").exists():
        result_file_path = pathlib.Path(eval_args["output_dir"], "all_results.json").as_posix()
    # elif pathlib.Path(os.environ.get("OUTPUT_DIR"), "train_results.json").exists():
        # result_file_path = pathlib.Path(os.environ.get("OUTPUT_DIR"), "train_results.json").as_posix()
    elif pathlib.Path(train_args["output_dir"], "train_results.json").exists():
        result_file_path = pathlib.Path(train_args["output_dir"], "train_results.json").as_posix()
    else:
        result_file_path = ""

    # 结果保存的目录路径
    resultPath = os.environ.get("RESULTPATH")
    pathlib.Path(resultPath).mkdir(parents=True, exist_ok=True)
    if result_file_path and resultPath:
        res_json = save_result(result_file_path, resultPath)
        logger.info(res_json)

    logger.info("=" * 100)

    # # 移动文件
    # train_dir = train_args["output_dir"]
    # eval_dir = eval_args["output_dir"]
    # for item in pathlib.Path(eval_dir).iterdir():
    #     if item.is_file():
    #         shutil.move(item.as_posix(), pathlib.Path(train_dir, item.name).as_posix())