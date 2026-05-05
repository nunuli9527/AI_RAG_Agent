"""
加载 txt 提示词
"""

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompt():
    """
    加载系统提示词
    :return:
    """
    try:
        # 获取绝对路径
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompt]在yaml配置项中没有main_prompt_path配置项")
        return e

    try:
        # 读取系统提示词
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompt]解析系统提示词出错, {str(e)}")
        return e


def load_rag_prompt():
    """
    加载RAG总结提示词
    :return:
    """
    try:
        # 获取绝对路径
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompt]在yaml配置项中没有rag_summarize_prompt_path配置项")
        return e

    try:
        # 读取系统提示词
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompt]解析RAG总结提示词出错, {str(e)}")
        return e


def load_report_prompt():
    """
    加载报告生成提示词
    :return:
    """
    try:
        # 获取绝对路径
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompt]在yaml配置项中没有report_prompt_path配置项")
        return e

    try:
        # 读取系统提示词
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompt]解析报告生成提示词出错, {str(e)}")
        return e


if __name__ == '__main__':
    # print(load_system_prompt())
    # print(load_rag_prompt())
    print(load_report_prompt())