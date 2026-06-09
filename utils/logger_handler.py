"""
日志封装
"""

import logging
from utils.path_tool import get_abs_path
import os
from datetime import datetime

# 日志保存的根目录
LOG_ROOT = get_abs_path("logs")

# 确保日志的目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志的输出格式配置 error info debug
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s: %(lineno)d - %(message)s'
)


def get_logger(
        name: str = "agent",
        console_level: int = logging.ERROR,
        file_level: int = logging.DEBUG,
        log_file = None
) -> logging.Logger:
    """
    获取日志器
    :param name: 日志器的名称
    :param console_level: 控制台的日志级别
    :param file_level: 文件的日志级别
    :param log_file: 日志文件的存放路径
    :return: 日志器
    """

    # 第一步: 先拿到日志对象
    logger = logging.getLogger(name)
    # 给日志器开一个 总开关
    logger.setLevel(logging.DEBUG)

    # 第二步：立刻判断, 有没有处理器
    # 避免重复添加handler, 防止日志重复打印
    if logger.handlers:
        return logger

    # 第三步：开始配置级别, 加控制台, 加文件处理器
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(console_handler)

    # 文件handler
    if not log_file:        # 日志文件的存放路径
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)

    logger.addHandler(file_handler)

    return logger

# 快捷获取日志器
logger = get_logger()


if __name__ == '__main__':
    logger.info("hello world")
    logger.error("错误日志")
    logger.warning("警告日志")
    logger.debug("debug日志")