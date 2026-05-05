"""
加载 pdf/txt、遍历目录、计算 MD5
"""

import os
import hashlib
from utils.logger_handler import logger
from langchain_community.document_loaders import TextLoader, PyPDFLoader

def get_file_md5_hex(file_path: str):
    """
    给一个文件路径算出 MD5 十六进制字符串
    :param file_path: 文件路径
    :return: 十六进制字符串
    """

    if not os.path.exists(file_path):
        logger.error(f"[md5计算]文件{file_path}不存在")
        return

    if not os.path.isfile(file_path):
        logger.error(f"[md5计算]路径{file_path}不是文件")
        return

    md5_obj = hashlib.md5()

    chunk_size = 4096      # 4KB分片, 避免文件过大导致内存溢出
    try:
        with open(file_path, "rb") as f:        # 必须二进制读取
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
            """
            chunk = f.read(chunk_size)
            while chunk:
                md5_obj.update(chunk)
                chunk = f.read(chunk_size)
            """
            md5_hex = md5_obj.hexdigest()
            return md5_hex
    except Exception as e:
        logger.error(f"[md5计算]文件{file_path}计算md5失败, {str(e)}")
        return None

def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    """
    给一个文件夹路径，找出来指定类型的文件，返回它们的完整路径
    :param path: 文件夹路径
    :param allowed_types: 被允许的文件类型
    :return: 返回元组格式的完整路径
    """
    files = []

    if not os.path.isdir(path):
        logger.error(f"[listdir_with_allowed_type]{path}不是文件夹")
        return allowed_types

    for f in os.listdir(path):
        if f.endswith(allowed_types):
            files.append(os.path.join(path, f))

    return tuple(files)

def pdf_loader(filepath: str, passward = None):
    """
    读取文件内容，把 PDF 变成 LangChain 能用的文档格式
    :param filepath: 文件路径
    :param passward: 密码
    :return: 文档格式
    """
    return PyPDFLoader(filepath).load()

def txt_loader(filepath: str):
    """
    读取文件内容，把 TXT 变成 LangChain 能用的文档格式
    :param filepath: 文件路径
    :return: 文档格式
    """
    return TextLoader(filepath, encoding="utf-8").load()