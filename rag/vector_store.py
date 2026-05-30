"""
向量数据库服务
作用：管理文档、切片、存入向量库、提供检索。
功能：
读取 PDF / TXT
文档分块
计算 MD5 去重
存入 Chroma 向量库
提供检索器给 RAG 服务
"""

import os.path
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from langchain_chroma import Chroma
from utils.path_tool import get_abs_path
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    """
    向量库服务: \n
    获取向量库的检索器get_retriever()\n
    从数据文件夹内读取数据文件, 转为向量存入向量库load_document
    """
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"])
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len
        )

        self.documents: list[Document] = []

    def get_documents(self) -> list[Document]:
        if self.documents:
            return self.documents
        results = self.vector_store.get()
        if results["documents"]:
            self.documents = [
                Document(page_content=content, metadata=meta)
                for content, meta in zip(results["documents"], results["metadatas"])
            ]
        return self.documents

    def get_retriever(self):
        """
        获取向量库的检索器
        :return: 向量库的检索器
        """
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
         从数据文件夹内读取数据文件, 转为向量存入向量库
         要计算文件的md5做去重
        :return:None
        """

        def check_md5_hex(md5_for_check: str):
            """
            检查md5是否被处理过
            :param md5_for_check: 需要被检查的字符串
            :return:
            """
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r"):
                # 遍历
                for line in open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8"):
                    if md5_for_check in line:
                        return True     # md5 处理过

                return  False           # md5 没处理过

        def save_md5_hex(md5_for_save: str):
            """
            保存md5至文件内
            :param md5_for_save: 需要被保存的md5
            :return: None
            """
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_save + "\n")

        def get_file_documents(read_path: str):
            """
            根据路径, 获取文件内容, 转成doc文档
            :param read_path: 文件路径
            :return:
            """
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        # 提供一个文件夹路径，找出来指定类型的文件，返回它们的完整路径
        allowed_file_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        # 通过完整的路径, 存入向量库, 保存md5
        for path in allowed_file_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]文件{path}已处理过, 跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]文件{path}为空, 跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]文件{path}为空, 跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 将分块文档加入本地追踪列表，供 BM25 等混合检索复用
                self.documents.extend(split_document)

                # 记录这个已经处理好的md5, 避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]文件{path}处理完成")

            except Exception as e:
                # exc_info=True 会记录详细的报错堆栈, 如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]文件{path}处理失败, {str(e)}", exc_info=True)



if __name__ == '__main__':

    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")

    for r in res:
        print(r.page_content)
        print("-"*20)

