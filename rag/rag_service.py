"""
RAG 总结问答服务（查资料）
作用：根据用户问题，去向量库查资料，将提问和参考资料发给模型，生成总结回答。
功能：
调用向量库检索文档
拼接参考资料
调用大模型生成回答
给 Agent 提供 rag_summerize 工具能力
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


def print_prompt(prompt):
    print("-"*20)
    print(prompt.to_string())
    print("-"*20)
    return prompt

class RagSummarizeService():
    """
    总结服务: 用户提问, 搜索参考资料, 将提问和参考资料发给模型, 让模型总结回复
    """
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        初始化链
        :return: 链
        """
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[str]:
        """
        通过向量库搜索内容
        :param query: 用户问题
        :return: 搜索到的内容
        """
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> str:
        """
        总结
        :param query: 用户问题
        :return: 总结结果
        """
        context_docs = self.retriever_docs(query)

        context = ""
        # counter = 0
        # for doc in context_docs:
        #     counter += 1
        #     context += f"[参考资料{counter}]: 参考资料: {doc.page_content} | 参考元数据: {doc.metadata}\n"

        # 用换行分开，每个文档块单独一行
        context = "\n".join([doc.page_content for doc in context_docs])

        return self.chain.invoke(
            {
                "input": query,
                "context": context
            }
        )


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
