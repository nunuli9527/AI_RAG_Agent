"""
RAG 总结问答服务（查资料）
作用：根据用户问题，去向量库查资料，将提问和参考资料发给模型，生成总结回答。
功能：
调用向量库检索文档
拼接参考资料
调用大模型生成回答
给 Agent 提供 rag_summerize 工具能力
"""

from langchain_core.output_parsers import StrOutputParser

from rag.vector_store import VectorStoreService
from rag.bm25_RRF import BM25HybridSearch
from rag.reranker import RerankerService
from utils.prompt_loader import load_rag_prompt
from utils.config_handler import chroma_conf
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


class RagSummarizeService():
    """
    总结服务: 用户提问, 搜索参考资料, 将提问和参考资料发给模型, 让模型总结回复
    """
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.vector_retriever = self.vector_store.get_retriever()

        # 确保文档已加载（md5去重保证不会重复处理）
        self.vector_store.load_document()

        # 初始化BM25混合检索，复用向量库已有的文档分块
        self.hybrid_search = BM25HybridSearch(documents=self.vector_store.get_documents())

        # 初始化Reranker精排服务
        self.reranker = RerankerService()

        # 粗排候选数：RRF融合后保留多少条给Reranker精排
        self.candidate_k = chroma_conf.get("rerank_candidate_k", 20)

        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        初始化链
        :return: 链
        """
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str):
        # 第一步：RRF 粗排 → 召回 Top-N 候选
        candidates = self.hybrid_search.hybrid_search(
            query, self.vector_retriever, top_k=self.candidate_k
        )
        # 第二步：Reranker 精排 → 取 Top-5 最终结果
        return self.reranker.rerank(query, candidates, top_k=5)

    def rag_summarize(self, query: str) -> str:
        """
        总结
        :param query: 用户问题
        :return: 总结结果
        """
        context_docs = self.retriever_docs(query)

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
