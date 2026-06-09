"""
Rerank 重排序服务
作用：对粗排候选文档做精排，提升检索精度。
流程：RRF 融合粗排 → 取 Top-N 候选 → Reranker 精排 → 取 Top-K 最终结果
使用 DashScope text-rerank API（gte-rerank 模型）。
"""

from typing import List
from dashscope import TextReRank
from langchain_core.documents import Document
from utils.config_handler import rag_conf
from utils.logger_handler import logger


class RerankerService:
    """
    重排序服务：用 DashScope rerank API 对候选文档精排。
    输入 query + 候选文档列表，返回按相关性降序排列的 Top-K 文档。
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or rag_conf.get("rerank_model_name", "gte-rerank")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        对候选文档重排序，返回 Top-K 结果。

        :param query:   用户查询
        :param documents: 候选文档列表（粗排结果）
        :param top_k:   最终返回的文档数量
        :return:        重排后的 Top-K 文档
        """
        if not documents:
            logger.warning("[Reranker] 候选文档为空，跳过重排")
            return []

        # 如果候选数 <= top_k，不需要调用 rerank API
        if len(documents) <= top_k:
            logger.info(f"[Reranker] 候选文档数({len(documents)})<=Top-K({top_k})，直接返回")
            return documents

        # 提取文档文本列表
        doc_texts = [doc.page_content for doc in documents]

        try:
            response = TextReRank.call(
                model=self.model_name,
                query=query,
                documents=doc_texts,
                top_n=top_k,
            )

            if response.status_code != 200:
                logger.error(
                    f"[Reranker] API 返回错误: code={response.status_code}, "
                    f"message={response.message}"
                )
                return self._fallback(documents, top_k)

            # 按 API 返回的 index 和 relevance_score 重建文档列表
            reranked = []
            for item in response.output.results:
                idx = item.index
                score = item.relevance_score
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    doc.metadata["rerank_score"] = round(score, 4)
                    reranked.append(doc)

            logger.info(
                f"[Reranker] 重排完成: {len(documents)}条候选 → {len(reranked)}条精排结果"
            )
            return reranked

        except Exception as e:
            logger.error(f"[Reranker] 重排异常，降级为原始粗排结果: {str(e)}", exc_info=True)
            return self._fallback(documents, top_k)

    def _fallback(self, documents: List[Document], top_k: int) -> List[Document]:
        """
        降级：直接截取粗排的前 Top-K 条。
        """
        logger.info(f"[Reranker] 降级：截取粗排前{min(top_k, len(documents))}条")
        return documents[:top_k]


if __name__ == "__main__":
    from langchain_core.documents import Document

    # ======== 测试1: 基本重排功能 ========
    print("=" * 60)
    print("测试1: 基本重排功能")
    print("=" * 60)

    candidates = [
        Document(page_content="今天天气真好，适合出去散步。"),
        Document(page_content="扫地机器人可以自动清扫地面，解放双手。"),
        Document(page_content="最新款智能手机搭载了AI芯片。"),
        Document(page_content="科沃斯扫地机器人具有激光导航和自动集尘功能。"),
        Document(page_content="股票市场今天大幅上涨。"),
        Document(page_content="石头科技的扫地机器人以算法见长，避障能力强。"),
        Document(page_content="如何选择一台好的笔记本电脑？"),
    ]

    reranker = RerankerService()
    query = "扫地机器人哪个品牌好？"
    results = reranker.rerank(query, candidates, top_k=3)

    print(f'\n查询: "{query}"')
    print(f"重排后 Top-{len(results)}:")
    for i, doc in enumerate(results):
        score = doc.metadata.get("rerank_score", "N/A")
        print(f"  [{i}] score={score} | {doc.page_content[:60]}...")

    # ======== 测试2: 边界情况 ========
    print("\n" + "=" * 60)
    print("测试2: 边界情况")
    print("=" * 60)

    # 空文档列表
    empty_result = reranker.rerank(query, [], top_k=3)
    print(f"空文档列表 → 返回{len(empty_result)}条")

    # 候选数 <= top_k
    few_docs = candidates[:2]
    few_result = reranker.rerank(query, few_docs, top_k=3)
    print(f"候选数({len(few_docs)})<=top_k(3) → 返回{len(few_result)}条(不调API)")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
