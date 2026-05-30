from typing import List
import jieba
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def preprocessing_func(text: str) -> List[str]:
    return list(jieba.cut(text))


def rrf(vector_results_ids: List[int], text_results_ids: List[int], k: int = 10, m: int = 60) -> List[int]:
    doc_scores = {}
    for rank, doc_id in enumerate(vector_results_ids):
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m)
    for rank, doc_id in enumerate(text_results_ids):
        doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1 / (rank + m)
    sorted_results = [
        d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    ][:k]
    return sorted_results


class BM25HybridSearch:
    def __init__(self, documents: List[Document]):
        self.docs = documents
        self.bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            preprocess_func=preprocessing_func
        )
        self.bm25_retriever.k = 10

    def hybrid_search(self, query: str, vector_retriever, top_k: int = 5) -> List[Document]:
        vector_res: List[Document] = vector_retriever.invoke(query)
        bm25_res: List[Document] = self.bm25_retriever.invoke(query)

        # 用 page_content 作为文档块的标识建立反向索引
        doc_to_idx = {doc.page_content: i for i, doc in enumerate(self.docs)}

        vector_ids = []
        for doc in vector_res:
            idx = doc_to_idx.get(doc.page_content)
            if idx is not None:
                vector_ids.append(idx)

        bm25_ids = []
        for doc in bm25_res:
            idx = doc_to_idx.get(doc.page_content)
            if idx is not None:
                bm25_ids.append(idx)

        fused_ids = rrf(vector_ids, bm25_ids, k=top_k)
        return [self.docs[doc_id] for doc_id in fused_ids]


if __name__ == '__main__':
    from langchain_core.documents import Document

    # ======== 构造测试文档（模拟扫地机器人知识库分块） ========
    test_docs = [
        Document(page_content="小户型适合选择超薄扫地机器人，厚度小于8cm的机型可以轻松进入沙发和床底。"),
        Document(page_content="养宠物的家庭推荐选择大吸力扫地机器人，吸力至少3000Pa以上才能有效清理宠物毛发。"),
        Document(page_content="扫地机器人的电池容量决定了清扫面积，一般5200mAh电池可覆盖200平米户型。"),
        Document(page_content="激光导航比视觉导航精度更高，适合复杂户型；视觉导航成本更低，适合小户型。"),
        Document(page_content="扫拖一体机器人需要关注水箱容量，200ml以上水箱适合100平米以上的户型。"),
        Document(page_content="小米扫地机器人性价比高，科沃斯主打高端市场，石头科技在算法方面有优势。"),
        Document(page_content="扫地机器人噪音一般在55-70分贝之间，高端机型可控制在55分贝以下。"),
        Document(page_content="边刷和主刷需要定期清理，建议每周清理一次以防止毛发缠绕影响清扫效果。"),
        Document(page_content="故障排除：扫地机器人迷路通常是激光头脏了或传感器被遮挡，清理后重启即可。"),
        Document(page_content="选购时要关注吸力、电池、导航方式、噪音、水箱容量这五个核心参数。"),
    ]

    print("=" * 60)
    print("测试1: 验证 RRF 融合算法")
    print("=" * 60)

    # 模拟两路检索结果（数字代表文档索引）
    vector_ranking = [0, 3, 6, 8]       # 向量召回排序
    bm25_ranking = [9, 3, 0, 1, 5]     # BM25召回排序

    fused = rrf(vector_ranking, bm25_ranking, k=5, m=60)
    print(f"向量召回排名: {vector_ranking}")
    print(f"BM25召回排名:  {bm25_ranking}")
    print(f"RRF融合后排名: {fused}")
    print()

    # RRF 数学验证：
    # doc 0: 1/(0+60) + 1/(2+60) = 1/60 + 1/62 ≈ 0.03280  (向量第1 + BM25第3)
    # doc 3: 1/(1+60) + 1/(1+60) = 1/61 + 1/61 ≈ 0.03279  (向量第2 + BM25第2)
    # doc 0 得分略高于 doc 3，应排第一
    doc0_score = 1/60 + 1/62
    doc3_score = 1/61 + 1/61
    print(f"  doc 0 得分: 1/60 + 1/62 = {doc0_score:.5f}")
    print(f"  doc 3 得分: 1/61 + 1/61 = {doc3_score:.5f}")
    assert fused[0] == 0, f"预期文档0排名第一，实际{fused[0]}排名第一"
    assert fused[1] == 3, f"预期文档3排名第二，实际{fused[1]}排名第二"
    print("  RRF 数学逻辑验证通过")

    print()
    print("=" * 60)
    print("测试2: 初始化 BM25HybridSearch")
    print("=" * 60)

    hybrid = BM25HybridSearch(documents=test_docs)
    print(f"已索引文档数: {len(hybrid.docs)}")
    print(f"BM25检索器k值: {hybrid.bm25_retriever.k}")
    print()

    print("=" * 60)
    print("测试3: 纯BM25检索（不融合）")
    print("=" * 60)

    bm25_only = hybrid.bm25_retriever.invoke("小户型")
    print(f'查询"小户型" → BM25返回{len(bm25_only)}条:')
    for i, doc in enumerate(bm25_only):
        print(f"  [{i}] {doc.page_content[:60]}...")
    print()

    print("=" * 60)
    print("测试4: 混合检索（向量模拟 + BM25 融合）")
    print("=" * 60)

    # 构造一个模拟的向量检索器
    class MockVectorRetriever:
        def invoke(self, query: str) -> List[Document]:
            mapping = {
                "小户型": [test_docs[0], test_docs[3], test_docs[2]],
                "宠物":   [test_docs[1], test_docs[8], test_docs[7]],
                "故障":   [test_docs[8], test_docs[6], test_docs[5]],
                "选购":   [test_docs[9], test_docs[5], test_docs[2]],
            }
            return mapping.get(query, [])

    mock_retriever = MockVectorRetriever()

    for query in ["小户型", "宠物", "故障", "选购"]:
        results = hybrid.hybrid_search(query, mock_retriever, top_k=3)
        print(f'\n查询: "{query}" → 融合后Top-{len(results)}:')
        for i, doc in enumerate(results):
            print(f"  [{i}] {doc.page_content[:70]}...")

    print()
    print("=" * 60)
    print("测试5: 边界情况")
    print("=" * 60)

    # 空查询
    empty_results = hybrid.hybrid_search("", mock_retriever, top_k=3)
    print(f'空查询返回: {len(empty_results)}条')

    # 无匹配查询
    none_results = hybrid.hybrid_search("航天飞机", mock_retriever, top_k=3)
    print(f'"航天飞机"(无关查询)返回: {len(none_results)}条')

    # 向量召回为空的情况
    class EmptyVectorRetriever:
        def invoke(self, query: str) -> List[Document]:
            return []

    empty_hybrid = BM25HybridSearch(documents=test_docs[:3])
    results_no_vec = empty_hybrid.hybrid_search("小户型", EmptyVectorRetriever(), top_k=2)
    print(f'仅BM25(向量为空)返回: {len(results_no_vec)}条 → 应降级为纯BM25排序')

    # 文档集为空的情况
    try:
        empty_hybrid2 = BM25HybridSearch(documents=[])
        print("空文档集初始化: 未抛异常(可能BM25内部处理)")
    except Exception as e:
        print(f"空文档集初始化异常: {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print("全部测试完成")
    print("=" * 60)
