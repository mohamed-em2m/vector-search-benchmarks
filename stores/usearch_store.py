from typing import List, Tuple, Any
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry

@VectorStoreRegistry.register("usearch", "USearch (HNSW)")
class USearchStore(AbstractVectorStore):
    def __init__(self):
        self.store = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from langchain_community.vectorstores import USearch
            return True
        except ImportError:
            return False

    @classmethod
    def build(cls, docs: List[Document], embeddings: Any, vecs: Any, texts: List[str], metadatas: List[dict], embed_dim: int, **kwargs) -> "AbstractVectorStore":
        from langchain_community.vectorstores import USearch
        from langchain_community.docstore.in_memory import InMemoryDocstore
        import usearch.index
        from langchain_core.embeddings import Embeddings

        # Configurable parameters
        metric = kwargs.get("metric", "cos")
        connectivity = kwargs.get("connectivity", None)
        expansion_add = kwargs.get("expansion_add", None)
        expansion_search = kwargs.get("expansion_search", None)

        # Build the USearch index with configurable params
        index_kwargs = {"ndim": embed_dim, "metric": metric}
        if connectivity is not None:
            index_kwargs["connectivity"] = int(connectivity)
        if expansion_add is not None:
            index_kwargs["expansion_add"] = int(expansion_add)
        if expansion_search is not None:
            index_kwargs["expansion_search"] = int(expansion_search)

        index = usearch.index.Index(**index_kwargs)
        instance = cls()
        instance.store = USearch(
            embedding=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            ids=[]
        )
        precomputed = vecs.tolist()
        text_to_vec = {t: v for t, v in zip(texts, precomputed)}
        class MockEmbed(Embeddings):
            def embed_documents(self, t): return [text_to_vec.get(x) or embeddings.embed_query(x) for x in t]
            def embed_query(self, q): return embeddings.embed_query(q)
        
        instance.store.embedding = MockEmbed()
        instance.store.add_texts(texts, metadatas=metadatas)
        instance.store.embedding = embeddings
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        return (embed_dim * 4 * num_docs) / 1e6
