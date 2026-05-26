from typing import List, Tuple, Any
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry

@VectorStoreRegistry.register("baseline", "Baseline (InMem)")
class BaselineStore(AbstractVectorStore):
    def __init__(self):
        self.store = None

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def build(cls, docs: List[Document], embeddings: Any, vecs: Any, texts: List[str], metadatas: List[dict], embed_dim: int, **kwargs) -> "AbstractVectorStore":
        instance = cls()
        instance.store = InMemoryVectorStore(embeddings)
        instance.store.add_texts(texts, embeddings=vecs.tolist(), metadatas=metadatas)
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        return (embed_dim * 4 * num_docs) / 1e6
