from typing import List, Tuple, Any
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry

@VectorStoreRegistry.register("turbovec", "TurboVec (3bit)")
class TurboVecStore(AbstractVectorStore):
    def __init__(self):
        self.store = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from turbovec.langchain import TurboQuantVectorStore
            return True
        except ImportError:
            return False

    @classmethod
    def build(cls, docs: List[Document], embeddings: Any, vecs: Any, texts: List[str], metadatas: List[dict], embed_dim: int, **kwargs) -> "AbstractVectorStore":
        from turbovec.langchain import TurboQuantVectorStore
        bit_width = kwargs.get("bit_width", 3)
        instance = cls()
        instance.store = TurboQuantVectorStore.from_documents(
            documents=docs, embedding=embeddings, bit_width=bit_width
        )
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        bit_width = kwargs.get("bit_width", 3)
        bytes_per_vec = (embed_dim * bit_width + 7) // 8
        return (bytes_per_vec * num_docs) / 1e6
