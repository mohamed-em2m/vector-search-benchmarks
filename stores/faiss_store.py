from typing import List, Tuple, Any
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry

@VectorStoreRegistry.register("faiss", "FAISS (FlatL2)")
class FaissStore(AbstractVectorStore):
    def __init__(self):
        self.store = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from langchain_community.vectorstores import FAISS
            return True
        except ImportError:
            return False

    @classmethod
    def build(cls, docs: List[Document], embeddings: Any, vecs: Any, texts: List[str], metadatas: List[dict], embed_dim: int, **kwargs) -> "AbstractVectorStore":
        from langchain_community.vectorstores import FAISS as LangchainFAISS
        instance = cls()
        instance.store = LangchainFAISS.from_embeddings(
            text_embeddings=list(zip(texts, vecs.tolist())),
            embedding=embeddings,
            metadatas=metadatas,
        )
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        return (embed_dim * 4 * num_docs) / 1e6
