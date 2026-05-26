import uuid
from typing import List, Tuple, Any
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry


@VectorStoreRegistry.register("qdrant", "Qdrant (in-mem)")
class QdrantStore(AbstractVectorStore):
    def __init__(self):
        self.store = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            from qdrant_client import QdrantClient

            return True
        except ImportError:
            return False

    @classmethod
    def build(
        cls,
        docs: List[Document],
        embeddings: Any,
        vecs: Any,
        texts: List[str],
        metadatas: List[dict],
        embed_dim: int,
        **kwargs,
    ) -> "AbstractVectorStore":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.embeddings import Embeddings

        col = f"bench_{uuid.uuid4().hex[:8]}"
        client = QdrantClient(":memory:")
        client.create_collection(
            col, vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE)
        )

        precomputed = vecs.tolist()

        class MockEmbed(Embeddings):
            def embed_documents(self, t):
                return precomputed

            def embed_query(self, q):
                return embeddings.embed_query(q)  # real embeddings for queries

        instance = cls()
        instance.store = QdrantVectorStore(
            client=client, collection_name=col, embedding=MockEmbed()
        )
        instance.store.add_texts(texts, metadatas=metadatas)
        # Removed: instance.store.embeddings = embeddings  ← was the bug
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        return (embed_dim * 4 * num_docs) / 1e6
