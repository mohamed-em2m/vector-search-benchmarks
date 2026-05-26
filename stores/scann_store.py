from typing import List, Tuple, Any
import numpy as np
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry


@VectorStoreRegistry.register("scann", "ScaNN (Google)")
class ScaNNStore(AbstractVectorStore):
    def __init__(self):
        self.searcher = None
        self.docs: List[Document] = []
        self.embeddings_model = None
        self._mode = "auto"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import scann
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
        import scann

        instance = cls()
        instance.docs = list(docs)
        instance.embeddings_model = embeddings

        mode = kwargs.get("mode", "auto")
        instance._mode = mode

        # Normalize vectors for cosine similarity (ScaNN dot_product on
        # unit vectors == cosine similarity)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (vecs / norms).astype(np.float32)

        num_docs = len(docs)

        # Resolve mode
        use_brute_force = (
            mode == "brute_force" or (mode == "auto" and num_docs < 1000)
        )

        if use_brute_force:
            instance.searcher = (
                scann.scann_ops_pybind.builder(normalized, 10, "dot_product")
                .score_brute_force()
                .build()
            )
        else:
            # Tree-AH mode — configurable partitioning
            num_leaves = int(kwargs.get("num_leaves", max(10, int(np.sqrt(num_docs)))))
            num_leaves_to_search = int(
                kwargs.get("num_leaves_to_search", max(5, num_leaves // 5))
            )
            reorder_num = int(kwargs.get("reorder_num", min(200, num_docs)))

            instance.searcher = (
                scann.scann_ops_pybind.builder(normalized, 10, "dot_product")
                .tree(
                    num_leaves=num_leaves,
                    num_leaves_to_search=num_leaves_to_search,
                    training_sample_size=min(num_docs, 250000),
                )
                .score_ah(2, anisotropic_quantization_threshold=0.2)
                .reorder(reorder_num)
                .build()
            )

        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        query_vec = np.array(
            self.embeddings_model.embed_query(query), dtype=np.float32
        )
        # Normalize query vector to match the indexed normalized vectors
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        indices, scores = self.searcher.search(query_vec, final_num_neighbors=k)

        results = []
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self.docs):
                results.append((self.docs[idx], float(score)))
        return results

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        mode = kwargs.get("mode", "auto")
        use_brute_force = (
            mode == "brute_force" or (mode == "auto" and num_docs < 1000)
        )
        if use_brute_force:
            return (embed_dim * 4 * num_docs) / 1e6
        else:
            # AH: 2 bytes per dim (quantized) + tree metadata overhead
            return (embed_dim * 2 * num_docs) / 1e6
