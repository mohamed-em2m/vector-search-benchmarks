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
        self._num_leaves = 0
        self._embed_dim = 0

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
        instance._embed_dim = embed_dim

        mode = kwargs.get("mode", "auto")
        instance._mode = mode

        # Normalize vectors for cosine similarity (ScaNN dot_product on
        # unit vectors == cosine similarity)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (vecs / norms).astype(np.float32)

        num_docs = len(docs)
        use_brute_force = mode == "brute_force" or (mode == "auto" and num_docs < 1000)

        if use_brute_force:
            instance.searcher = (
                scann.scann_ops_pybind.builder(normalized, 10, "dot_product")
                .score_brute_force()
                .build()
            )
            instance._num_leaves = 0
        else:
            num_leaves = int(kwargs.get("num_leaves", max(10, int(np.sqrt(num_docs)))))

            # FIX 3: num_leaves_to_search must be <= num_leaves; use 10% with a
            # floor of 1 and a ceiling of num_leaves (not a hard-coded 5 that
            # searches 50% of leaves when num_leaves is small)
            num_leaves_to_search = int(
                kwargs.get(
                    "num_leaves_to_search",
                    min(num_leaves, max(1, num_leaves // 10)),
                )
            )
            reorder_num = int(kwargs.get("reorder_num", min(200, num_docs)))

            instance.searcher = (
                scann.scann_ops_pybind.builder(normalized, 10, "dot_product")
                .tree(
                    num_leaves=num_leaves,
                    num_leaves_to_search=num_leaves_to_search,
                    training_sample_size=min(num_docs, 250000),
                )
                # dimensions_per_block=2 is a block-size param, not a byte width
                .score_ah(2, anisotropic_quantization_threshold=0.2)
                .reorder(reorder_num)
                .build()
            )
            instance._num_leaves = num_leaves

        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        query_vec = np.array(self.embeddings_model.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        indices, scores = self.searcher.search(query_vec, final_num_neighbors=k)

        results = []
        for idx, score in zip(indices, scores):
            if 0 <= idx < len(self.docs):
                # FIX 4: clamp dot-product score from [-1, 1] to [0, 1] so
                # callers get a consistent similarity range across all stores
                similarity = float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
                results.append((self.docs[idx], similarity))
        return results

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        mode = kwargs.get("mode", "auto")
        use_brute_force = mode == "brute_force" or (mode == "auto" and num_docs < 1000)

        if use_brute_force:
            return (embed_dim * 4 * num_docs) / 1e6
        else:
            # FIX 1: AH quantizes to uint8 = 1 byte/dim, not 2
            # Also account for tree centroids: num_leaves × embed_dim × float32
            num_leaves = int(kwargs.get("num_leaves", max(10, int(np.sqrt(num_docs)))))
            quantized_bytes = embed_dim * 1 * num_docs
            centroid_bytes = num_leaves * embed_dim * 4
            return (quantized_bytes + centroid_bytes) / 1e6
