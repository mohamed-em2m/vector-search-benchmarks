import uuid
from typing import List, Tuple, Any
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry


@VectorStoreRegistry.register("qdrant", "Qdrant (in-mem)")
class QdrantStore(AbstractVectorStore):
    def __init__(self):
        self.store = None
        self._quantization = None
        self._embed_dim = 0
        self._num_docs = 0

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

        # ── Quantization config ───────────────────────────────────────────
        # quantization: "scalar" (default None = no quantization)
        #               "product"
        # scalar options:
        #   scalar_type: "int8" (default)
        #   scalar_quantile: 0.99 (default)
        #   scalar_always_ram: true (default)
        # product options:
        #   pq_compression: "x4" | "x8" | "x16" | "x32" | "x64" (default "x16")
        #   pq_always_ram: true (default)
        quantization = kwargs.get("quantization", None)
        quantization_config = None

        if quantization == "scalar":
            from qdrant_client.models import (
                ScalarQuantization,
                ScalarQuantizationConfig,
                ScalarType,
            )
            scalar_type_str = str(kwargs.get("scalar_type", "int8")).lower()
            scalar_type = ScalarType.INT8 if scalar_type_str == "int8" else ScalarType.INT8
            quantization_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=scalar_type,
                    quantile=float(kwargs.get("scalar_quantile", 0.99)),
                    always_ram=bool(kwargs.get("scalar_always_ram", True)),
                )
            )

        elif quantization == "product":
            from qdrant_client.models import (
                ProductQuantization,
                ProductQuantizationConfig,
                CompressionRatio,
            )
            compression_map = {
                "x4":  CompressionRatio.X4,
                "x8":  CompressionRatio.X8,
                "x16": CompressionRatio.X16,
                "x32": CompressionRatio.X32,
                "x64": CompressionRatio.X64,
            }
            ratio_str = str(kwargs.get("pq_compression", "x16")).lower()
            compression = compression_map.get(ratio_str, CompressionRatio.X16)
            quantization_config = ProductQuantization(
                product=ProductQuantizationConfig(
                    compression=compression,
                    always_ram=bool(kwargs.get("pq_always_ram", True)),
                )
            )

        # ── Create collection ─────────────────────────────────────────────
        col = f"bench_{uuid.uuid4().hex[:8]}"
        client = QdrantClient(":memory:")

        create_kwargs = dict(
            collection_name=col,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        if quantization_config is not None:
            create_kwargs["quantization_config"] = quantization_config

        client.create_collection(**create_kwargs)

        # ── Ingest with pre-computed vectors ─────────────────────────────
        precomputed = vecs.tolist()
        text_to_vec = {t: v for t, v in zip(texts, precomputed)}

        class MockEmbed(Embeddings):
            def embed_documents(self, t):
                return [text_to_vec.get(x) or embeddings.embed_query(x) for x in t]

            def embed_query(self, q):
                return embeddings.embed_query(q)

        instance = cls()
        instance._quantization = quantization
        instance._embed_dim = embed_dim
        instance._num_docs = len(docs)
        instance.store = QdrantVectorStore(
            client=client, collection_name=col, embedding=MockEmbed()
        )
        instance.store.add_texts(texts, metadatas=metadatas)
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        from qdrant_client.models import SearchParams, QuantizationSearchParams

        search_params = None
        if self._quantization is not None:
            search_params = SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=False,
                )
            )
        return self.store.similarity_search_with_score(
            query, k=k, search_params=search_params
        )

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        quantization = kwargs.get("quantization", None)
        if quantization == "scalar":
            # INT8: 1 byte per dimension instead of 4
            return (embed_dim * 1 * num_docs) / 1e6
        elif quantization == "product":
            compression_map = {"x4": 4, "x8": 8, "x16": 16, "x32": 32, "x64": 64}
            ratio = compression_map.get(str(kwargs.get("pq_compression", "x16")).lower(), 16)
            return (embed_dim * 4 * num_docs) / ratio / 1e6
        # No quantization — full float32
        return (embed_dim * 4 * num_docs) / 1e6
