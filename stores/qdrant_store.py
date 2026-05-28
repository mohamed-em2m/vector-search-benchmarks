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
            from langchain_qdrant import QdrantVectorStore

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
        from qdrant_client.models import Distance, VectorParams, PointStruct
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.embeddings import Embeddings

        quantization = kwargs.get("quantization", None)
        print(f"Building Qdrant store with quantization={quantization}")

        quantization_config = None

        if quantization == "scalar":
            from qdrant_client.models import (
                ScalarQuantization,
                ScalarQuantizationConfig,
                ScalarType,
            )

            scalar_type_str = str(kwargs.get("scalar_type", "int8")).lower()
            scalar_type = (
                ScalarType.INT8 if scalar_type_str == "int8" else ScalarType.INT8
            )
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
                "x4": CompressionRatio.X4,
                "x8": CompressionRatio.X8,
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
        elif quantization == "binary":
            from qdrant_client.models import (
                BinaryQuantization,
                BinaryQuantizationConfig,
                BinaryQuantizationEncoding,
                BinaryQuantizationQueryEncoding,
            )

            encoding_str = str(kwargs.get("binary_encoding", "one_bit")).lower()
            encoding = None
            if encoding_str in ("two_bits", "2_bits", "2bit", "2bits"):
                encoding = BinaryQuantizationEncoding.TWO_BITS
            elif encoding_str in ("one_and_half_bits", "1.5_bits", "1.5bit", "1.5bits"):
                encoding = BinaryQuantizationEncoding.ONE_AND_HALF_BITS

            query_encoding_str = str(
                kwargs.get("binary_query_encoding", "default")
            ).lower()
            query_encoding = None
            if query_encoding_str == "binary":
                query_encoding = BinaryQuantizationQueryEncoding.BINARY
            elif query_encoding_str == "scalar8bits":
                query_encoding = BinaryQuantizationQueryEncoding.SCALAR8BITS
            elif query_encoding_str == "scalar4bits":
                query_encoding = BinaryQuantizationQueryEncoding.SCALAR4BITS

            quantization_config = BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=bool(kwargs.get("binary_always_ram", True)),
                    encoding=encoding,
                    query_encoding=query_encoding,
                )
            )
        # 1. Guard the import properly
        elif quantization == "turbo":
            try:
                from qdrant_client.models import (
                    TurboQuantization,
                    TurboQuantQuantizationConfig,
                    TurboQuantBitSize,
                )
            except ImportError:
                raise RuntimeError(
                    "TurboQuant requires qdrant-client >= 1.13.0 (Qdrant server 1.18+). "
                    "Run: pip install --upgrade qdrant-client"
                )

            bits_str = str(kwargs.get("turbo_bits", "bits4")).lower()
            # Map to the canonical API strings
            bit_map = {
                "bits4": TurboQuantBitSize.BITS4,
                "4": TurboQuantBitSize.BITS4,
                "4bit": TurboQuantBitSize.BITS4,
                "bits2": TurboQuantBitSize.BITS2,
                "2": TurboQuantBitSize.BITS2,
                "2bit": TurboQuantBitSize.BITS2,
                "bits1_5": TurboQuantBitSize.BITS1_5,
                "1.5": TurboQuantBitSize.BITS1_5,
                "1.5bit": TurboQuantBitSize.BITS1_5,
                "bits1": TurboQuantBitSize.BITS1,
                "1": TurboQuantBitSize.BITS1,
                "1bit": TurboQuantBitSize.BITS1,
            }
            bits = bit_map.get(bits_str, TurboQuantBitSize.BITS4)

            quantization_config = TurboQuantization(
                turbo=TurboQuantQuantizationConfig(
                    always_ram=bool(kwargs.get("turbo_always_ram", True)),
                    bits=bits,
                )
            )
        col = f"bench_{uuid.uuid4().hex[:8]}"
        client = QdrantClient(":memory:")
        create_kwargs = dict(
            collection_name=col,
            vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
        )
        if quantization_config is not None:
            create_kwargs["quantization_config"] = quantization_config
        client.create_collection(**create_kwargs)

        # upload pre-computed vectors directly via upsert
        precomputed = vecs.tolist()
        points = [
            PointStruct(
                id=i,
                vector=precomputed[i],
                # Nested 'metadata' dict aligns with the default LangChain-Qdrant payload schema
                payload={"page_content": texts[i], "metadata": metadatas[i]},
            )
            for i in range(len(texts))
        ]
        client.upsert(collection_name=col, points=points)

        class MockEmbed(Embeddings):
            def embed_documents(self, t):
                return embeddings.embed_documents(t)

            def embed_query(self, q):
                return embeddings.embed_query(q)

        instance = cls()
        instance._quantization = quantization
        instance._embed_dim = embed_dim
        instance._num_docs = len(docs)
        instance._scalar_always_ram = bool(kwargs.get("scalar_always_ram", True))
        instance._kwargs = kwargs
        instance.store = QdrantVectorStore(
            client=client, collection_name=col, embedding=MockEmbed()
        )
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        from qdrant_client.models import SearchParams, QuantizationSearchParams

        search_params = None
        if self._quantization is not None:
            # rescore=True ensures quantized candidates are re-ranked using original vectors
            search_params = SearchParams(
                quantization=QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                )
            )

        return self.store.similarity_search_with_score(
            query, k=k, search_params=search_params
        )

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        quantization = kwargs.get("quantization", None)
        if quantization == "scalar":
            quantized = embed_dim * 1 * num_docs
            if bool(kwargs.get("scalar_always_ram", True)):
                original = embed_dim * 4 * num_docs
                return (quantized + original) / 1e6
            return quantized / 1e6
        elif quantization == "product":
            compression_map = {"x4": 4, "x8": 8, "x16": 16, "x32": 32, "x64": 64}
            ratio = compression_map.get(
                str(kwargs.get("pq_compression", "x16")).lower(), 16
            )
            quantized = (embed_dim * 4 * num_docs) / ratio
            if bool(kwargs.get("pq_always_ram", True)):
                original = embed_dim * 4 * num_docs
                return (quantized + original) / 1e6
            return quantized / 1e6
        elif quantization == "binary":
            encoding_str = str(kwargs.get("binary_encoding", "one_bit")).lower()
            bits = 1
            if encoding_str in ("two_bits", "2_bits", "2bit", "2bits"):
                bits = 2
            elif encoding_str in ("one_and_half_bits", "1.5_bits", "1.5bit", "1.5bits"):
                # 1.5-bit binary achieves a physical 24x compression ratio of float32,
                # which translates to exactly 1.333333 bits per dimension (32 / 24)
                bits = 4 / 3
            quantized = (embed_dim * bits * num_docs) / 8
            if bool(kwargs.get("binary_always_ram", True)):
                original = embed_dim * 4 * num_docs
                return (quantized + original) / 1e6
            return quantized / 1e6
        elif quantization == "turbo":
            bits_str = str(kwargs.get("turbo_bits", "bits4")).lower()
            bits = 4
            if bits_str in ("bits2", "2", "2bit", "2bits"):
                bits = 2
            elif bits_str in ("bits1_5", "1.5", "1.5bit", "1.5bits"):
                # TurboQuant 1.5-bit uses exactly 1.5 bits per dimension
                bits = 1.5
            elif bits_str in ("bits1", "1", "1bit", "1bits"):
                bits = 1
            quantized = (embed_dim * bits * num_docs) / 8
            if bool(kwargs.get("turbo_always_ram", True)):
                original = embed_dim * 4 * num_docs
                return (quantized + original) / 1e6
            return quantized / 1e6

        # No quantization — full float32
        return (embed_dim * 4 * num_docs) / 1e6
