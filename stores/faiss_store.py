from typing import List, Tuple, Any
import numpy as np
from langchain_core.documents import Document
from core.store import AbstractVectorStore
from core.registry import VectorStoreRegistry


@VectorStoreRegistry.register("faiss", "FAISS (FlatL2)")
class FaissStore(AbstractVectorStore):
    def __init__(self):
        self.index = None
        self.docs: List[Document] = []
        self.embeddings_model = None
        self._index_type = "flat_l2"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import faiss
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
        import faiss

        index_type = kwargs.get("index_type", "flat_l2")

        instance = cls()
        instance.docs = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)
        ]
        instance.embeddings_model = embeddings
        instance._index_type = index_type

        data = np.ascontiguousarray(vecs, dtype=np.float32)

        if index_type == "flat_l2":
            index = faiss.IndexFlatL2(embed_dim)
            index.add(data)

        elif index_type == "flat_ip":
            # Normalize for cosine-like inner product
            faiss.normalize_L2(data)
            index = faiss.IndexFlatIP(embed_dim)
            index.add(data)

        elif index_type == "hnsw":
            hnsw_m = int(kwargs.get("hnsw_m", 32))
            ef_construction = int(kwargs.get("efConstruction", 200))
            ef_search = int(kwargs.get("efSearch", 128))
            index = faiss.IndexHNSWFlat(embed_dim, hnsw_m)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            index.add(data)

        elif index_type == "ivf_pq":
            nlist = int(kwargs.get("nlist", 100))
            m = int(kwargs.get("m", 8))
            nbits = int(kwargs.get("nbits", 8))
            nprobe = int(kwargs.get("nprobe", 10))

            # Ensure enough data for training
            nlist = min(nlist, len(docs))

            quantizer = faiss.IndexFlatL2(embed_dim)
            index = faiss.IndexIVFPQ(quantizer, embed_dim, nlist, m, nbits)
            index.train(data)
            index.add(data)
            index.nprobe = nprobe

        elif index_type == "sq":
            qtype_name = kwargs.get("qtype", "QT_8bit")
            # Map string names to FAISS scalar quantizer types
            sq_types = {
                "QT_8bit": faiss.ScalarQuantizer.QT_8bit,
                "QT_4bit": faiss.ScalarQuantizer.QT_4bit,
                "QT_fp16": faiss.ScalarQuantizer.QT_fp16,
                "QT_6bit": faiss.ScalarQuantizer.QT_6bit,
                "QT_8bit_uniform": faiss.ScalarQuantizer.QT_8bit_uniform,
                "QT_4bit_uniform": faiss.ScalarQuantizer.QT_4bit_uniform,
                "QT_8bit_direct": faiss.ScalarQuantizer.QT_8bit_direct,
            }
            qtype = sq_types.get(qtype_name)
            if qtype is None:
                raise ValueError(
                    f"Unknown FAISS SQ qtype '{qtype_name}'. "
                    f"Choose from: {', '.join(sq_types.keys())}"
                )
            index = faiss.IndexScalarQuantizer(embed_dim, qtype)
            index.train(data)
            index.add(data)

        else:
            raise ValueError(
                f"Unknown FAISS index_type '{index_type}'. "
                f"Choose from: flat_l2, flat_ip, hnsw, ivf_pq, sq"
            )

        instance.index = index
        return instance

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        query_vec = np.array(
            self.embeddings_model.embed_query(query), dtype=np.float32
        ).reshape(1, -1)

        # Normalize query vector for inner-product indices
        if self._index_type == "flat_ip":
            import faiss
            faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            results.append((self.docs[idx], float(dist)))
        return results

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        index_type = kwargs.get("index_type", "flat_l2")
        if index_type in ("flat_l2", "flat_ip", "hnsw"):
            # Full float32 vectors stored
            return (embed_dim * 4 * num_docs) / 1e6
        elif index_type == "ivf_pq":
            # PQ: m sub-quantizers × nbits per code
            m = int(kwargs.get("m", 8))
            nbits = int(kwargs.get("nbits", 8))
            bytes_per_vec = (m * nbits + 7) // 8
            return (bytes_per_vec * num_docs) / 1e6
        elif index_type == "sq":
            qtype = kwargs.get("qtype", "QT_8bit")
            bits_map = {
                "QT_8bit": 8, "QT_4bit": 4, "QT_fp16": 16,
                "QT_6bit": 6, "QT_8bit_uniform": 8,
                "QT_4bit_uniform": 4, "QT_8bit_direct": 8,
            }
            bits = bits_map.get(qtype, 8)
            return (embed_dim * bits * num_docs) / (8 * 1e6)
        else:
            return (embed_dim * 4 * num_docs) / 1e6
