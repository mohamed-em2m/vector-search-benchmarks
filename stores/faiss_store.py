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
        self._hnsw_m = 32

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
            # FIX 2: normalize a copy — do not mutate the caller's array
            data_ip = data.copy()
            faiss.normalize_L2(data_ip)
            index = faiss.IndexFlatIP(embed_dim)
            index.add(data_ip)

        elif index_type == "hnsw":
            hnsw_m = int(kwargs.get("hnsw_m", 32))
            ef_construction = int(kwargs.get("efConstruction", 200))
            ef_search = int(kwargs.get("efSearch", 128))
            index = faiss.IndexHNSWFlat(embed_dim, hnsw_m)
            index.hnsw.efConstruction = ef_construction
            index.hnsw.efSearch = ef_search
            index.add(data)
            instance._hnsw_m = hnsw_m  # store for theoretical_bytes

        elif index_type == "ivf_pq":
            nlist = int(kwargs.get("nlist", 100))
            m = int(kwargs.get("m", 8))
            nbits = int(kwargs.get("nbits", 8))
            nprobe = int(kwargs.get("nprobe", 10))

            # FIX 3: guard against nlist > num vectors using len(texts) consistently
            nlist = min(nlist, len(texts))

            quantizer = faiss.IndexFlatL2(embed_dim)
            index = faiss.IndexIVFPQ(quantizer, embed_dim, nlist, m, nbits)
            index.train(data)
            index.add(data)
            index.nprobe = nprobe

        elif index_type == "sq":
            qtype_name = kwargs.get("qtype", "QT_8bit")
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
        import faiss

        query_vec = np.array(
            self.embeddings_model.embed_query(query), dtype=np.float32
        ).reshape(1, -1)

        if self._index_type == "flat_ip":
            # FIX 2: normalize a copy here too
            query_vec = query_vec.copy()
            faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # FIX 5: unify score semantics — convert all to similarity in [0, 1]
            if self._index_type == "flat_l2":
                # L2 distance: 0 = identical; convert to similarity via 1/(1+d)
                score = 1.0 / (1.0 + float(dist))
            else:
                # Inner product / cosine on unit vecs: already in [-1, 1]
                score = float(np.clip((dist + 1.0) / 2.0, 0.0, 1.0))
            results.append((self.docs[idx], score))
        return results

    @classmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        index_type = kwargs.get("index_type", "flat_l2")

        if index_type in ("flat_l2", "flat_ip"):
            return (embed_dim * 4 * num_docs) / 1e6

        elif index_type == "hnsw":
            # FIX 1: HNSW stores float32 vectors + the graph (M neighbor IDs
            # per node stored as int32, with an extra layer-0 copy at 2×M)
            hnsw_m = int(kwargs.get("hnsw_m", 32))
            vector_bytes = embed_dim * 4 * num_docs
            # layer-0 has 2*M links, upper layers average ~M/2 total per node
            graph_bytes = (2 * hnsw_m + hnsw_m // 2) * 4 * num_docs
            return (vector_bytes + graph_bytes) / 1e6

        elif index_type == "ivf_pq":
            m = int(kwargs.get("m", 8))
            nbits = int(kwargs.get("nbits", 8))
            bytes_per_vec = (m * nbits + 7) // 8
            return (bytes_per_vec * num_docs) / 1e6

        elif index_type == "sq":
            qtype = kwargs.get("qtype", "QT_8bit")
            bits_map = {
                "QT_8bit": 8,
                "QT_4bit": 4,
                "QT_fp16": 16,
                "QT_6bit": 6,
                "QT_8bit_uniform": 8,
                "QT_4bit_uniform": 4,
                "QT_8bit_direct": 8,
            }
            bits = bits_map.get(qtype, 8)
            return (embed_dim * bits * num_docs) / (8 * 1e6)

        else:
            return (embed_dim * 4 * num_docs) / 1e6
