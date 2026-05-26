from abc import ABC, abstractmethod
from typing import List, Tuple, Any

from langchain_core.documents import Document

class AbstractVectorStore(ABC):
    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if the required dependencies are installed."""
        pass

    @classmethod
    @abstractmethod
    def build(cls, docs: List[Document], embeddings: Any, vecs: Any, texts: List[str], metadatas: List[dict], embed_dim: int, **kwargs) -> "AbstractVectorStore":
        """Build and return the store instance."""
        pass

    @abstractmethod
    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Perform a similarity search."""
        pass

    @classmethod
    @abstractmethod
    def theoretical_bytes(cls, embed_dim: int, num_docs: int, **kwargs) -> float:
        """Calculate theoretical memory usage in MB."""
        pass
