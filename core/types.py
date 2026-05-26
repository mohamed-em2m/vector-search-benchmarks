from dataclasses import dataclass
from typing import Any, List, Dict

@dataclass(frozen=True)
class TestQuery:
    query: str
    gold_keywords: List[str]
    description: str

@dataclass(frozen=True)
class SearchResult:
    doc_content: str
    metadata: Dict[str, Any]
    score: float

@dataclass(frozen=True)
class BuildStats:
    index_time_s: float
    net_rss_mb: float
    peak_rss_mb: float
    theoretical_mb: float
    docs_per_sec: float
