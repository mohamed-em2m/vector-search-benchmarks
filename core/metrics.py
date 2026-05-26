import math
import statistics
import numpy as np
from scipy.stats import kendalltau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from langchain_core.documents import Document
from typing import List

def is_relevant(doc: Document, gold_keywords: List[str]) -> bool:
    if not gold_keywords:
        return False
    combined = (doc.page_content + " " + str(doc.metadata)).lower()
    return any(kw.lower() in combined for kw in gold_keywords)

def recall_at_k(results, gold_keywords, k):
    hits = sum(1 for doc, _ in results[:k] if is_relevant(doc, gold_keywords))
    total_relevant = max(1, sum(1 for doc, _ in results if is_relevant(doc, gold_keywords)))
    return hits / total_relevant

def precision_at_k(results, gold_keywords, k):
    if k == 0:
        return 0.0
    return sum(1 for doc, _ in results[:k] if is_relevant(doc, gold_keywords)) / k

def score_stats(scores: List[float]) -> dict:
    if not scores:
        return {}
    return {
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
        "min": min(scores),
        "max": max(scores),
        "gap_top2": scores[0] - scores[1] if len(scores) > 1 else float("nan"),
    }

def kendall_tau(baseline_contents, other_contents):
    common = [c for c in baseline_contents if c in other_contents]
    if len(common) < 2:
        return float("nan")
    rank_b = [baseline_contents.index(c) for c in common]
    rank_o = [other_contents.index(c) for c in common]
    tau, _ = kendalltau(rank_b, rank_o)
    return tau

def jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)

def token_overlap_pct(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    return jaccard_similarity(tokens_a, tokens_b) * 100

def tfidf_cosine_similarity(texts_a: List[str], texts_b: List[str]) -> float:
    sims = []
    for a, b in zip(texts_a, texts_b):
        if not a.strip() or not b.strip():
            continue
        try:
            vec = TfidfVectorizer().fit_transform([a, b])
            sims.append(cos_sim(vec[0], vec[1])[0][0])
        except Exception:
            pass
    return (statistics.mean(sims) * 100) if sims else float("nan")

def score_scale_similarity(b_scores, t_scores) -> float:
    if len(b_scores) < 2 or len(t_scores) < 2:
        return float("nan")
    n = min(len(b_scores), len(t_scores))
    b_arr = np.array(b_scores[:n], dtype=float)
    t_arr = np.array(t_scores[:n], dtype=float)
    if b_arr.std() == 0 or t_arr.std() == 0:
        return 100.0 if np.allclose(b_arr, t_arr) else float("nan")
    corr = float(np.corrcoef(b_arr, t_arr)[0, 1])
    return max(0.0, corr) * 100

def rank_position_similarity(b_contents, t_contents) -> float:
    common = [c for c in b_contents if c in t_contents]
    if not common:
        return 0.0
    k = max(len(b_contents), len(t_contents), 1)
    diffs = [abs(b_contents.index(c) - t_contents.index(c)) for c in common]
    mean_diff = statistics.mean(diffs)
    max_possible = k - 1 if k > 1 else 1
    return max(0.0, (1 - mean_diff / max_possible)) * 100

def compute_similarity_report(b_results, t_results) -> dict:
    b_contents = [d.page_content for d, _ in b_results]
    t_contents = [d.page_content for d, _ in t_results]
    b_scores = [s for _, s in b_results]
    t_scores = [s for _, s in t_results]

    set_jaccard = jaccard_similarity(set(b_contents), set(t_contents)) * 100
    top1_tok = (token_overlap_pct(b_contents[0], t_contents[0]) if b_contents and t_contents else float("nan"))
    tfidf_cos = tfidf_cosine_similarity(b_contents, t_contents)
    score_corr = score_scale_similarity(b_scores, t_scores)
    rank_acc = rank_position_similarity(b_contents, t_contents)

    components = [v for v in [set_jaccard, top1_tok, tfidf_cos, score_corr, rank_acc] if not (isinstance(v, float) and math.isnan(v))]
    overall = statistics.mean(components) if components else float("nan")

    return {
        "result_set_jaccard_%": set_jaccard,
        "top1_token_overlap_%": top1_tok,
        "tfidf_cosine_%": tfidf_cos,
        "score_corr_%": score_corr,
        "rank_position_acc_%": rank_acc,
        "overall_similarity_%": overall,
    }
