from agents.query_domain_agent import refine_query
from agents.retrieval_filter_agent import retrieve_papers
from agents.ranking_agent import rank_papers
from agents.analysis_reasoning_agent import analyze_multiple
from agents.synthesis_insight_agent import generate_insights

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np


# =========================
# SIMILARITY
# =========================
def compute_similarity(papers):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [p["title"] + " " + p["abstract"] for p in papers]
    emb = model.encode(texts)

    sim = cosine_similarity(emb)
    return np.round(sim, 2)


# =========================
# TRENDS
# =========================
def compute_trends(papers):
    year_count = {}

    for p in papers:
        y = p.get("year", 0)
        year_count[y] = year_count.get(y, 0) + 1

    return sorted(year_count.items())


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(query, max_papers=5, time_filter="All"):

    retrieval_query = refine_query(query)

    papers = retrieve_papers(
        retrieval_query,
        max_results=max_papers * 3,
        time_filter=time_filter
    )

    if not papers:
        return {}

    ranked = rank_papers(query, papers, max_papers)

    # 🔥 ANALYSIS
    analyses = analyze_multiple(ranked)

    for i, p in enumerate(ranked):
        p["fields"] = analyses[i]

    # 🔥 INSIGHTS
    comparison, insights, gaps, recs = generate_insights(analyses)

    # 🔥 VISUALS
    similarity = compute_similarity(ranked)
    trends = compute_trends(ranked)

    return {
        "papers": ranked,
        "analyses": analyses,
        "insights": insights,
        "comparison": comparison,
        "gaps": gaps,
        "recommendations": recs,
        "similarity": similarity,
        "trends": trends
    }