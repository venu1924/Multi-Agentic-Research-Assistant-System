import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agents.analysis_reasoning_agent import analyze_multiple
from agents.query_domain_agent import refine_query
from agents.ranking_agent import rank_papers
from agents.retrieval_filter_agent import retrieve_papers
from agents.synthesis_insight_agent import generate_insights


def compute_similarity(papers):
    if not papers:
        return np.array([])
    if len(papers) == 1:
        return np.array([[1.0]])

    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [
            f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()
            for paper in papers
        ]
        embeddings = model.encode(texts)
        return np.round(cosine_similarity(embeddings), 2)
    except Exception as exc:
        print("Similarity error:", exc)
        return np.eye(len(papers))


def _normalize_paper(paper):
    normalized = dict(paper)
    normalized.setdefault("title", "Untitled")
    normalized.setdefault("abstract", "")
    normalized.setdefault("year", "")
    normalized.setdefault("authors", [])
    normalized.setdefault("pdf_url", None)
    return normalized

def _is_uploaded_paper(paper):
    title = paper.get("title", "")
    authors = paper.get("authors", [])
    return title.startswith("[Uploaded]") or "User Uploaded" in authors



def run_pipeline(query, max_papers=5, time_filter="All", focus="general", papers=None):
    retrieval_query = refine_query(query)

    if papers is None:
        candidate_papers = retrieve_papers(
            retrieval_query,
            max_results=max_papers * 3,
            time_filter=time_filter,
        )
    else:
        candidate_papers = [_normalize_paper(paper) for paper in papers]

    if not candidate_papers:
        return {
            "papers": [],
            "analyses": [],
            "insights": "",
            "comparison": [],
            "gaps": "",
            "recommendations": "",
            "similarity": np.array([]),
        }

    ranking_query = f"{query} {focus}".strip()

    uploaded_papers = [paper for paper in candidate_papers if _is_uploaded_paper(paper)]
    retrieved_papers = [paper for paper in candidate_papers if not _is_uploaded_paper(paper)]

    if uploaded_papers:
        ranked = rank_papers(ranking_query, retrieved_papers, max_papers) if retrieved_papers else []
        for uploaded in uploaded_papers:
            uploaded["score"] = uploaded.get("score", 1.0)
            ranked.append(uploaded)
        for index, paper in enumerate(ranked, start=1):
            paper["rank"] = index
    else:
        ranked = rank_papers(ranking_query, candidate_papers, max_papers)

    analyses = analyze_multiple(ranked)

    for index, paper in enumerate(ranked):
        paper["fields"] = analyses[index] if index < len(analyses) else {}

    comparison, insights, gaps, recommendations = generate_insights(analyses)
    similarity = compute_similarity(ranked)

    return {
        "papers": ranked,
        "analyses": analyses,
        "insights": insights,
        "comparison": comparison,
        "gaps": gaps,
        "recommendations": recommendations,
        "similarity": similarity,
    }
