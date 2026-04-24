import arxiv
from datetime import datetime


# =========================
# KEYWORD EXTRACTION
# =========================
def extract_keywords(query):
    return [w for w in query.lower().split() if len(w) > 3][:4]


# =========================
# BUILD ARXIV QUERY
# =========================
def build_arxiv_query(query):

    keywords = extract_keywords(query)

    if not keywords:
        return f'all:"{query}"'

    keyword_query = " OR ".join(
        [f'ti:"{k}" OR abs:"{k}"' for k in keywords]
    )

    # Restrict to CS AI domain
    return f"cat:cs.AI AND ({keyword_query})"


# =========================
# FETCH PAPERS (USED BY RETRIEVAL AGENT)
# =========================
def fetch_arxiv_papers(query, max_papers=5):

    search_query = build_arxiv_query(query)

    search = arxiv.Search(
        query=search_query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )

    client = arxiv.Client()

    papers = []

    for paper in client.results(search):
        try:
            arxiv_id = paper.entry_id.split("/")[-1]

            papers.append({
                "title": paper.title,
                "abstract": paper.summary,
                "year": paper.published.year,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}"
            })
        except:
            continue

    return papers


# =========================
# PUBLICATION TREND (FINAL FIXED VERSION)
# =========================
def get_publication_trend(query, start_year=2020):

    client = arxiv.Client()
    current_year = datetime.now().year

    # 🔥 IMPORTANT: use relevance (NOT date)
    search = arxiv.Search(
        query=f"all:{query}",
        max_results=300,   # larger pool for distribution
        sort_by=arxiv.SortCriterion.Relevance
    )

    # initialize year counts
    year_counts = {year: 0 for year in range(start_year, current_year + 1)}

    try:
        for paper in client.results(search):
            year = paper.published.year

            if year in year_counts:
                year_counts[year] += 1

    except Exception as e:
        print("Trend Error:", e)

    # convert to sorted list
    trend = [(year, year_counts[year]) for year in sorted(year_counts)]

    return trend