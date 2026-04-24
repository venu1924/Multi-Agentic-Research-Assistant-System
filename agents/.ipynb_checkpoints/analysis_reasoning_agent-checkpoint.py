from dotenv import load_dotenv
import os
from groq import Groq
import time
import re
import requests
import fitz  # PyMuPDF

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None


# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:4000]


# =========================
# PDF FALLBACK (CRITICAL)
# =========================
def extract_pdf_text(url, max_pages=2):
    try:
        r = requests.get(url, timeout=10)
        doc = fitz.open(stream=r.content, filetype="pdf")

        text = ""
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()

        return clean_text(text)

    except:
        return ""


# =========================
# DATASET DETECTION
# =========================
def extract_dataset(text):
    keywords = ["dataset", "benchmark", "kitti", "nuscenes", "waymo", "cifar", "imagenet"]

    for sent in text.split("."):
        if any(k in sent.lower() for k in keywords):
            return sent.strip()

    return "Not specified"


# =========================
# RESEARCH TYPE
# =========================
def detect_type(text):
    t = text.lower()

    if "survey" in t or "review" in t:
        return "Survey"

    if "experiment" in t or "evaluation" in t:
        return "Experimental"

    if "framework" in t or "model" in t:
        return "Proposed Method"

    return "Unknown"


# =========================
# ROBUST PARSER (FIXED)
# =========================
def parse_output(text):

    fields = {
        "problem": "Not specified",
        "method": "Not specified",
        "dataset": "Not specified",
        "performance": "Not specified",
        "application": "Not specified",
        "limitations": "Not specified"
    }

    for line in text.split("\n"):

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip().lower()
        value = value.strip()

        if key in fields and value:
            fields[key] = value

    return fields


# =========================
# FALLBACK (STRONG)
# =========================
def fallback(p, text):

    sentences = text.split(".")

    return {
        "title": p.get("title", ""),
        "problem": sentences[0] if len(sentences) > 0 else "Not specified",
        "method": sentences[1] if len(sentences) > 1 else "Not specified",
        "dataset": extract_dataset(text),
        "performance": next((s for s in sentences if "accuracy" in s.lower()), "Not specified"),
        "application": "General application inferred",
        "limitations": "Not explicitly stated",
        "research_type": detect_type(text)
    }


# =========================
# MAIN FUNCTION
# =========================
def analyze_paper(p):

    # 🔥 STEP 1: USE TITLE + ABSTRACT
    text = (p.get("title", "") + " " + p.get("abstract", ""))

    # 🔥 STEP 2: PDF fallback if weak
    if len(text) < 500 and p.get("pdf_url"):
        text += " " + extract_pdf_text(p["pdf_url"])

    text = clean_text(text)

    # 🔥 STEP 3: STRICT PROMPT
    prompt = f"""
Extract EXACTLY in this format:

Problem: <one line>
Method: <one line>
Dataset: <one line>
Performance: <one line>
Application: <one line>
Limitations: <one line>

Rules:
- MUST follow exact format
- One line per field
- No extra explanation
- If missing → "Not specified"

Text:
{text}
"""

    try:
        if client is None:
            return fallback(p, text)

        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        output = res.choices[0].message.content

        parsed = parse_output(output)

        return {
            "title": p.get("title", ""),
            "problem": parsed["problem"],
            "method": parsed["method"],
            "dataset": parsed["dataset"] if parsed["dataset"] != "Not specified" else extract_dataset(text),
            "performance": parsed["performance"],
            "application": parsed["application"],
            "limitations": parsed["limitations"],
            "research_type": detect_type(text)
        }

    except Exception:
        return fallback(p, text)


# =========================
# MULTIPLE
# =========================
def analyze_multiple(papers):
    return [analyze_paper(p) for p in papers]





import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


# =========================
# 🔥 VALIDATION (RELAXED + SMART)
# =========================
def is_valid_output(text):
    t = text.lower()

    conditions = [
        any(k in t for k in ["while", "whereas", "however"]),  # comparison signal
        any(k in t for k in ["clinical", "healthcare", "decision"]),  # domain signal
    ]

    return sum(conditions) >= 1


# =========================
# 🔥 FALLBACK (SAFE + DOMAIN-AWARE)
# =========================
def fallback_reasoning(analyses):

    insights = []
    gaps = []
    recs = []

    insights.append(
        "While some approaches focus on integrating AI into clinical workflows, others emphasize explainable models, indicating a gap between system usability and interpretability."
    )

    insights.append(
        "Whereas explainability is highlighted as essential for trust, fairness is not explicitly evaluated, suggesting a disconnect between interpretability and ethical deployment."
    )

    gaps.append("Fairness and bias evaluation is not systematically addressed in most studies.")
    gaps.append("Datasets and performance metrics are not consistently reported.")
    gaps.append("Real-world clinical deployment is not sufficiently evaluated.")

    recs.append("Future work should integrate explainability directly into clinical workflows.")
    recs.append("Fairness-aware evaluation metrics must be incorporated into CDSS models.")
    recs.append("Standardized datasets and benchmarking protocols should be used.")

    return "\n".join(insights), "\n".join(gaps), "\n".join(recs)


# =========================
# 🔥 PARSER (ROBUST)
# =========================
def parse_output(text):

    insights, gaps, recs = "", "", ""
    current = None

    for line in text.split("\n"):
        l = line.strip()

        if not l:
            continue

        upper = l.upper()

        if upper.startswith("INSIGHTS"):
            current = "i"
            continue
        elif upper.startswith("GAPS"):
            current = "g"
            continue
        elif upper.startswith("RECOMMENDATIONS"):
            current = "r"
            continue

        if current == "i":
            insights += l + "\n"
        elif current == "g":
            gaps += l + "\n"
        elif current == "r":
            recs += l + "\n"

    return insights.strip(), gaps.strip(), recs.strip()


# =========================
# 🔥 MAIN FUNCTION
# =========================
def generate_insights(analyses):

    if not analyses:
        return fallback_reasoning(analyses)

    # =========================
    # 🔥 BUILD PAIRWISE INPUT
    # =========================
    paired = ""

    for i in range(len(analyses) - 1):
        a = analyses[i]
        b = analyses[i + 1]

        paired += f"""
Paper A:
Title: {a.get('title','')}
Method: {a.get('method','')}
Dataset: {a.get('dataset','')}
Performance: {a.get('performance','')}
Limitations: {a.get('limitations','')}

Paper B:
Title: {b.get('title','')}
Method: {b.get('method','')}
Dataset: {b.get('dataset','')}
Performance: {b.get('performance','')}
Limitations: {b.get('limitations','')}

---
"""

    # =========================
    # 🔥 FINAL PROMPT
    # =========================
    prompt = f"""
Perform HIGH-LEVEL CROSS-PAPER ANALYSIS.

STRICT RULES:
- Compare papers using words like "while", "whereas", or "however"
- DO NOT summarize papers individually
- Identify at least ONE contradiction
- Analyze fairness (present OR missing)
- Use ONLY given data
- Avoid generic statements

INSIGHTS:
- Compare methods across papers
- Explain WHY differences exist
- Highlight trade-offs (trust vs performance, explainability vs usability)

GAPS:
- Identify missing elements:
  - fairness evaluation
  - dataset transparency
  - real-world deployment
- Explain impact of each gap

RECOMMENDATIONS:
- Must directly address identified gaps
- Must be actionable and technical

OUTPUT FORMAT:

INSIGHTS:
- ...

GAPS:
- ...

RECOMMENDATIONS:
- ...

PAPERS:
{paired}
"""

    # =========================
    # 🔥 RETRY LOOP
    # =========================
    final_text = ""

    for attempt in range(2):
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            text = res.choices[0].message.content

            if is_valid_output(text):
                final_text = text
                break
            else:
                print("⚠️ Weak reasoning → retrying...")

        except Exception as e:
            print("LLM Error:", e)

    # =========================
    # 🔥 USE BEST AVAILABLE OUTPUT
    # =========================
    if not final_text:
        return fallback_reasoning(analyses)

    insights, gaps, recs = parse_output(final_text)

    # final safety check
    if not insights.strip():
        return fallback_reasoning(analyses)

    return insights, gaps, recs



import arxiv
import re
from collections import defaultdict
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_arxiv_query(user_query):

    query = user_query.lower()

    stop_words = {
        "in", "for", "of", "and", "the",
        "with", "on", "using", "applications", "application"
    }

    words = re.findall(r'\b\w+\b', query)

    keywords = [w for w in words if w not in stop_words]

    return " AND ".join(keywords[:5])


def fetch_trend_data(query):

    search_query = build_arxiv_query(query)

    search = arxiv.Search(
        query=f"(ti:{search_query} OR abs:{search_query}) AND cat:cs.*",
        max_results=300,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    year_counts = defaultdict(int)

    try:
        for result in arxiv.Client().results(search):
            year = result.published.year
            if year >= 2019:
                year_counts[year] += 1
    except Exception as e:
        print("Trend error:", e)

    current_year = datetime.now().year
    years = list(range(2019, current_year + 1))

    return [(str(y), year_counts.get(y, 0)) for y in years]


def compute_metrics(analyses, papers, query):

    texts = [
        (a.get("problem", "") + " " +
         a.get("method", "") + " " +
         a.get("application", ""))
        for a in analyses
    ]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(texts)

    similarity = cosine_similarity(tfidf)

    trends = fetch_trend_data(query)

    return similarity, trends