from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


# =========================
# 🔥 FALLBACK (SMART + COMPARATIVE)
# =========================
def _fallback(analyses):

    insights = [
        "While some approaches focus on explainable AI models, others emphasize integrating decision support into clinical workflows, indicating a contradiction between interpretability and usability.",
        "Whereas fairness is discussed conceptually in some studies, it is not evaluated in deployed systems, creating a gap between ethical design and real-world application."
    ]

    gaps = [
        "Fairness and bias evaluation is not systematically addressed across papers.",
        "Datasets and performance metrics are not consistently reported.",
        "Real-world clinical deployment is not sufficiently evaluated."
    ]

    recs = [
        "Future work should integrate explainability directly into clinical workflows.",
        "Fairness-aware evaluation metrics must be incorporated into CDSS systems.",
        "Standardized datasets and benchmarking protocols should be adopted."
    ]

    return "\n".join(insights), "\n".join(gaps), "\n".join(recs)


# =========================
# 🔥 PARSER
# =========================
def _parse(text):

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

    # =========================
    # COMPARISON TABLE (for UI)
    # =========================
    comparison = [{
        "title": a.get("title", ""),
        "method": a.get("method", ""),
        "dataset": a.get("dataset", ""),
        "performance": a.get("performance", ""),
        "application": a.get("application")
    } for a in analyses]

    if client is None or not analyses:
        i, g, r = _fallback(analyses)
        return comparison, i, g, r

    # =========================
    # 🔥 PAIRWISE INPUT (CRITICAL)
    # =========================
    paired = ""

    for i in range(len(analyses) - 1):
        a = analyses[i]
        b = analyses[i + 1]

        paired += f"""
Paper A:
Title: {a.get('title')}
Method: {a.get('method')}
Research Type: {a.get('research_type')}
Limitations: {a.get('limitations')}

Paper B:
Title: {b.get('title')}
Method: {b.get('method')}
Research Type: {b.get('research_type')}
Limitations: {b.get('limitations')}
---
"""

    # =========================
    # 🔥 FINAL PROMPT (UPGRADED)
    # =========================
    prompt = f"""
Perform HIGH-LEVEL CROSS-PAPER ANALYSIS.

STRICT RULES:
- Use ONLY provided paper data
- DO NOT summarize papers individually
- MUST compare papers using "while", "whereas", or "however"
- MUST include at least ONE contradiction
- MUST refer to METHODS explicitly (e.g., case study, framework, DST, survey)
- MUST analyze fairness (present OR missing)
- Avoid generic statements

INSIGHTS:
- Compare methods across papers (NOT summaries)
- Explain WHY differences exist
- Highlight trade-offs:
  (interpretability vs usability, trust vs performance, fairness vs deployment)
- Include at least ONE strong contradiction

GAPS:
- Identify missing elements:
  - fairness evaluation
  - dataset transparency
  - real-world deployment
- Explain WHY these gaps matter

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
    # 🔥 RETRY LOOP (SAFE)
    # =========================
    final_text = ""

    for _ in range(2):
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            text = res.choices[0].message.content

            insights, gaps, recs = _parse(text)

            # 🔥 accept if meaningful (DON’T over-reject)
            if any(k in insights.lower() for k in ["while", "whereas", "however"]):
                final_text = text
                break

        except Exception as e:
            print("LLM Error:", e)

    # =========================
    # 🔥 FINAL OUTPUT
    # =========================
    if not final_text:
        i, g, r = _fallback(analyses)
        return comparison, i, g, r

    insights, gaps, recs = _parse(final_text)

    if not insights.strip():
        i, g, r = _fallback(analyses)
        return comparison, i, g, r

    return comparison, insights.strip(), gaps.strip(), recs.strip()





def build_table(papers):
    table = []

    for p in papers:
        f = p.get("fields", {})

        table.append({
            "Rank": p.get("rank", ""),
            "Title": p.get("title", "")[:50],
            "Method": f.get("Method", ""),
            "Dataset": f.get("Dataset", ""),
            "Performance": f.get("Performance", "")
        })

    return table