import os
import sys
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.query_domain_agent import detect_domains
from agents.query_domain_agent import refine_query
from agents.ranking_agent import rank_papers
from agents.retrieval_filter_agent import retrieve_papers
from main import run_pipeline
from utils.ieee_baseline_table import IEEE_BASELINE_COMPARISON

st.set_page_config(layout="wide")
st.title("Multi Agentic Research  Assistant System ")

# =========================
# INPUTS
# =========================
query = st.text_input("Enter Research Topic")

time_period = st.selectbox(
    "Time Period",
    ["All", "2025-present", "2024-present", "2023-present"]
)

focus = st.selectbox(
    "Research Focus",
    ["general", "efficiency", "optimization", "scalability", "security"]
)

num_papers = st.slider("Number of Papers", 1, 5, 5)

st.markdown("---")

# =========================
# RETRIEVE PAPERS
# =========================
if st.button("Retrieve Papers"):

    if not query:
        st.warning("Please enter a research topic")
        st.stop()

    with st.spinner("Fetching papers..."):

        start = time.time()

        retrieval_query = refine_query(query)
        domains = detect_domains(query)

        papers = retrieve_papers(
            retrieval_query,
            max_results=num_papers * 3,
            time_filter=time_period
        )

        ranked = rank_papers(query + " " + focus, papers, num_papers) if papers else []

        st.session_state["retrieved"] = ranked
        st.session_state["retrieval_time"] = round(time.time() - start, 2)
        st.session_state["retrieval_query"] = retrieval_query
        st.session_state["domains"] = domains
        st.session_state["focus"] = focus
        st.session_state.pop("result", None)

        if ranked:
            st.success(f"Retrieved {len(ranked)} papers")
        else:
            st.warning("No papers found. Try simplifying query.")

# =========================
# SHOW RETRIEVED PAPERS
# =========================
if "retrieved" in st.session_state:

    st.subheader("📄 Retrieved Papers")

    st.caption(f"Time: {st.session_state['retrieval_time']} sec")
    st.caption(f"Query: {st.session_state['retrieval_query']}")
    st.caption(f"Focus: {st.session_state['focus']}")
    st.caption(f"Domains: {', '.join(st.session_state['domains'])}")

    for p in st.session_state["retrieved"]:
        st.markdown(f"### {p['title']} ({p['year']})")

        if p.get("pdf_url"):
            st.markdown(f"[📥 Open PDF]({p['pdf_url']})")

        st.write("Score:", round(p.get("score", 0), 4))
        st.markdown("---")

    # =========================
    # RUN ANALYSIS
    # =========================
    if st.button("Run Analysis"):

        with st.spinner("Running analysis..."):

            result = run_pipeline(
                query=query + " " + focus,
                max_papers=num_papers,
                time_filter=time_period
            )

            st.session_state["result"] = result

# =========================
# ANALYSIS OUTPUT
# =========================
if "result" in st.session_state:

    result = st.session_state["result"]

    # ------------------------
    # TOP PAPERS
    # ------------------------
    st.subheader("🏆 Top Papers")

    for p in result.get("papers", []):
        st.markdown(f"### {p['title']} ({p['year']})")

    # ------------------------
    # EXTRACTED FIELDS (FIXED)
    # ------------------------
    st.subheader("📊 Extracted Fields")

    for p in result.get("papers", []):

        fields = p.get("fields", {})

        st.markdown(f"### {p['title']}")

        st.markdown(f"**Problem:** {fields.get('problem','Not specified')}")
        st.markdown(f"**Method:** {fields.get('method','Not specified')}")
        st.markdown(f"**Dataset:** {fields.get('dataset','Not specified')}")
        st.markdown(f"**Performance:** {fields.get('performance','Not specified')}")
        st.markdown(f"**Application:** {fields.get('application','Not specified')}")
        st.markdown(f"**Limitations:** {fields.get('limitations','Not specified')}")
        st.markdown(f"**Research Type:** {fields.get('research_type','Not specified')}")

        st.markdown("---")

    # ------------------------
    # INSIGHTS
    # ------------------------
    st.subheader("🧠 Insights")

    insights = result.get("insights", "")

    for line in insights.split("\n"):
        if line.strip():
            st.markdown(f"- {line}")

    # ------------------------
    # COMPARISON TABLE
    # ------------------------
    st.subheader("📋 Comparison Table")

    df = pd.DataFrame(result.get("comparison", []))
    st.dataframe(df, width="stretch")

    # ------------------------
    # IEEE BASELINE
    # ------------------------
    st.subheader("📊 IEEE Baseline Comparison")
    st.dataframe(pd.DataFrame(IEEE_BASELINE_COMPARISON), width="stretch")

    # ------------------------
    # SIMILARITY HEATMAP
    # ------------------------
    st.subheader("🔗 Similarity Heatmap")

    sim = result.get("similarity")

    if sim is not None and len(sim) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(sim, annot=True, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough papers for similarity")

    # ------------------------
    # TRENDS
    # ------------------------
    st.subheader("📈 Publication Trends")

    trends = result.get("trends")

    if trends:
        years = [int(y) for y, _ in trends]
        counts = [c for _, c in trends]

        fig, ax = plt.subplots()
        ax.plot(years, counts, marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Papers")
        st.pyplot(fig)

    # ------------------------
    # GAPS
    # ------------------------
    st.subheader("⚠️ Research Gaps")

    for g in result.get("gaps", "").split("\n"):
        if g.strip():
            st.markdown(f"- {g}")

    # ------------------------
    # RECOMMENDATIONS
    # ------------------------
    st.subheader("🚀 Recommendations")

    for r in result.get("recommendations", "").split("\n"):
        if r.strip():
            st.markdown(f"- {r}")