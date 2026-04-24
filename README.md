 Multi-Agentic  Research Assistant System

--> Overview

The Multi-Agentic  Research Assistant System is designed to automate the process of discovering, analyzing, and synthesizing academic research papers. The system focuses on Computer Science domains and transforms a user query into structured, research-ready insights using a pipeline of five intelligent agents.


--> System Architecture

The system follows a 5-agent architecture:

1. Query Processing and Domain Agent
   Refines the user query and detects the relevant research domain.

2. Retrieval and Filtering Agent
   Fetches research papers from arXiv and filters them based on relevance and time constraints.

3. Ranking Agent
   Ranks papers using semantic similarity to prioritize the most relevant results.

4. Analysis and Reasoning Agent
   Extracts structured information such as problem, method, dataset, performance, applications, and limitations.
   Performs cross-paper reasoning to identify trends, gaps, and insights.

5. Synthesis and Insight Agent
   Generates comparison tables, structured insights, research gaps, and recommendations.


--> Features

- Automated research paper retrieval from arXiv
- Semantic ranking of papers
- Structured information extraction
- Cross-paper analysis and reasoning
- Identification of research gaps and trends
- Generation of insights and recommendations
- Interactive visualization (similarity heatmap & trends)


--> How It Works

1. User enters a research query
2. Query is refined and domain is identified
3. Relevant papers are retrieved and filtered
4. Papers are ranked using semantic similarity
5. Key information is extracted from each paper
6. Cross-paper insights, gaps, and recommendations are generated
7. Results are presented in a structured format


--> Tech Stack

- Python
- Streamlit (UI)
- Sentence Transformers (semantic similarity)
- arXiv API
- Scikit-learn
- PyMuPDF (PDF parsing)
- Groq API (for reasoning)



--> Run the Project

pip install -r requirements.txt

streamlit run app/research_app2.py


--> Project Structure

project/
│
├── agents/                # 5 core agents
├── app/                   # Streamlit UI
├── retrieval/             # arXiv + semantic retrieval
├── ranking/               # ranking utilities
├── utils/                 # helper functions
├── requirements.txt
└── README.md



