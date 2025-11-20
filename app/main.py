import streamlit as st
from nlp_engine import FinancialNLP
import pandas as pd

st.set_page_config(page_title="FinDoc Intelligence", page_icon="📈", layout="wide")

st.title("📈 Financial Document Intelligence")
st.markdown("### *Extract Insights from Earnings Calls & Filings*")

# Sidebar / Configuration
with st.sidebar:
    st.header("Configuration")
    num_topics = st.slider("Number of Topics (LDA)", 2, 10, 3)
    summary_ratio = st.slider("Summary Length (Ratio)", 0.1, 0.5, 0.2)
    st.info("Paste a financial text or earnings call transcript to begin.")

# Default text input (Mock Earnings Call Segment)
default_text = """
Good afternoon, everyone, and welcome to our Q4 earnings call. 
Revenue grew by 15% year-over-year to $4.5 billion, driven by strong demand in our cloud computing division. 
However, operating margins were impacted by higher supply chain costs and inflationary pressures. 
We are effectively managing our inventory levels.
Looking ahead, we expect to invest heavily in AI infrastructure. 
Our CEO, Jane Doe, emphasized that strategic acquisitions in Europe will drive future growth. 
We are cautious about the macroeconomic headwinds but remain optimistic about our long-term strategy.
"""

text_input = st.text_area("Input Text", value=default_text, height=250)

if st.button("Analyze Document"):
    if len(text_input) < 50:
        st.warning("Please enter a longer text for meaningful analysis.")
    else:
        nlp = FinancialNLP(text_input)
        
        with st.spinner("Running NLP Pipeline..."):
            # Run Analysis
            summary = nlp.generate_textrank_summary(ratio=summary_ratio)
            topics = nlp.perform_topic_modeling(n_topics=num_topics)
            entities = nlp.extract_entities()

        # Display Results via Tabs
        tab1, tab2, tab3 = st.tabs(["📝 Executive Summary", "📊 Thematic Topics", "🏢 Entities"])

        with tab1:
            st.subheader("Extractive Summary (TextRank)")
            st.markdown(f"> {summary}")
            st.caption(f"Reduced content length by approx {100 - (summary_ratio*100)}%")

        with tab2:
            st.subheader("Latent Dirichlet Allocation (LDA) Topics")
            
            # Format topics for display
            topic_data = []
            for idx, words in topics.items():
                topic_data.append({"Topic ID": f"Topic {idx+1}", "Key Terms": ", ".join(words)})
            
            df_topics = pd.DataFrame(topic_data)
            st.table(df_topics)

        with tab3:
            st.subheader("Named Entity Recognition")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Organizations**")
                orgs = [e for e, label in entities if label == "ORGANIZATION"]
                if orgs:
                    for o in set(orgs): st.markdown(f"- 🏢 {o}")
                else:
                    st.write("No organizations detected.")

            with col2:
                st.markdown("**Persons**")
                persons = [e for e, label in entities if label == "PERSON"]
                if persons:
                    for p in set(persons): st.markdown(f"- 👤 {p}")
                else:
                    st.write("No persons detected.")