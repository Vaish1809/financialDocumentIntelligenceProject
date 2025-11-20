# Financial Document Intelligence Tool

**FinDoc Intel** is an NLP-powered analytics dashboard designed to process financial documents such as earnings call transcripts, 10-K filings, and news articles. It automates the extraction of key insights, reducing manual review time.

## 🧠 Core Features

1.  **Automated Summarization**: Uses **TextRank** (graph-based algorithm) to extract the most critical sentences.
2.  **Topic Modeling**: Implements **Latent Dirichlet Allocation (LDA)** via `scikit-learn` to discover hidden themes (e.g., "Growth", "Risk", "Revenue").
3.  **Entity Extraction**: Uses **NLTK** to identify companies, organizations, and locations mentioned in the text.
4.  **Interactive Dashboard**: Built with **Streamlit** for real-time analysis.

## 🚀 Tech Stack

* **Python 3.10**
* **Scikit-learn** (LDA, TF-IDF)
* **NLTK** (Tokenization, NER, Stopwords)
* **NetworkX** (Graph algorithms for TextRank)
* **Streamlit** (UI)
* **Docker**

## 🛠 How to Run

1.  **Prerequisites**: Docker Desktop installed.
2.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
3.  **Access the App**:
    Open http://localhost:8501 in your browser.

4.  **Test Data**:
    Paste a sample earnings call transcript into the text area to see the analysis.