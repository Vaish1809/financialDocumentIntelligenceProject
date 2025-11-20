import nltk
import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class FinancialNLP:
    def __init__(self, text):
        self.text = text
        self.sentences = sent_tokenize(text)
        self.stop_words = set(stopwords.words('english'))

    def _sentence_similarity(self, sent1, sent2):
        """Calculate Cosine Similarity between two sentences"""
        sent1 = [w.lower() for w in sent1 if w.isalnum()]
        sent2 = [w.lower() for w in sent2 if w.isalnum()]
        
        all_words = list(set(sent1 + sent2))
        
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        for w in sent1: vector1[all_words.index(w)] += 1
        for w in sent2: vector2[all_words.index(w)] += 1
        
        return 1 - cosine_distance(vector1, vector2)

    def generate_textrank_summary(self, ratio=0.3):
        """
        Implements TextRank algorithm for extractive summarization.
        1. Build similarity matrix.
        2. Build graph.
        3. Calculate PageRank.
        4. Sort sentences by rank.
        """
        if len(self.sentences) < 2:
            return self.text

        similarity_matrix = np.zeros((len(self.sentences), len(self.sentences)))
        
        clean_sentences = [word_tokenize(s) for s in self.sentences]

        for i in range(len(self.sentences)):
            for j in range(len(self.sentences)):
                if i != j:
                    similarity_matrix[i][j] = self._sentence_similarity(clean_sentences[i], clean_sentences[j])

        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(self.sentences)), reverse=True)
        
        # Select top N sentences based on ratio
        num_sentences = max(1, int(len(self.sentences) * ratio))
        selected_sentences = [s for _, s in ranked_sentences[:num_sentences]]
        
        return " ".join(selected_sentences)

    def perform_topic_modeling(self, n_topics=3, n_words=5):
        """
        Uses sklearn LDA to find topics.
        """
        # Preprocessing for LDA
        vectorizer = CountVectorizer(stop_words='english', token_pattern=r'\b[a-zA-Z]{3,}\b')
        doc_term_matrix = vectorizer.fit_transform([self.text])
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        topics = {}
        for index, topic in enumerate(lda.components_):
            top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_words:]]
            topics[index] = top_words
            
        return topics

    def extract_entities(self):
        """
        Uses NLTK ne_chunk to find Persons and Organizations.
        """
        words = word_tokenize(self.text)
        pos_tags = nltk.pos_tag(words)
        chunks = nltk.ne_chunk(pos_tags)
        
        entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label'):
                label = chunk.label()
                if label in ['PERSON', 'ORGANIZATION', 'GPE']:
                    entity_name = " ".join(c[0] for c in chunk)
                    entities.append((entity_name, label))
        
        return entities