import streamlit as st
import difflib
import re
from collections import Counter
import string

# Try importing sklearn, provide fallback if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("⚠️ scikit-learn not found! Install with: `pip install scikit-learn numpy`")
    st.info("The app will work with limited functionality using other algorithms.")

# Page configuration
st.set_page_config(
    page_title="Advanced Plagiarism Tracker",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .highlight-match {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔍 Advanced Plagiarism Tracker</div>', unsafe_allow_html=True)

# Preprocessing function
def preprocess_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Tokenization
def tokenize(text):
    """Split text into words"""
    return text.split()

# Method 1: Cosine Similarity using TF-IDF
def cosine_similarity_check(text1, text2):
    """Calculate similarity using TF-IDF and cosine similarity"""
    if not SKLEARN_AVAILABLE:
        return None
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity * 100
    except:
        return 0.0

# Method 2: Jaccard Similarity
def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity coefficient"""
    set1 = set(tokenize(text1))
    set2 = set(tokenize(text2))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0.0
    return (len(intersection) / len(union)) * 100

# Method 3: Sequence Matcher (Longest Common Subsequence)
def sequence_similarity(text1, text2):
    """Calculate similarity using SequenceMatcher"""
    return difflib.SequenceMatcher(None, text1, text2).ratio() * 100

# Method 4: N-gram Overlap
def ngram_similarity(text1, text2, n=3):
    """Calculate similarity based on n-gram overlap"""
    def get_ngrams(text, n):
        words = tokenize(text)
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    
    ngrams1 = set(get_ngrams(text1, n))
    ngrams2 = set(get_ngrams(text2, n))
    
    if len(ngrams1) == 0 or len(ngrams2) == 0:
        return 0.0
    
    intersection = ngrams1.intersection(ngrams2)
    return (len(intersection) / max(len(ngrams1), len(ngrams2))) * 100

# Method 5: Levenshtein Distance (character-level)
def levenshtein_similarity(text1, text2):
    """Calculate similarity using Levenshtein distance"""
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    max_len = max(len(text1), len(text2))
    if max_len == 0:
        return 100.0
    distance = levenshtein_distance(text1, text2)
    return (1 - distance / max_len) * 100

# Find matching phrases
def find_matching_phrases(text1, text2, min_length=5):
    """Find matching phrases between two texts"""
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    matches = []
    
    for i in range(len(words1)):
        for j in range(len(words2)):
            k = 0
            while (i + k < len(words1) and j + k < len(words2) and 
                   words1[i + k] == words2[j + k]):
                k += 1
            if k >= min_length:
                match = ' '.join(words1[i:i+k])
                matches.append((match, k))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)

# Main app
def main():
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        algorithm = st.selectbox(
            "Detection Algorithm",
            ["All Methods", "Cosine Similarity (TF-IDF)", "Jaccard Similarity", 
             "Sequence Matcher", "N-gram Overlap", "Levenshtein Distance"]
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This tool uses multiple algorithms to detect plagiarism:
        
        - **Cosine Similarity**: TF-IDF vectorization
        - **Jaccard**: Set-based comparison
        - **Sequence Matcher**: Longest common subsequence
        - **N-gram**: Phrase overlap detection
        - **Levenshtein**: Character-level distance
        """)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Original Text")
        text1 = st.text_area(
            "Enter the original text:",
            height=300,
            placeholder="Paste the original text here..."
        )
    
    with col2:
        st.subheader("📝 Text to Check")
        text2 = st.text_area(
            "Enter the text to check for plagiarism:",
            height=300,
            placeholder="Paste the text to check here..."
        )
    
    if st.button("🔍 Analyze Plagiarism", type="primary", use_container_width=True):
        if not text1 or not text2:
            st.error("⚠️ Please enter text in both fields!")
            return
        
        # Preprocess texts
        proc_text1 = preprocess_text(text1)
        proc_text2 = preprocess_text(text2)
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Calculate similarities
        results = {}
        
        if algorithm in ["All Methods", "Cosine Similarity (TF-IDF)"]:
            if SKLEARN_AVAILABLE:
                results["Cosine Similarity"] = cosine_similarity_check(proc_text1, proc_text2)
            elif algorithm == "Cosine Similarity (TF-IDF)":
                st.error("Cosine Similarity requires scikit-learn. Please install it.")
                return
        
        if algorithm in ["All Methods", "Jaccard Similarity"]:
            results["Jaccard Similarity"] = jaccard_similarity(proc_text1, proc_text2)
        
        if algorithm in ["All Methods", "Sequence Matcher"]:
            results["Sequence Matcher"] = sequence_similarity(proc_text1, proc_text2)
        
        if algorithm in ["All Methods", "N-gram Overlap"]:
            results["N-gram Overlap"] = ngram_similarity(proc_text1, proc_text2)
        
        if algorithm in ["All Methods", "Levenshtein Distance"]:
            results["Levenshtein Similarity"] = levenshtein_similarity(proc_text1, proc_text2)
        
        # Display results
        cols = st.columns(len(results))
        for idx, (method, score) in enumerate(results.items()):
            with cols[idx]:
                st.metric(
                    label=method,
                    value=f"{score:.2f}%",
                    delta=None
                )
        
        # Overall similarity
        if results:
            avg_similarity = sum(results.values()) / len(results)
        else:
            st.error("No algorithms could be run. Please install required packages.")
            return
        
        st.markdown("---")
        st.subheader("🎯 Overall Assessment")
        
        # Progress bar for overall similarity
        st.progress(avg_similarity / 100)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Similarity", f"{avg_similarity:.2f}%")
        with col2:
            st.metric("Highest Match", f"{max(results.values()):.2f}%")
        with col3:
            st.metric("Lowest Match", f"{min(results.values()):.2f}%")
        
        # Verdict
        if avg_similarity >= 80:
            st.error("🚨 **HIGH PLAGIARISM DETECTED** - The texts are highly similar!")
        elif avg_similarity >= 50:
            st.warning("⚠️ **MODERATE PLAGIARISM** - Significant similarities found!")
        elif avg_similarity >= 30:
            st.info("ℹ️ **LOW PLAGIARISM** - Some similarities detected.")
        else:
            st.success("✅ **NO SIGNIFICANT PLAGIARISM** - Texts appear to be original.")
        
        # Find matching phrases
        st.markdown("---")
        st.subheader("🔗 Matching Phrases")
        
        matches = find_matching_phrases(proc_text1, proc_text2, min_length=3)
        
        if matches:
            st.write(f"Found **{len(matches)}** matching phrase(s):")
            for idx, (phrase, length) in enumerate(matches[:10], 1):
                st.markdown(f"{idx}. `{phrase}` ({length} words)")
        else:
            st.info("No significant matching phrases found.")
        
        # Detailed comparison
        with st.expander("📋 Detailed Text Comparison"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text (processed)**")
                st.text(proc_text1[:500] + "..." if len(proc_text1) > 500 else proc_text1)
            with col2:
                st.markdown("**Text to Check (processed)**")
                st.text(proc_text2[:500] + "..." if len(proc_text2) > 500 else proc_text2)

if __name__ == "__main__":
    main()
