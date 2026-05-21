import os
import streamlit as st
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

try:
    st.set_page_config(page_title="RAG Concepts Visualizer", page_icon="🧠", layout="wide")
except st.errors.StreamlitAPIException:
    pass

st.title("🧠 RAG Concepts Visualizer")
st.markdown("Understand the core concepts of Retrieval-Augmented Generation: **Tokenization**, **Embeddings**, and **Chunking**.")

@st.cache_resource
def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(GEMMA_MODEL)
    except Exception as exc:
        st.error(f"Failed to load tokenizer for {GEMMA_MODEL}. Error: {exc}")
        return None

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = load_tokenizer()
embedding_model = load_embedding_model()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

tab1, tab2, tab3 = st.tabs(["1. Tokenization", "2. Embeddings & Distance", "3. Text Chunking"])

# --- TAB 1: Tokenization ---
with tab1:
    st.header("1. Tokenization")
    st.markdown("""
    **What is a token?**  
    Language Models do not read text like humans do. Instead, they break down text into smaller pieces called **tokens**. 
    A token can be a whole word, part of a word, or just a single character. Each token is assigned a unique ID number that the model understands.
    """)
    
    text_to_tokenize = st.text_area("Enter text to tokenize:", value="Hello! This is a simple test for tokenization.", key="tok_text")
    
    if st.button("Tokenize Text"):
        if text_to_tokenize and tokenizer:
            with st.spinner("Tokenizing..."):
                tokens = tokenizer.tokenize(text_to_tokenize)
                token_ids = tokenizer.encode(text_to_tokenize, add_special_tokens=False)
                
                st.success(f"**Total Tokens:** {len(tokens)}")
                
                # Display tokens and IDs side-by-side in columns
                cols = st.columns(6)
                for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
                    with cols[i % 6]:
                        st.info(f"**{tok}**\n\n`ID: {tid}`")
        elif not tokenizer:
            st.warning("Tokenizer not loaded.")
        else:
            st.warning("Please enter some text.")

# --- TAB 2: Embeddings & Vectors ---
with tab2:
    st.header("2. Embeddings & Distance")
    st.markdown("""
    **What is an embedding?**  
    An embedding converts text into a list of numbers (a vector) in high-dimensional space. 
    Texts with similar meanings will have vectors that point in similar directions, meaning they are closer together in this space.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("Vector 1 Text:", value="The quick brown fox jumps over the lazy dog.", key="emb_text1")
    with col2:
        text2 = st.text_area("Vector 2 Text:", value="A fast colored fox leaps over a sleepy dog.", key="emb_text2")
        
    if st.button("Generate & Compare Vectors"):
        if text1 and text2:
            with st.spinner("Generating embeddings..."):
                emb1 = embedding_model.encode(text1)
                emb2 = embedding_model.encode(text2)
                
                cos_sim = cosine_similarity(emb1, emb2)
                euc_dist = euclidean_distance(emb1, emb2)
                
                st.subheader("Comparison Results")
                m1, m2 = st.columns(2)
                m1.metric("Cosine Similarity", f"{cos_sim:.4f}", help="Ranges from -1 to 1. Closer to 1 means the texts are more similar in meaning.")
                m2.metric("Euclidean Distance", f"{euc_dist:.4f}", help="Straight line distance between vectors. Lower means they are closer.")
                
                st.subheader("Vector Visualization (First 20 dimensions)")
                st.markdown(f"The full vector has **{len(emb1)} dimensions** (it's an array of {len(emb1)} numbers). Here are the first 20 plotted side-by-side:")
                
                df = pd.DataFrame({
                    "Dimension": range(1, 21),
                    "Vector 1": emb1[:20],
                    "Vector 2": emb2[:20]
                })
                st.bar_chart(df.set_index("Dimension"))
                
                with st.expander("Show raw vector values"):
                    st.write("**Raw Vector 1 (first 10 values):**", np.round(emb1[:10], 4))
                    st.write("**Raw Vector 2 (first 10 values):**", np.round(emb2[:10], 4))
        else:
            st.warning("Please enter text for both vectors.")

# --- TAB 3: Chunking ---
with tab3:
    st.header("3. Text Chunking")
    st.markdown("""
    **What is chunking?**  
    When building a Vector Database for large documents (like PDFs or books), you can't embed the entire text at once because AI models have limits on how much text they can read at once. 
    Instead, you split the document into smaller **chunks**. Adding an **overlap** ensures that context isn't lost between chunk boundaries.
    """)
    
    chunk_text = st.text_area(
        "Long Document Text:", 
        value="Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence displayed by animals and humans. Leading AI textbooks define the field as the study of 'intelligent agents': any system that perceives its environment and takes actions that maximize its chance of achieving its goals. Some popular accounts use the term 'artificial intelligence' to describe machines that mimic 'cognitive' functions that humans associate with the human mind, such as 'learning' and 'problem solving', however, this definition is rejected by major AI researchers.",
        height=150,
        key="chk_text"
    )
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        chunk_size = st.number_input("Chunk Size (characters)", min_value=10, max_value=1000, value=150, step=10)
    with col_c2:
        overlap = st.number_input("Overlap (characters)", min_value=0, max_value=chunk_size-1, value=30, step=5)
        
    if st.button("Chunk Text"):
        if chunk_text:
            chunks = []
            i = 0
            while i < len(chunk_text):
                chunks.append(chunk_text[i:i + chunk_size])
                i += (chunk_size - overlap)
                if i >= len(chunk_text):
                    break
                
            st.success(f"Created {len(chunks)} chunks!")
            for idx, c in enumerate(chunks):
                st.info(f"**Chunk {idx+1}** (Length: {len(c)} characters)\n\n{c}")
        else:
            st.warning("Please enter some text to chunk.")
