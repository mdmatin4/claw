import re
import shutil
import gc
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os

GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-3-2b")

st.set_page_config(page_title="iClaw Interactive", layout="wide")

SAMPLE_TEXT = """Dr. Mohammad Husain is a Professor and the Inaugural Director of the PolySec Cyber Lab, a Center for Cyber Security and Forensics Education, Research, and Outreach in the Department of Computer Science at the California State Polytechnic University, Pomona (Cal Poly Pomona), CA. He has over five years’ experience in leadership, development, and administration of Computer Science and Cyber Security academic and co-curricular programs at a comprehensive university. He has significant leadership experience in extramural funding, graduate program, financial and strategic planning, assessment and accreditation, faculty governance, and building academic vision in diverse academic environments. His academic vision and leadership skills have resulted in state and national-level programs and diverse academic settings across multiple institutions.
Dr. Husain’s administrative leadership includes the Program Directorship of the Cal-Bridge Computer Science (CS) program, a California State University (CSU)-University of California (UC) Ph.D. pathway program for Computer Science and Engineering students from underrepresented minority communities. Dr. Husain leads the steering committee comprised of six CSUs, six UCs, and one community college. The program was launched in Fall 2020 and recruited 18 CSU students successfully. He is also the founding organizing chair of the SFSCon, a national cybersecurity training workshop for the CyberCorps Scholarship for Service (SFS) students. In Fall 2021, around 141 CyberCorps SFS students (BS/MS/Ph.D.) from 42 different US universities participated in this 2-day training.
At Cal Poly Pomona, he is the Principal Investigator (PI) of nine externally funded grants and co-PI of two externally funded grants totaling over $4.1M. As a part of the NSF CyberCorps SFS grant, he has trained students in Cyber Security and placed them at the NSA, DHS, NASA, MITRE, Sandia lab, Aerospace Corp., MIT Lincoln Lab, US Airforce, US Army, and US Navy since 2015. As a part of the ONR grant, he has trained Navy, Marine, and Army ROTC cadets from the following schools: USC, UCLA, UCSD, USD, SDSU, Point Loma, CSU San Marcos, and CPP. As the administrative lead of these grants, Dr. Husain has also managed over 30+ faculty and staff since 2015.
"""

DB_PATH = Path("./vector_db/file_index")
COLLECTION_NAME = "local_files"


def split_text_v2(text, chunk_size=500, overlap=100):
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_client(path: str):
    return chromadb.PersistentClient(path=path)


def get_collection(client, name: str, recreate: bool = False):
    existing = [c.name for c in client.list_collections()]

    if recreate and name in existing:
        client.delete_collection(name)

    if name in existing and not recreate:
        return client.get_collection(name)

    return client.create_collection(name)


@st.cache_resource
def load_llm(model_name: str = GEMMA_MODEL):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Gemma 3 2B model '{model_name}'. "
            "Make sure the model is available locally or set GEMMA_MODEL to a valid Hugging Face repo ID, "
            "and authenticate with `huggingface-cli login` if necessary."
        ) from exc
    return model, tokenizer


def generate_answer(model, tokenizer, context: str, question: str, max_length: int = 256):
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not contained in the context, say that you do not know.\n\n"
        "Context:\n" + context + "\n\nQuestion: " + question + "\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_length,
        do_sample=False,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def load_text_files(paths):
    documents = []
    metadatas = []

    for folder in paths:
        root = Path(folder.strip())
        if not root.exists():
            continue

        for file_path in root.rglob("*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                documents.append(text)
                metadatas.append({"file_path": str(file_path), "file_name": file_path.name})
            except Exception:
                continue

    return documents, metadatas


def index_documents(collection, model, texts, metadatas, chunk_size, overlap):
    new_ids = []
    embeddings = []
    docs = []
    metas = []

    for i, text in enumerate(texts):
        chunks = split_text_v2(text, chunk_size=chunk_size, overlap=overlap)
        for j, chunk in enumerate(chunks):
            new_ids.append(f"doc-{i}-{j}")
            embeddings.append(model.encode(chunk).tolist())
            docs.append(chunk)
            metas.append(metadatas[i] if i < len(metadatas) else {"source": f"text-{i}"})

    if new_ids:
        collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=docs,
            metadatas=metas,
        )

    return len(new_ids)


def query_collection(collection, model, query, n_results=3):
    if not query:
        return None
    query_embedding = model.encode(query).tolist()
    return collection.query(query_embeddings=[query_embedding], n_results=n_results)


def main():
    st.title("iClaw Interactive Vector Search")
    st.write("Use this app to build a local vector index and search with natural language queries.")

    model = load_model()
    client = get_client(str(DB_PATH))

    with st.sidebar:
        st.header("Indexing Configuration")
        st.markdown("Use sample text or provide folder paths with `.txt` files.")
        use_sample_text = st.checkbox("Use sample text", value=True)
        sample_text = st.text_area("Sample text to index", value=SAMPLE_TEXT, height=250)
        folder_input = st.text_area("Folder paths to scan (one per line)", value="", height=120)
        chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=500, step=50)
        overlap = st.number_input("Chunk overlap", min_value=0, max_value=chunk_size - 1, value=100, step=10)
        collection_name = st.text_input("Collection name", value=COLLECTION_NAME)

        if st.button("Initialize / Load collection"):
            try:
                collection = get_collection(client, collection_name, recreate=False)
                st.success(f"Loaded collection '{collection_name}'.")
            except Exception as exc:
                st.error(f"Failed to load collection: {exc}")

        if st.button("Rebuild collection"):
            try:
                collection = get_collection(client, collection_name, recreate=True)
                st.success(f"Rebuilt collection '{collection_name}'.")
            except Exception as exc:
                st.error(f"Failed to rebuild collection: {exc}")

        st.divider()
        st.header("Index data")
        if st.button("Index sample text"):
            try:
                collection = get_collection(client, collection_name)
                doc_texts = [sample_text] if use_sample_text else []
                doc_metas = [{"source": "sample_text"}] if use_sample_text else []
                if folder_input.strip():
                    folders = [line.strip() for line in folder_input.splitlines() if line.strip()]
                    file_texts, file_metas = load_text_files(folders)
                    doc_texts.extend(file_texts)
                    doc_metas.extend(file_metas)

                if not doc_texts:
                    st.warning("No text found to index. Add sample text or a valid folder path.")
                else:
                    count = index_documents(collection, model, doc_texts, doc_metas, int(chunk_size), int(overlap))
                    st.success(f"Indexed {count} chunk(s) into '{collection_name}'.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.header("Query the Collection")
    query_collection_name = st.sidebar.text_input("Collection name for query", value=COLLECTION_NAME)
    collection = get_collection(client, query_collection_name)

    query = st.text_input("Ask a question", value="What is Dr. Husain's role?")
    n_results = st.slider("Number of results", min_value=1, max_value=5, value=3)
    llm_model_name = GEMMA_MODEL
    st.text_input("LLM model (fixed)", value=llm_model_name, disabled=True)

    if st.button("Search") and query:
        try:
            results = query_collection(collection, model, query, n_results=n_results)
            if results and results.get("documents"):
                hits = results["documents"][0]
                metas = results.get("metadatas", [[]])[0]
                scores = results.get("distances", [[]])[0] if "distances" in results else []

                # Combine and sort by score descending so the highest score appears first
                result_items = list(zip(scores, hits, metas))
                result_items.sort(key=lambda x: x[0], reverse=True)

                context = "\n\n".join([hit for _, hit, _ in result_items])
                llm_answer = None
                with st.spinner("Generating answer with Gemma 3 2B..."):
                    llm_model, llm_tokenizer = load_llm(llm_model_name)
                    llm_answer = generate_answer(llm_model, llm_tokenizer, context, query)

                if llm_answer:
                    st.subheader("LLM Answer")
                    st.write(llm_answer)
                    st.write("---")

                for idx, (score, hit, meta) in enumerate(result_items):
                    st.subheader(f"Result {idx + 1}")
                    st.write(f"Score: {score:.4f}")
                    st.write(hit)
                    st.write(meta)
                    st.write("---")
            else:
                st.info("No results found. Try indexing more content or broadening the query.")
        except Exception as exc:
            st.error(f"Search failed: {exc}")


if __name__ == "__main__":
    main()
