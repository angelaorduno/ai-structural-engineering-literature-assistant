from pathlib import Path
import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI

# -------------------------
# Paths and environment
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="AI Structural Engineering Literature Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Structural Engineering Literature Assistant")
st.write("Chat with your structural engineering and AI literature collection.")

# -------------------------
# Load embedding model once
# -------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# -------------------------
# Helper functions
# -------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text: str, chunk_size: int = 700) -> list[str]:
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
        if text[i:i + chunk_size].strip()
    ]


@st.cache_resource
def build_knowledge_base():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    all_chunks = []
    chunk_sources = []

    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)

        if text.strip():
            chunks = chunk_text(text, chunk_size=700)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(pdf_file.name)

    if not all_chunks:
        return pdf_files, [], [], None

    embeddings = model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return pdf_files, all_chunks, chunk_sources, index


def retrieve_context(question: str, index, all_chunks, chunk_sources, top_k: int = 5):
    query_embedding = model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k=top_k)

    retrieved_chunks = []
    retrieved_sources = []

    for idx in indices[0]:
        retrieved_chunks.append(all_chunks[idx])
        retrieved_sources.append(chunk_sources[idx])

    context = "\n\n".join(retrieved_chunks)

    unique_sources = []
    for src in retrieved_sources:
        if src not in unique_sources:
            unique_sources.append(src)

    return context, retrieved_chunks, retrieved_sources, unique_sources


def build_prompt(question: str, context: str, chat_history: list[dict], sources: list[str]) -> str:
    history_text = ""
    for msg in chat_history[-6:]:
        role = msg["role"].capitalize()
        history_text += f"{role}: {msg['content']}\n"

    source_list = "\n".join(f"- {src}" for src in sources)

    return f"""
You are an academic assistant helping analyze structural engineering and AI literature.

Use only the provided context to answer the question clearly and accurately.
If the answer is not supported by the context, say so.

When you answer:
1. Synthesize information clearly and academically.
2. If possible, mention which source files support the answer.
3. End your response with a short section titled "Sources Used" and list the relevant source file names.
4. Do not invent citations or papers not provided.

Recent conversation:
{history_text}

Available sources:
{source_list}

Context:
{context}

Question:
{question}
"""


# -------------------------
# Build knowledge base
# -------------------------
pdf_files, all_chunks, chunk_sources, index = build_knowledge_base()

if not pdf_files:
    st.warning("No PDF files were found in the data folder.")
    st.stop()

st.success(f"Loaded {len(pdf_files)} PDFs and {len(all_chunks)} text chunks.")

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("Collection")
    st.write(f"**PDFs loaded:** {len(pdf_files)}")
    st.write(f"**Text chunks:** {len(all_chunks)}")

    with st.expander("Show loaded documents"):
        for pdf in pdf_files:
            st.write(f"- {pdf.name}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# -------------------------
# Chat state
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Display chat history
# -------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        if message["role"] == "assistant":
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.write(f"- {src}")

            if "retrieved_chunks" in message and message["retrieved_chunks"]:
                with st.expander("Retrieved context"):
                    for i, (chunk, source) in enumerate(
                        zip(message["retrieved_chunks"], message["retrieved_sources"]),
                        start=1
                    ):
                        st.markdown(f"**Match {i} — {source}**")
                        st.write(chunk)
                        st.markdown("---")

# -------------------------
# Chat input
# -------------------------
question = st.chat_input("Ask a question about your literature collection")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating answer..."):
            try:
                context, retrieved_chunks, retrieved_sources, unique_sources = retrieve_context(
                    question, index, all_chunks, chunk_sources, top_k=5
                )

                prompt = build_prompt(question, context, st.session_state.messages[:-1], unique_sources)

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content
                # Show the answer first
                st.markdown("### Answer")
                st.write(answer)

                with st.expander("Sources"):
                    for src in unique_sources:
                        st.write(f"- {src}")

                with st.expander("Retrieved context"):
                    for i, (chunk, source) in enumerate(
                        zip(retrieved_chunks, retrieved_sources),
                        start=1
                    ):
                        st.markdown(f"**Match {i} — {source}**")
                        st.write(chunk)
                        st.markdown("---")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": unique_sources,
                        "retrieved_chunks": retrieved_chunks,
                        "retrieved_sources": retrieved_sources,
                    }
                )

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "retrieved_chunks": [],
                        "retrieved_sources": [],
                    }
                )