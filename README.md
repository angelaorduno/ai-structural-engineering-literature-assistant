# 💜 AI Structural Engineering Literature Assistant

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff69b4)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-purple)
![LLM](https://img.shields.io/badge/LLM-GPT--4o--mini-pink)
![Status](https://img.shields.io/badge/Project-Active-violet)

</p>

---

## 🌸 Overview

The **AI Structural Engineering Literature Assistant** is an AI-powered research tool that allows users to upload or query multiple academic papers and ask natural language questions about their contents.

The system performs **semantic search across engineering literature** and generates answers using a **Retrieval Augmented Generation (RAG)** pipeline.

Designed for:

- Civil engineering researchers
- Structural engineering students
- AI + engineering interdisciplinary research
- Literature reviews and research discovery

---

## 🧠 How It Works

The system combines several modern AI components:

```
PDF Papers
   ↓
Text Extraction
   ↓
Chunking
   ↓
Sentence Embeddings
   ↓
FAISS Vector Search
   ↓
Relevant Context Retrieval
   ↓
LLM Question Answering
```

This architecture enables the assistant to **search across multiple research papers simultaneously** and provide contextual answers.

---

## ⚙️ Technologies Used

| Technology | Purpose |
|------------|--------|
| **Python** | Core programming language |
| **Streamlit** | Interactive web interface |
| **SentenceTransformers** | Document embeddings |
| **FAISS** | Vector similarity search |
| **OpenAI GPT Models** | Context-aware question answering |
| **PyPDF** | PDF text extraction |

---

## 📂 Project Structure

```
ai-structural-engineering-literature-assistant
│
├── data/               # Research papers (PDFs)
├── src/
│   ├── app.py          # Streamlit interface
│   ├── chunk_text.py
│   ├── create_embeddings.py
│   ├── search_chunks.py
│   └── ask_document.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Running the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/angelaorduno/ai-structural-engineering-literature-assistant.git
cd ai-structural-engineering-literature-assistant
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add OpenAI API key

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

---

### 5️⃣ Run the app

```bash
streamlit run src/app.py
```

The app will open at:

```
http://localhost:8501
```

---

## 💬 Example Questions

Users can ask questions such as:

- What machine learning methods are used in structural health monitoring?
- What are the challenges of explainable AI in civil engineering?
- How are GANs applied in structural design automation?
- What optimization methods are used in shear wall design?

---

## 🌟 Features

✔ Multi-document semantic search  
✔ Natural language question answering  
✔ AI-assisted literature review  
✔ Engineering-focused knowledge retrieval  
✔ Interactive Streamlit interface

---

## 🔬 Research Applications

This tool can support:

- Literature reviews
- Engineering research exploration
- Academic paper synthesis
- AI-assisted engineering design research

---

## 👩‍💻 Author

**Angela Diaz**  
PhD Candidate — Data Science  
Focus: **AI, Optimization, and Applied Engineering Systems**

---

## 💡 Future Improvements

- Multi-paper citation tracking
- PDF highlighting of source passages
- Research graph visualization
- Cloud deployment
- Larger document collections

---

<p align="center">

💜 *Built with AI for the future of engineering research*

</p>
