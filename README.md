# 🦜🔗 LangChain RAG Pipeline — Pinecone + Google Gemini

A **Retrieval-Augmented Generation (RAG)** demo built with LangChain, Pinecone vector store, and Google Gemini. This project compares three approaches to answering questions — raw LLM inference, manual RAG chain, and a declarative LCEL-style RAG chain.

---

## ✨ Features

- 🔍 **Semantic Search** — Retrieve relevant document chunks from Pinecone using vector similarity
- 🤖 **Google Gemini** — Uses `gemini-2.5-flash` as the LLM and `gemini-embedding-001` for embeddings (free tier)
- 🧱 **Two RAG styles** — Manual step-by-step chain vs. declarative LCEL pipeline
- 📥 **Document Ingestion** — Load, split, embed, and upsert any text file into Pinecone
- ⚖️ **Side-by-side comparison** — See how RAG improves over raw LLM responses

---

## 📁 Project Structure

```
Langchain Vector Gist/
├── main.py            # RAG query pipeline (3 comparison modes)
├── ingestion.py       # Document loader + Pinecone upsert
├── mediumblog1.txt    # Sample knowledge base document
├── textsplitters.md   # Notes on LangChain text splitting strategies
├── pyproject.toml     # Project metadata and dependencies (uv)
├── uv.lock            # Locked dependency versions
├── .python-version    # Python version pin
├── .gitignore         # Standard Python gitignore
└── README.md          # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.13+**
- [`uv`](https://github.com/astral-sh/uv) package manager
- A [Pinecone](https://www.pinecone.io/) account and index
- A [Google AI Studio](https://aistudio.google.com/) API key

### 1. Clone the repository

```bash
git clone https://github.com/ahte24/Langchain.git
cd Langchain
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Set up environment variables

Create a `.env` file in the root of the project:

```env
GOOGLE_API_KEY=your_google_api_key_here
INDEX_NAME=your_pinecone_index_name_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

---

## 📥 Ingesting Documents

Before querying, you need to load your documents into Pinecone. The `ingestion.py` script handles this:

```bash
uv run python ingestion.py
```

**What it does:**

1. Loads `mediumblog1.txt` using LangChain's `TextLoader`
2. Splits the text into **1000-character chunks** using `CharacterTextSplitter`
3. Generates embeddings via `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`)
4. Upserts all chunks into your Pinecone index

---

## 🔎 Running the RAG Pipeline

```bash
uv run python main.py
```

The script runs three modes back-to-back for comparison:

| Mode                     | Description                                                      |
| ------------------------ | ---------------------------------------------------------------- |
| **0 — Raw LLM**          | Query sent directly to `gemini-2.5-flash` with no context        |
| **1 — RAG without LCEL** | Manual retrieval → format → prompt → LLM chain                   |
| **2 — RAG with LCEL**    | Declarative pipe-style chain using LangChain Expression Language |

### Example Query

```python
query = "What is Pinecone in machine learning"
```

---

## 🧠 Architecture

```
                    ┌─────────────────────────────────┐
                    │           mediumblog1.txt         │
                    └────────────────┬────────────────┘
                                     │ ingestion.py
                                     ▼
                    ┌─────────────────────────────────┐
                    │     Pinecone Vector Store         │
                    │  (gemini-embedding-001 vectors)   │
                    └────────────────┬────────────────┘
                                     │
              User Query             │  Similarity Search (k=3)
                  │                  ▼
                  │     ┌────────────────────────┐
                  └────►│     Retriever           │
                         └───────────┬────────────┘
                                     │ Retrieved Chunks
                                     ▼
                         ┌───────────────────────┐
                         │   ChatPromptTemplate   │
                         │  (context + question)  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │  gemini-2.5-flash LLM  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                               Final Answer
```

---

## 📦 Dependencies

| Package                  | Purpose                                |
| ------------------------ | -------------------------------------- |
| `langchain`              | Core LangChain framework               |
| `langchain-community`    | Community loaders (TextLoader, etc.)   |
| `langchain-google-genai` | Gemini LLM + Embeddings integration    |
| `langchain-pinecone`     | Pinecone vector store integration      |
| `langchain-ollama`       | Ollama local LLM support               |
| `langchainhub`           | Pull prompts from LangChain Hub        |
| `python-dotenv`          | Load environment variables from `.env` |
| `black`                  | Code formatter                         |
| `isort`                  | Import sorter                          |

---

## 🔗 References

- [LangChain Docs](https://python.langchain.com/docs/)
- [Pinecone Docs](https://docs.pinecone.io/)
- [Google AI Studio](https://aistudio.google.com/)
- [LangChain Expression Language (LCEL)](https://python.langchain.com/docs/expression_language/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
