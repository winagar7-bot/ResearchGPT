# ResearchGPT

I built this project to tackle a major headache with standard AI document chat tools: hallucinations and vague answers. Instead of trusting an LLM to guess where it found information, this system acts as a true research assistant. It shreds scientific papers into smart text chunks, runs a quick semantic search, and roots every single answer in exact context with verified, page-level citations.

It gives you the technical depth companies want for high-level AI roles, but without the over-complicated infrastructure.

## What It Does

* **Precise Citations:** Maps every single chunk of text back to its original page number, so you can always check the source.
* **Fast Local Search:** Uses optimized `bge-small-en-v1.5` embeddings to find the right information in milliseconds without relying on heavy cloud APIs.
* **Smart Data Splitting:** Chunks text intelligently so thoughts aren't cut in half across arbitrary page boundaries.
* **Strict Answer Guardrails:** Forces the LLM to stay within the bounds of your documents. If the answer isn't in the text, it won't make it up.

---

##  Tech Stack

* **Orchestration:** LangChain
* **Embeddings:** Hugging Face (`BAAI/bge-small-en-v1.5`)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **PDF Processing:** PyMuPDF (`fitz`)
* **Language:** Python

---

##  How It Works

```
[Academic PDF] ──> [PyMuPDF Parser] ──> [Smart Text Chunking]
                                               │
                                               ▼ (Keeps Source File & Page Number)
[User Question] ──> [BGE Embeddings] ──> [FAISS Vector Index]
     │                                         │
     ▼                                         ▼
[Strict LLM Prompt Template] <─────── [Top Relevant Text Blocks]

```

---

##  Quick Start (Google Colab / Local Setup)

### 1. Grab the Dependencies

Run this in your terminal or a notebook cell to get set up:

```bash
pip install langchain langchain-community langchain-huggingface faiss-cpu pymupdf sentence-transformers

```

### 2. Run an End-to-End Test

Here is a quick script to see the backend, vector storage, and query tracking working in harmony:

```python
from langchain_core.documents import Document

# 1. Initialize the engine and embeddings
engine = ResearchGPTEngine(embedding_model="BAAI/bge-small-en-v1.5")

# 2. Add some mock papers with exact metadata to test page tracking
documents = [
    Document(
        page_content="Our architecture uses a dynamic vector routing layer that lowers computational complexity from O(N^2) to O(N log N).", 
        metadata={"source": "neural_ctx_2026.pdf", "page": 1}
    )
]

# 3. Index everything into the FAISS database
engine.build_vector_index(documents)

# 4. Ask a question and see the clean prompt payload ready for an LLM
payload = ask_research_assistant("What is the computational complexity?", engine)
print(payload["prompt_ready_for_llm"])

```

---

##  Core Methods API

### `build_vector_index(documents: List[Document])`

Takes your processed document chunks, converts them into high-quality vector embeddings, and builds a searchable local FAISS index.

### `retrieve_context(query: str, k: int)`

Runs a direct similarity match to pull the top `k` most relevant text chunks that match what the user is asking.

### `format_context_with_citations(docs: List[Document])`

Cleans up the raw data matches and splits them into two clean payloads: a clean text block to feed the LLM, and a tidy JSON structure for frontend UI citation chips.

---

##  Talking Points for Interviews

If you are showcasing this project to hiring managers or engineering teams, here are the exact engineering choices you should talk about:

1. **Token Window Planning:** Text split strategies use a 1,000-character chunk limit with a 200-character overlap. This keeps ideas from getting cut in half without wasting unnecessary context tokens.
2. **Local Vector Storage over Cloud APIs:** Choosing a local, in-memory FAISS setup over a cloud database like Pinecone keeps costs at zero and drops internal database search latency below 50ms.
3. **Beating Hallucinations at the Source:** Instead of asking an LLM to remember where it found something, this engine handles citations strictly at the database level before the prompt is ever created. It forces absolute transparency.
