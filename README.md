# LLM Document Question-Answering System

A production-grade Retrieval-Augmented Generation (RAG) service that turns any document (PDF, DOCX, TXT, E-mail, etc.) into a searchable knowledge base and answers natural-language questions with sourced, confidence-scored responses.

---

## 🌐 High-Level Architecture

User → FastAPI → Query Embedding
↘──────────────┐
Pinecone Vector DB ← Chunk Embeddings
│
Top-k Relevant Chunks
│
Adaptive Answer Extractor
│
JSON Response



| Layer                 | Main Responsibilities | Key Tech |
|-----------------------|-----------------------|----------|
| **Document Ingestion**| Download, OCR, text extraction, cleaning | `pdfminer`, `tesseract`, `docx2txt` |
| **Chunking**          | 1,200-char chunks ±150-char overlap, metadata tagging | Custom Python |
| **Embedding**         | 768-dim semantic vectors | Google Gemini `models/embedding-001` |
| **Vector Storage**    | Similarity search, namespace isolation | Pinecone |
| **Query Processor**   | Embed question, semantic search, re-rank | FastAPI async |
| **Answer Engine**     | Multi-strategy extraction + LLM generation (optional) | Custom “AdaptiveAnswerExtractor” |
| **API Gateway**       | Auth, rate-limit, CORS, JSON API | FastAPI |

---

## ⚙️ Flow Pipeline

1. **Upload / URL Input**  
2. **Document Processor**  
    • Detect format • Convert to text • Clean headers/footers  
3. **Chunk & Embed**  
    • Context-aware splitting • Gemini embeddings  
4. **Vector Upsert**  
    • Store chunks + metadata in Pinecone  
5. **Query**  
    • Embed question • Retrieve top-k chunks (cos sim ≥ 0.45)  
6. **Adaptive Extraction**  
    • Semantic, numerical, structural, contextual strategies  
    • Confidence scoring → best answer  
7. **Response**  
    • `{"answers": ["…", "…"]}` returned to caller  

---

## 🗄️ Project Structure

├── api/
│ ├── endpoints.py # FastAPI routes
│ └── auth.py # Token auth
├── services/
│ ├── document_processor.py # Ingestion & cleaning
│ ├── vector_store.py # Pinecone wrapper
│ ├── adaptive_extractor.py # Smart answer finder
│ └── gemini_service.py # Google Gemini calls
├── models/
│ ├── database_models.py # SQLAlchemy ORM
│ └── pydantic_models.py # API schemas
├── database.py # Postgres engine
├── config.py # Env settings
├── main.py # Uvicorn entry-point
└── README.md




---

## 🚀 Quick Start

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env # edit DB, Pinecone, Gemini keys

uvicorn main:app --host 0.0.0.0 --port 8000 --reload


---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Postgres connection string |
| `PINECONE_API_KEY` / `PINECONE_ENV` | Pinecone credentials |
| `GEMINI_API_KEY` | Google Gemini key |
| `VECTOR_DIMENSION` | Embedding size (default 768) |
| `LOG_LEVEL` | `INFO`, `DEBUG`, etc. |

---

## 🛠️ Tech Stack

- **FastAPI** • **Pydantic** • **SQLAlchemy**  
- **Google Gemini Embedding**  
- **Pinecone Vector DB**  
- **Docker & Poetry** for packaging  
- **Alembic** for migrations

---

## 📈 Performance

| Metric                        | Typical Value |
|-------------------------------|---------------|
| PDF processing (20 pages)     | 15-30 s |
| Query latency (top-k=5)       | 1-3 s |
| Answer accuracy (structured)  | 85-95 % |

---

## 🧩 Extensibility Roadmap

1. **Multi-modal** (tables, images)  
2. **Feedback loop** → fine-tune extraction thresholds  
3. **Streaming chat UI** with WebSockets  
4. **Micro-services split** (ingest vs. query)  
5. **Model swapping** (OpenAI, Cohere, Llama-CPP)  

---

## 🤝 Contributing

1. Fork & branch (`feat/<name>`)  
2. Run `pre-commit install` for linting  
3. Add tests (`pytest`)  
4. Open a PR—describe changes clearly  

---


