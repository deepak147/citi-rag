# CITI Bank AI Assistant
A secure, interactive RAG chatbot implementation using modern AI tools and techniques for banking documentation processing.

## Project Structure
```
├── backend/
│   ├── __init__.py
│   └── retrieve.py       # Core RAG implementation
├── config/
│   ├── config.yml        # NeMo Guardrails config
│   ├── flow.co
│   └── prompts.yml       # System prompts
├── ingest/
│   ├── Citi_Marketplace.pdf
│   ├── Client Manual - Consumer Accounts.pdf
│   ├── ingest.py         # Document ingestion
│   └── semantic_ingest.py
├── ragas_evaluate/
│   ├── eval_plot.py
│   ├── evaluate.py
│   ├── testset_generate.py
│   └── evaluation_results.csv
├── main.py               # Streamlit interface
├── poetry.lock
├── pyproject.toml
└── .env
```

## Tech Stack

- **Core**: OpenAI GPT-4, Pinecone Vector Store
- **Security**: NeMo Guardrails
- **Monitoring**: Langfuse
- **Evaluation**: RAGAS
- **Frontend**: Streamlit
- **Environment**: Poetry

## Setup

1. **Environment Setup**
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

2. **Configure Environment Variables**
```bash
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
INDEX_NAME=
LANGCHAIN_API_KEY = 
LANGCHAIN_TRACING_V2 = 
LANGCHAIN_PROJECT = 
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=
```

3. **Data Ingestion**
```bash
poetry run python ingest/ingest.py
```

4. **Launch Application**
```bash
poetry run streamlit run main.py
```

## Features

- Async retrieval with multi-query generation
- Real-time response streaming
- Document evaluation using RAGAS
- Comprehensive security with NeMo Guardrails
- Performance monitoring via Langfuse
- Streamlit-based chat interface

## Development

1. **Document Processing**
- Layout-aware PDF parsing
- Semantic chunking
- Multi-modal content handling

2. **Evaluation**
```bash
poetry run python ragas_evaluate/evaluate.py
```

3. **Monitoring**
Access Langfuse dashboard for:
- Pipeline tracing
- Performance metrics
- Response quality assessment