# IntentGuard

IntentGuard is a LangGraph-based AI agent that classifies user query intents, provides RAG (Retrieval-Augmented Generation) responses, and escalates queries requiring human review for admin approval. It uses Azure OpenAI for LLM interactions, FAISS for vector search, and SQLite for storing flagged queries.

## Prerequisites

- Python 3.9+
- Azure OpenAI account with endpoint, deployment, and API key
- A PDF document for building the RAG index 

## Installation

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Environment Setup

A `.env` template file is provided in the root directory. Fill in your actual Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=<your_endpoint>
AZURE_OPENAI_API_KEY=<your_api_key>
AZURE_OPENAI_API_VERSION=<your_api_version>
AZURE_OPENAI_DEPLOYMENT=<your_model_deployment_name>
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

VECTOR_STORE_PATH=index.faiss
MEMORY_INDEX_PATH=memory_index.faiss
DB_PATH=flagged_queries.db
```

## Creating the RAG Index

The agent uses a FAISS vector index for RAG. A sample `policy.pdf` is included for testing. To build the index:

1. Use your own PDF document (replace `policy.pdf` with your file).
2. Run the index creation (the sample code in `rag.py` is set up for `policy.pdf`):

   ```
   python rag.py
   ```

   Or modify `rag.py` to point to your document:

   ```python
   # In rag.py __main__ block, change:
   build_and_save_index(r"path/to/your/document.pdf", r'index.faiss')
   ```

   The `build_and_save_index` function:
   - Loads the PDF using PyPDFLoader.
   - Splits the text into chunks (1000 chars with 150 overlap).
   - Embeds the chunks using Azure OpenAI embeddings.
   - Saves the FAISS index to `index.faiss` (default path).

## Running the Agent

### User Flow

To interact as a user:

1. Ensure the RAG index exists (see above).
2. Run:
   ```
   python run_user.py
   ```
3. Enter queries when prompted. The agent will classify intent, provide RAG responses, or escalate if needed.
4. Type 'exit' to quit.

### Admin Flow

To review and approve/reject escalated queries:

1. Run:
   ```
   python run_admin.py
   ```
2. The agent fetches pending flagged queries from the DB.
3. For each query, review the rationale, decide 'approve' or 'reject', and provide a comment.
4. Decisions are recorded in the memory index for future learning.
5. Type 'exit' or press Enter to continue reviewing; it exits when no queries remain.

