# `pdf_loader.ipynb` Walkthrough

This notebook continues from document ingestion into a basic RAG pipeline.

It covers:

1. Loading PDFs.
2. Splitting PDF text into chunks.
3. Creating embeddings.
4. Storing embeddings in ChromaDB.
5. Retrieving relevant chunks.
6. Sending retrieved context to an LLM.

The advanced RAG section is intentionally skipped.

## 1. Notebook Heading

```markdown
### RAG Pipelines- Data Ingestion to Vector DB Pipeline
```

This describes the main purpose of the notebook:

```text
PDF data -> document objects -> chunks -> embeddings -> vector database
```

## 2. Imports

```python
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
```

`os` is used for file-system operations.

`PyPDFLoader` and `PyMuPDFLoader` are LangChain PDF loaders.

`RecursiveCharacterTextSplitter` is used to split large text into smaller chunks.

`Path` is Python's modern file-path utility. It makes it easier to search directories and handle file paths.

## 3. Loading All PDFs

```python
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
```

This defines a function that accepts a directory path.

`all_documents` starts as an empty list. It will store all loaded PDF page documents.

`pdf_dir = Path(pdf_directory)` converts the input path string into a `Path` object.

```python
pdf_files = list(pdf_dir.glob("**/*.pdf"))
```

This recursively finds all PDF files inside the directory.

The pattern:

```text
**/*.pdf
```

means:

- search this folder and all subfolders
- match files ending in `.pdf`

```python
print(f"Found {len(pdf_files)} PDF files to process")
```

This prints how many PDF files were found.

## 4. Processing Each PDF

```python
for pdf_file in pdf_files:
    print(f"\nProcessing: {pdf_file.name}")
    try:
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
```

The loop processes one PDF at a time.

`pdf_file.name` gives only the file name, not the full path.

`PyPDFLoader(str(pdf_file))` creates a PDF loader for that file.

`loader.load()` extracts text and metadata from the PDF.

Usually, each PDF page becomes one LangChain `Document`.

For example, a 10-page PDF may produce:

```python
[
    Document(page_content="page 1 text", metadata={"page": 0, ...}),
    Document(page_content="page 2 text", metadata={"page": 1, ...}),
    ...
]
```

## 5. Adding Metadata

```python
for doc in documents:
    doc.metadata["source_file"] = pdf_file.name
    doc.metadata["file_type"] = "pdf"
```

This adds custom metadata to each page document.

Example:

```python
{
    "source": "../data/pdf/attention.pdf",
    "page": 0,
    "source_file": "attention.pdf",
    "file_type": "pdf"
}
```

This metadata becomes useful later for:

- citations
- filtering by file
- debugging retrieval results
- showing source pages to users

## 6. Collecting All Documents

```python
all_documents.extend(documents)
print(f"  Loaded {len(documents)} pages")
```

`extend()` adds every page document from the current PDF into `all_documents`.

This is different from `append()`.

If `documents` has 10 page documents:

```python
all_documents.extend(documents)
```

adds all 10 page documents individually.

At the end:

```python
print(f"\nTotal documents loaded: {len(all_documents)}")
return all_documents
```

The function returns all loaded page-level documents.

## 7. Running the PDF Loader

```python
all_pdf_documents = process_all_pdfs("../data")
```

This searches for all PDFs under:

```text
../data
```

The output is:

```python
all_pdf_documents
```

which is a list of LangChain `Document` objects.

At this point, the system has page-level documents, but pages may be too large or too unfocused for good RAG retrieval.

That is why the next step is text splitting.

## 8. Defining the Text Splitting Function

```python
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
```

This function splits documents into smaller text chunks.

Chunking is important because:

- LLMs have context limits.
- Vector search works better with focused chunks.
- A full PDF page may contain multiple unrelated ideas.
- Smaller chunks improve retrieval precision.

## 9. Creating the RecursiveCharacterTextSplitter

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
```

`chunk_size=1000` means the splitter tries to create chunks around 1000 characters.

`chunk_overlap=200` means each chunk shares 200 characters with the previous chunk.

Example:

```text
Chunk 1: characters 0-1000
Chunk 2: characters 800-1800
Chunk 3: characters 1600-2600
```

The overlap helps preserve context across chunk boundaries.

`length_function=len` means character count is used to measure chunk size.

`separators=["\n\n", "\n", " ", ""]` tells the splitter where to split text, in priority order:

1. paragraph breaks
2. line breaks
3. spaces
4. individual characters

This is why it is called recursive. It tries cleaner separators first, then falls back to smaller separators if needed.

## 10. Splitting the Documents

```python
split_docs = text_splitter.split_documents(documents)
```

This converts page-level documents into chunk-level documents.

Important: the metadata from the original document is preserved.

So a chunk still knows:

- source file
- page number
- file type

Then:

```python
print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
```

This shows how many page documents became how many chunks.

The notebook also prints an example chunk:

```python
if split_docs:
    print(f"\nExample chunk:")
    print(f"Content: {split_docs[0].page_content[:200]}...")
    print(f"Metadata: {split_docs[0].metadata}")
```

This is a sanity check.

It shows:

- the first 200 characters of the first chunk
- metadata attached to that chunk

Finally:

```python
return split_docs
```

## 11. Running the Splitter

```python
chunks = split_documents(all_pdf_documents)
chunks
```

Now the flow is:

```text
PDF files -> page documents -> text chunks
```

These chunks are the units that will be embedded and stored in the vector database.

## 12. Embedding and Vector Store Imports

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
```

Important imports:

`numpy` stores numerical arrays.

`SentenceTransformer` loads an embedding model.

`chromadb` provides the vector database.

`uuid` creates unique document IDs.

`List`, `Dict`, and `Any` are type hints.

Some imports are not used in the shown basic code:

- `Settings`
- `Tuple`
- `cosine_similarity`

## 13. EmbeddingManager Class

```python
class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
```

This class wraps embedding-model loading and embedding generation.

## 14. Initializing the Embedding Manager

```python
def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
```

The default embedding model is:

```text
all-MiniLM-L6-v2
```

This is a small, commonly used Sentence Transformers model.

It converts text into numeric vectors. Texts with similar meanings should produce vectors that are close together.

```python
self.model_name = model_name
self.model = None
self._load_model()
```

The model name is saved.

`self.model` starts as `None`.

`self._load_model()` loads the model.

## 15. Loading the Embedding Model

```python
def _load_model(self):
    try:
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    except Exception as e:
        print(f"Error loading model {self.model_name}: {e}")
        raise
```

This method loads the Sentence Transformer model.

For `all-MiniLM-L6-v2`, the embedding dimension is typically 384.

That means each text chunk becomes a vector with 384 numbers.

Example conceptually:

```python
"What is attention?" -> [0.02, -0.13, 0.45, ...]
```

## 16. Generating Embeddings

```python
def generate_embeddings(self, texts: List[str]) -> np.ndarray:
```

This method accepts a list of text strings.

```python
if not self.model:
    raise ValueError("Model not loaded")
```

This prevents embedding generation if the model did not load.

```python
embeddings = self.model.encode(texts, show_progress_bar=True)
```

This converts every text into a vector.

If you pass 100 chunks, the shape may be:

```python
(100, 384)
```

Meaning:

- 100 text chunks
- 384 numbers per chunk

Then:

```python
return embeddings
```

## 17. Creating the Embedding Manager

```python
embedding_manager = EmbeddingManager()
embedding_manager
```

This creates an `EmbeddingManager` object and loads the embedding model.

## 18. VectorStore Class

```python
class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
```

This class wraps ChromaDB operations.

It handles:

- creating/opening a Chroma database
- creating/opening a collection
- adding documents and embeddings

## 19. Initializing the Vector Store

```python
def __init__(
    self,
    collection_name: str = "pdf_documents",
    persist_directory: str = "../data/vector_store"
):
```

The default collection name is:

```text
pdf_documents
```

The database is stored at:

```text
../data/vector_store
```

Then:

```python
self.collection_name = collection_name
self.persist_directory = persist_directory
self.client = None
self.collection = None
self._initialize_store()
```

The object stores configuration and then initializes ChromaDB.

## 20. Initializing ChromaDB

```python
os.makedirs(self.persist_directory, exist_ok=True)
self.client = chromadb.PersistentClient(path=self.persist_directory)
```

This creates the vector-store folder if needed.

`PersistentClient` means the vector database is saved on disk and can be reused later.

```python
self.collection = self.client.get_or_create_collection(
    name=self.collection_name,
    metadata={"description": "PDF document embeddings for RAG"}
)
```

This either opens an existing Chroma collection or creates a new one.

A Chroma collection stores:

- document IDs
- text chunks
- embeddings
- metadata

```python
print(f"Existing documents in collection: {self.collection.count()}")
```

This shows how many documents are already stored.

## 21. Adding Documents to the Vector Store

```python
def add_documents(self, documents: List[Any], embeddings: np.ndarray):
```

This method adds document chunks and their embeddings into ChromaDB.

```python
if len(documents) != len(embeddings):
    raise ValueError("Number of documents must match number of embeddings")
```

There must be exactly one embedding per document chunk.

Then it prepares four lists:

```python
ids = []
metadatas = []
documents_text = []
embeddings_list = []
```

ChromaDB needs:

- unique IDs
- metadata
- raw document text
- embedding vectors

Inside the loop:

```python
doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
ids.append(doc_id)
```

This creates a unique ID for each chunk.

```python
metadata = dict(doc.metadata)
metadata["doc_index"] = i
metadata["content_length"] = len(doc.page_content)
metadatas.append(metadata)
```

This copies the chunk metadata and adds:

- chunk index
- content length

```python
documents_text.append(doc.page_content)
embeddings_list.append(embedding.tolist())
```

The raw text is stored.

The embedding is converted from a NumPy array to a normal Python list because ChromaDB expects list-like data.

Finally:

```python
self.collection.add(
    ids=ids,
    embeddings=embeddings_list,
    metadatas=metadatas,
    documents=documents_text
)
```

This writes everything to the vector database.

## 22. Creating the Vector Store

```python
vectorstore = VectorStore()
vectorstore
```

This initializes ChromaDB and opens/creates the `pdf_documents` collection.

## 23. Converting Chunks to Embeddings

```python
texts = [doc.page_content for doc in chunks]
```

This extracts plain text from each chunk.

```python
embeddings = embedding_manager.generate_embeddings(texts)
```

This converts every chunk into a vector.

```python
vectorstore.add_documents(chunks, embeddings)
```

This stores:

- chunk text
- chunk embedding
- chunk metadata
- generated ID

At this point, the searchable knowledge base exists.

## 24. Retriever Pipeline

```python
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
```

This class searches ChromaDB for chunks relevant to a user query.

## 25. Initializing the Retriever

```python
def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
    self.vector_store = vector_store
    self.embedding_manager = embedding_manager
```

The retriever needs:

- the vector store containing document embeddings
- the embedding manager to embed user queries

The query and document chunks must use the same embedding model.

## 26. Retrieval Method

```python
def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
```

Inputs:

`query` is the user's search question.

`top_k` is the number of results to return.

`score_threshold` is the minimum similarity score required.

## 27. Embedding the Query

```python
query_embedding = self.embedding_manager.generate_embeddings([query])[0]
```

The query is converted into an embedding vector.

The query is wrapped in a list because `generate_embeddings()` expects a list of strings.

`[0]` extracts the first embedding from the returned array.

## 28. Searching ChromaDB

```python
results = self.vector_store.collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=top_k
)
```

This searches the vector store for the closest document embeddings.

ChromaDB returns:

- IDs
- documents
- metadata
- distances

## 29. Processing Retrieval Results

```python
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]
ids = results["ids"][0]
```

The result arrays are nested because ChromaDB supports multiple queries at once.

Since this notebook sends one query, it uses `[0]` to access the first query's results.

Then:

```python
similarity_score = 1 - distance
```

The code treats ChromaDB's returned distance as cosine distance.

Lower distance means more similar.

Higher `similarity_score` means more relevant.

Then it creates a clean dictionary for each result:

```python
{
    "id": doc_id,
    "content": document,
    "metadata": metadata,
    "similarity_score": similarity_score,
    "distance": distance,
    "rank": i + 1
}
```

This makes retrieval output easier to use later.

## 30. Creating and Testing the Retriever

```python
rag_retriever = RAGRetriever(vectorstore, embedding_manager)
```

This connects the vector database and embedding model.

Then:

```python
rag_retriever.retrieve("What is attention is all you need")
rag_retriever.retrieve("Unified Multi-task Learning Framework")
```

These test semantic search.

The retriever is not only doing exact keyword search. It searches for chunks whose embeddings are semantically close to the query.

## 31. Loading Environment Variables

```python
from dotenv import load_dotenv
load_dotenv()

print(os.getenv("GROQ_API_KEY"))
```

This loads environment variables from a `.env` file.

The notebook expects:

```text
GROQ_API_KEY=your_api_key
```

Printing the API key is not recommended in real projects. It is safer to only check whether the key exists.

## 32. Importing Groq/LangChain Classes

```python
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
```

`ChatGroq` lets LangChain call a Groq-hosted chat model.

`PromptTemplate` helps format prompts.

`HumanMessage` wraps user messages for LangChain chat models.

`SystemMessage` is imported but not used in the shown basic class.

## 33. GroqLLM Class

```python
class GroqLLM:
```

This class wraps LLM response generation.

```python
def __init__(self, model_name: str = "gemma2-9b-it", api_key: str = None):
```

The default Groq model is:

```text
gemma2-9b-it
```

```python
self.api_key = api_key or os.environ.get("GROQ_API_KEY")
```

The class uses the passed API key or reads it from the environment.

```python
if not self.api_key:
    raise ValueError("Groq API key is required...")
```

If no key exists, the class raises an error.

```python
self.llm = ChatGroq(
    groq_api_key=self.api_key,
    model_name=self.model_name,
    temperature=0.1,
    max_tokens=1024
)
```

This creates the LLM client.

`temperature=0.1` makes answers more deterministic.

`max_tokens=1024` limits answer length.

## 34. Generating an Answer with Context

```python
def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
```

This method receives:

- the user question
- retrieved context
- a maximum length argument, though `max_length` is not actually used inside the method

It creates a prompt:

```python
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
)
```

This is the RAG generation step:

```text
retrieved context + user question -> LLM answer
```

Then:

```python
formatted_prompt = prompt_template.format(context=context, question=query)
messages = [HumanMessage(content=formatted_prompt)]
response = self.llm.invoke(messages)
return response.content
```

The prompt is formatted, wrapped as a human message, sent to the LLM, and the response text is returned.

## 35. Simple Response Generation

```python
def generate_response_simple(self, query: str, context: str) -> str:
```

This method does the same basic thing with a simpler prompt:

```python
simple_prompt = f"""Based on this context: {context}

Question: {query}

Answer:"""
```

Then it sends the prompt to the model and returns the answer.

## 36. Initializing GroqLLM

```python
try:
    groq_llm = GroqLLM(api_key=os.getenv("GROQ_API_KEY"))
    print("Groq LLM initialized successfully!")
except ValueError as e:
    print(f"Warning: {e}")
    print("Please set your GROQ_API_KEY environment variable to use the LLM.")
    groq_llm = None
```

This tries to initialize the LLM.

If no API key exists, it does not crash the whole notebook. Instead, it sets:

```python
groq_llm = None
```

## 37. Retrieving Context Before Generation

```python
rag_retriever.retrieve("Unified Multi-task Learning Framework")
```

This retrieves relevant chunks.

At this point, it does not yet pass the chunks to the LLM. It only verifies retrieval.

## 38. Simple Integrated RAG Pipeline

```python
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
```

This imports Groq and loads environment variables again.

```python
groq_api_key = os.getenv("GROQ_API_KEY")
```

Reads the Groq API key.

```python
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="gemma2-9b-it",
    temperature=0.1,
    max_tokens=1024
)
```

Creates the Groq LLM directly, without using the `GroqLLM` wrapper class.

## 39. `rag_simple()` Function

```python
def rag_simple(query, retriever, llm, top_k=3):
```

This function combines retrieval and generation.

Step 1: retrieve relevant chunks.

```python
results = retriever.retrieve(query, top_k=top_k)
```

Step 2: combine retrieved chunk text into one context string.

```python
context = "\n\n".join([doc["content"] for doc in results]) if results else ""
```

Step 3: handle the case where no context is found.

```python
if not context:
    return "No relevant context found to answer the question."
```

Step 4: create a prompt.

```python
prompt = f"""Use the following context to answer the question concisely.
    Context:
    {context}

    Question: {query}

    Answer:"""
```

Step 5: call the LLM.

```python
response = llm.invoke([prompt.format(context=context, query=query)])
return response.content
```

Small note: because `prompt` is already an f-string, the later `.format(...)` is unnecessary.

Cleaner alternatives:

```python
response = llm.invoke(prompt)
```

or:

```python
response = llm.invoke([HumanMessage(content=prompt)])
```

depending on the LangChain version.

## 40. Running the Simple RAG Pipeline

```python
answer = rag_simple("What is attention mechanism?", rag_retriever, llm)
print(answer)
```

This runs the full basic RAG flow:

```text
Question
 -> embed question
 -> retrieve similar chunks from ChromaDB
 -> combine chunks into context
 -> send context and question to Groq LLM
 -> print answer
```

## Main Lesson from `pdf_loader.ipynb`

The notebook builds a basic end-to-end RAG pipeline:

```text
PDFs
 -> PDF page documents
 -> smaller chunks
 -> embeddings
 -> ChromaDB vector store
 -> query embedding
 -> semantic retrieval
 -> retrieved context
 -> LLM-generated answer
```

This is the core RAG process.

