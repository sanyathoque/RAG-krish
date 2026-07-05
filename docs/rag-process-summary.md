# End-to-End RAG Process Summary

This is the complete basic process taught across the two notebooks.

## 1. Start with Raw Data

The data can come from:

- manually written text
- `.txt` files
- a folder of text files
- PDF files

Raw data is not yet ready for RAG. It first needs to be loaded into a consistent structure.

## 2. Convert Data into LangChain Documents

LangChain uses `Document` objects.

Each document has:

```python
page_content
metadata
```

`page_content` contains the actual text.

`metadata` contains information such as:

- source file
- page number
- file type
- author
- creation date

This metadata is useful later for citations and debugging.

## 3. Split Long Documents into Chunks

Full pages or full documents can be too large.

The notebooks use:

```python
RecursiveCharacterTextSplitter
```

with:

```python
chunk_size=1000
chunk_overlap=200
```

This creates smaller, overlapping chunks.

Chunking improves retrieval because each chunk is more focused.

## 4. Convert Chunks into Embeddings

The notebook uses:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

An embedding model converts text into a vector of numbers.

Conceptually:

```text
"What is attention?" -> [0.02, -0.13, 0.45, ...]
```

Similar meanings should produce similar vectors.

## 5. Store Embeddings in ChromaDB

The notebook uses ChromaDB as the vector database.

Each stored item contains:

- ID
- chunk text
- embedding vector
- metadata

This makes semantic search possible.

## 6. Retrieve Relevant Chunks

When a user asks a question:

1. The question is embedded using the same embedding model.
2. ChromaDB compares the query embedding against stored chunk embeddings.
3. The closest chunks are returned.

This is semantic retrieval.

The query does not need to exactly match the words in the documents. It only needs to be close in meaning.

## 7. Send Context to the LLM

The retrieved chunks are joined into a context string.

Then the prompt looks like:

```text
Use the following context to answer the question.

Context:
...

Question:
...

Answer:
```

The LLM answers using the retrieved context.

## 8. Basic RAG Pipeline

The full process is:

```text
User question
 -> query embedding
 -> vector search
 -> retrieved chunks
 -> context prompt
 -> LLM answer
```

This is the basic RAG architecture.

## What Was Skipped

The advanced RAG sections were skipped as requested.

Skipped topics include:

- enhanced answer objects
- confidence scores
- source previews
- streaming simulation
- conversation history
- summarization
- advanced citations

