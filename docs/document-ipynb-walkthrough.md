# `document.ipynb` Walkthrough

This notebook is mainly about data ingestion: how raw content becomes LangChain `Document` objects.

## 1. Data Ingestion

The notebook starts with the markdown heading:

```markdown
### Data Ingestion
```

This tells you the first stage of RAG is getting data into a standard format.

In LangChain, that standard format is usually a list of `Document` objects.

## 2. Importing the Document Class

```python
from langchain_core.documents import Document
```

This imports LangChain's core `Document` class.

A LangChain `Document` normally has two important parts:

```python
page_content
metadata
```

`page_content` is the actual text that will later be searched, embedded, retrieved, and passed to an LLM.

`metadata` stores extra information about the content, such as source file, page number, author, or creation date.

## 3. Creating a Document Manually

```python
doc = Document(
    page_content="this is the main text content I am using to create RAG",
    metadata={
        "source": "exmaple.txt",
        "pages": 1,
        "author": "Krish Naik",
        "date_created": "2025-01-01"
    }
)
doc
```

This manually creates a LangChain document.

The important structure is:

```python
Document(
    page_content="actual searchable text",
    metadata={
        "source": "file name",
        "pages": page count,
        "author": author name
    }
)
```

In RAG:

- `page_content` is the text used for search and answer generation.
- `metadata` is useful for citations, filtering, debugging, and showing where an answer came from.

Small typo in the notebook:

```python
"exmaple.txt"
```

probably means:

```python
"example.txt"
```

## 4. Creating a Folder for Text Files

```python
import os
os.makedirs("../data/text_files", exist_ok=True)
```

This imports Python's built-in `os` module and creates a folder:

```text
../data/text_files
```

`exist_ok=True` means Python should not throw an error if the folder already exists.

This prepares a location where sample `.txt` files will be stored.

## 5. Creating Sample Text Files

```python
sample_texts = {
    "../data/text_files/python_intro.txt": """Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",

    "../data/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems
    """
}
```

This dictionary maps file paths to file contents.

For example:

```python
"../data/text_files/python_intro.txt"
```

is the file path.

The long triple-quoted string is the content that will be written into that file.

Then the notebook writes each text value into its corresponding file:

```python
for filepath, content in sample_texts.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Sample text files created!")
```

The loop does this:

1. Take a file path and text content from `sample_texts`.
2. Open the file in write mode.
3. Write the content into the file.
4. Move to the next file.

After this cell runs, the folder contains:

```text
python_intro.txt
machine_learning.txt
```

These are raw text files. The next step is to load them into LangChain.

## 6. Loading One Text File with TextLoader

```python
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
```

The first import is the older import path. The second import is the newer community package path.

Only this one is really needed:

```python
from langchain_community.document_loaders import TextLoader
```

Then:

```python
loader = TextLoader("../data/text_files/python_intro.txt", encoding="utf-8")
document = loader.load()
print(document)
```

`TextLoader` reads a single text file and turns it into LangChain `Document` objects.

Even for one file, `loader.load()` returns a list:

```python
[
    Document(
        page_content="Python Programming Introduction...",
        metadata={"source": "../data/text_files/python_intro.txt"}
    )
]
```

The key idea is that LangChain loaders usually return lists of `Document` objects, not plain strings.

## 7. Loading Many Text Files with DirectoryLoader

```python
from langchain_community.document_loaders import DirectoryLoader
```

`DirectoryLoader` loads many files from a folder.

```python
dir_loader = DirectoryLoader(
    "../data/text_files",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=False
)
```

Meaning:

```python
"../data/text_files"
```

Look inside this folder.

```python
glob="**/*.txt"
```

Find all `.txt` files recursively.

```python
loader_cls=TextLoader
```

Use `TextLoader` to load each matching file.

```python
loader_kwargs={"encoding": "utf-8"}
```

Pass UTF-8 encoding into each `TextLoader`.

```python
show_progress=False
```

Do not show a progress bar.

Then:

```python
documents = dir_loader.load()
documents
```

This loads both text files and returns a list like:

```python
[
    Document(page_content="Machine Learning Basics...", metadata={...}),
    Document(page_content="Python Programming Introduction...", metadata={...})
]
```

This is the multi-file version of the previous single-file loading step.

## 8. Loading PDF Files

```python
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
```

This imports PDF loaders.

`PyPDFLoader` uses the `pypdf` ecosystem.

`PyMuPDFLoader` uses PyMuPDF, which is often fast and can work well for PDF text extraction.

Then:

```python
dir_loader = DirectoryLoader(
    "../data/pdf",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=False
)
```

This tells LangChain:

1. Look inside `../data/pdf`.
2. Find all `.pdf` files.
3. Load each PDF using `PyMuPDFLoader`.

Then:

```python
pdf_documents = dir_loader.load()
pdf_documents
```

The result is a list of LangChain `Document` objects.

For PDFs, each page usually becomes a separate `Document`.

So a 15-page PDF may produce 15 documents:

```python
[
    Document(page_content="text from page 1", metadata={"page": 0, ...}),
    Document(page_content="text from page 2", metadata={"page": 1, ...}),
    ...
]
```

Each page-level document usually contains:

- extracted page text
- source file path
- page number
- PDF metadata

## 9. Checking the Type

```python
type(pdf_documents[0])
```

This confirms that the first PDF page is a LangChain `Document`:

```text
langchain_core.documents.base.Document
```

## Main Lesson from `document.ipynb`

This notebook teaches that different raw sources can be normalized into the same structure:

```text
manual text
text files
folders of text files
PDF files
```

all become:

```python
List[Document]
```

That list of `Document` objects is the starting point for RAG.

