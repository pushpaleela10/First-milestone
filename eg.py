import os
import shutil

# Document Loaders
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    CSVLoader
)

# Text Splitter (NEW IMPORT)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Chroma (NEW IMPORT LOCATION)
from langchain_community.vectorstores import Chroma



# 1. Startup Cleanup

if os.path.exists("chroma_db"):
    shutil.rmtree("chroma_db")



# 2. Load Documents
pdf_loader = DirectoryLoader(
    path="./Books",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

text_loader = DirectoryLoader(
    path="./Books",
    glob="**/*.txt",
    loader_cls=TextLoader
)

csv_loader = DirectoryLoader(
    path="./Books",
    glob="**/*.csv",
    loader_cls=CSVLoader
)

docs = []
docs.extend(pdf_loader.load())
docs.extend(text_loader.load())
docs.extend(csv_loader.load())

print("Total loaded docs:", len(docs))



# 3. Chunking

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)

splitted_docs = text_splitter.split_documents(docs)

print("Total chunks:", len(splitted_docs))



# 4. Metadata Cleaning
cleaned_docs = []

for doc in splitted_docs:
    doc.metadata = {"source": doc.metadata.get("source", "")}
    cleaned_docs.append(doc)



# 5. Embedding Model

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# 6. Create Vector Store

vector_store = Chroma.from_documents(
    documents=cleaned_docs,
    embedding=embedding_model,
    persist_directory="./chroma_db",
    collection_name="fresh_session"
)

vector_store.persist()

print("Vector DB created successfully ✅")
