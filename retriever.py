import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

documents = [
    Document(page_content="Langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM-based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="Open AI provides powerful embedding models"),
]

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 1}
)

#manually retrived relevant content
Query = "What is Langchain?"

retrieved_docs = retriever.invoke(Query)

# LLM initialize
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=os.getenv("Groq_API_KEY")
)

#manually pass retriver text to LLM
prompt = f"Based on the following text, answer the question:\n\n{retrieved_docs}\n\nQuestion: {Query}"

answer = llm.invoke(prompt)

#print the result
print(answer.content)
#citation