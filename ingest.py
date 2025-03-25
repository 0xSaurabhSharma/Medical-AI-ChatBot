import os
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Qdrant

embeddings = SentenceTransformerEmbeddings(model_name = "NeuML/pubmedbert-base-embeddings")
print("="*15,embeddings,"="*15)

loaders = DirectoryLoader(
    'data/', 
    glob="**/*.pdf", 
    show_progress = True, 
    loader_cls = PyPDFLoader)
docs = loaders.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 100)
docs_splits = text_splitter.split_documents(docs)

url = "http://localhost:6333/dashboard"
# url = "https://ubiquitous-acorn-gvpxgxq9v4jf99jg-6333.app.github.dev/dashboard"
qdrant = Qdrant.from_documents(
    docs_splits,
    embeddings,
    location=url,  # Correct argument for remote Qdrant instance
    # prefer_grpc = False,
    collection_name="vector_database"
)

print("="*15,"Vectior DB is Created!","="*15)