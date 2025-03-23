import os
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Load and prepare all documents from the knowledge base folder
def load_documents(folder_path="knowledge_base"):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith((".txt", ".md", ".csv")):
            loader = TextLoader(os.path.join(folder_path, file))
            docs = loader.load()
            all_docs.extend(docs)
    return all_docs

# Split documents into smaller chunks for vector embedding
def split_into_chunks(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# Create FAISS index and save it locally
def create_faiss_index(chunks, save_path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'device': 'cpu', 'normalize_embeddings': True}
    )
    db = FAISS.from_documents(chunks, embedding=embeddings)
    db.save_local(save_path)
    return db

# Load FAISS index from local directory
def load_faiss_index(path="vectorstore"):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={'device': 'cpu', 'normalize_embeddings': True}
    )
    return FAISS.load_local(folder_path=path, embeddings=embeddings)

# Retrieve context from FAISS based on a user query
def retrieve_context(db, query, k=3):
    docs = db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])
