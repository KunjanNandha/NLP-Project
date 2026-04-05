import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Ensure vectorstore folder exists
os.makedirs(DB_FAISS_PATH, exist_ok=True)


# Step 1: Load raw PDF(s)
def load_pdf_files(data):
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


documents = load_pdf_files(DATA_PATH)


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(documents)


# Step 3: Create Embedding Model
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


embedding_model = get_embedding_model()


# Step 4: Load existing FAISS index OR create a new one
index_file = os.path.join(DB_FAISS_PATH, "index.faiss")

if os.path.exists(index_file):
    # Load existing vectorstore
    print("Loading existing FAISS index...")
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    # Create new vectorstore
    print("No FAISS index found. Creating new vectorstore...")
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS index created and saved.")
