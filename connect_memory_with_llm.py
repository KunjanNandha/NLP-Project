import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# -------------------------------
# 1. SETUP LLM
# -------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": "512"
        }
    )
    return llm


# -------------------------------
# 2. CUSTOM PROMPT
# -------------------------------
CUSTOM_PROMPT_TEMPLATE = """
Use ONLY the information provided in the context to answer the user's question.
If the answer is not in the context, say: "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=CUSTOM_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# -------------------------------
# 3. LOAD OR CREATE VECTORSTORE (FAISS)
# -------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(DB_FAISS_PATH, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index_path = os.path.join(DB_FAISS_PATH, "index.faiss")

# If FAISS index exists → load it
if os.path.exists(index_path):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

# If FAISS index does NOT exist → create a new one
else:
    print("No FAISS index found. Creating a new one...")
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load PDFs
    DATA_PATH = "data/"
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = splitter.split_documents(documents)

    # Create FAISS
    db = FAISS.from_documents(text_chunks, embedding_model)

    # Save FAISS index
    db.save_local(DB_FAISS_PATH)
    print("FAISS index created and saved successfully!")

    # If FAISS index exists → load it
if os.path.exists(index_path):
    print("Loading existing FAISS index...")
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )

# If FAISS index does NOT exist → create a new one
else:
    print("No FAISS index found. Creating a new one...")
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    DATA_PATH = "data/"
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("FAISS index created and saved successfully!")


# -------------------------------
# 3B. CREATE RETRIEVER   ← ADD THIS
# -------------------------------
retriever = db.as_retriever(search_kwargs={"k": 3})


# -------------------------------
# 4. BUILD RETRIEVAL CHAIN
# -------------------------------
llm = load_llm(HUGGINGFACE_REPO_ID)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -------------------------------
# 5. RUN QUERY
# -------------------------------
user_query = input("Write Query Here: ")
response = chain.invoke(user_query)

print("\nRESULT:\n", response)