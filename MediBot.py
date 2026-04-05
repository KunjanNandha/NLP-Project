import os
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ----------------------------------------------------
# LOAD ENV VARIABLES
# ----------------------------------------------------
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    st.error("❌ GROQ_API_KEY missing in .env file.")
    st.stop()


# ----------------------------------------------------
# VECTORSTORE LOADER
# ----------------------------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ----------------------------------------------------
# PROMPT
# ----------------------------------------------------
CUSTOM_PROMPT_TEMPLATE = """
You are an AI healthcare assistant specializing in medical knowledge from the Gale Encyclopedia of Medicine.
Use the context below as your primary source. If the context is relevant, use it.
If the context is insufficient, use your general medical knowledge to answer helpfully.
Always add appropriate health disclaimers.

- Never diagnose or prescribe medication.
- For symptoms, always say: "Please consult a qualified healthcare professional for proper diagnosis."
- For remedies, always say: "This is not a substitute for professional medical advice."
- Maintain a friendly, supportive, and professional tone.

Important guidelines:
- If the user greets you (e.g., "Hello", "Hi"), respond warmly: "Hello! How can I assist you today? 😊"
- Never diagnose a condition or prescribe medication.
- For symptom-related questions, always add: "Please consult a qualified healthcare professional for a proper diagnosis."
- For herbal remedy questions, always add: "This is not a substitute for professional medical advice."
- Maintain a friendly, supportive, and professional tone at all times.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=CUSTOM_PROMPT_TEMPLATE,
)


# ----------------------------------------------------
# BUILD QA CHAIN
# ----------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


qa_chain = build_chain()


# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------
def main():
    st.set_page_config(page_title="MediBot", page_icon="💊", layout="wide")
    st.title("💊 Ask MediBot")
    st.write("**Your AI assistant for medical & herbal wellness ⚕️**")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask MediBot about symptoms, herbs, or remedies...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        try:
            with st.spinner("MediBot is thinking..."):
                response = qa_chain.invoke(user_input)

            st.chat_message("assistant").markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    main()
