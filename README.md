# Medical Chatbot (NLP-Project)

A Streamlit-based Medical Chatbot that uses an LLM connected with a memory/vector store to provide information and assistance based on provided medical data.

## Features
- Generates responses leveraging a Large Language Model (LLM).
- Uses a vector store for memory and context.
- Streamlit web interface (`MediBot.py`).

## Setup and Installation

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up the Groq API Key:**
    Set the `GROQ_API_KEY` environment variable. Alternatively, you can add it to a `.env` file in the project root.
    ```powershell
    $env:GROQ_API_KEY="your_api_key_here"
    ```

4. **Initialize Memory/Vector Store:**
    Before running the app, create the memory/vector store.
    ```bash
    python create_memory_with_llm.py
    ```

5. **Run the Streamlit application:**
    ```bash
    streamlit run MediBot.py
    ```
