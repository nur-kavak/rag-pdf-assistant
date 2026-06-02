# RAG Document Assistant

A "Chat with your Data" web application built with Streamlit, LangChain, and ChromaDB. 
This application uses Retrieval-Augmented Generation (RAG) to let you upload PDF documents and ask questions about their content. It uses OpenAI's embeddings and GPT-4o models to provide highly accurate, context-aware answers.

## Features
- **PDF Upload:** Automatically extracts text and chunks it for processing.
- **Local Vector Storage:** Uses ChromaDB to save embeddings locally (`./chroma_db`), preventing the need to re-upload and re-process the same document.
- **Knowledge Base Filter:** Select from previously processed documents to isolate your chat context.
- **Fast UI:** Built with Streamlit for a clean, interactive chat experience.

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nur-kavak/rag-pdf-assistant.git
   cd rag-document-assistant
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your default web browser.
