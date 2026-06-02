import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables from .env file
load_dotenv()

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"

def init_dirs():
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file):
    init_dirs()
    file_path = os.path.join(DOCS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_document(file_path):
    try:
        # 1. Load the PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 2. Split into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Embed using state-of-the-art embedding model
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # 4. Store in ChromaDB, persist to disk
        Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DIR)
        
        return True, "Document successfully processed and stored."
    except Exception as e:
        return False, str(e)

def get_answer(query, selected_doc_name=None):
    try:
        # Re-initialize embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Load the persisted Vector Store from disk
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        search_kwargs = {"k": 5}
        if selected_doc_name:
            # PyPDFLoader stores the file path in the 'source' metadata field
            source_path = os.path.join(DOCS_DIR, selected_doc_name)
            search_kwargs["filter"] = {"source": source_path}
            
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        
        # Use state-of-the-art inference model
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        system_prompt = (
            "You are a helpful and intelligent assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer or the context doesn't contain it, say that you don't know. "
            "Explain your reasoning clearly based on the context."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"Error occurred: {str(e)}"
