import streamlit as st
import os
from rag_engine import save_uploaded_file, process_document, get_answer

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")

st.title("📚 Chat with your Documents (RAG)")
st.markdown("Upload a PDF and ask questions about its content. Powered by **GPT-4o** and **ChromaDB**.")

# Sidebar for configuration and upload
with st.sidebar:
    st.header("1. Knowledge Base")
    
    existing_docs = []
    if os.path.exists("./docs"):
        existing_docs = [f for f in os.listdir("./docs") if f.endswith('.pdf')]
        
    selected_doc = None
    if existing_docs:
        selected_doc = st.selectbox("Select a document to chat with:", ["-- Select a document --"] + existing_docs)
    else:
        st.info("No documents processed yet. Upload one below!")

    st.markdown("---")
    st.header("2. Add New Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if st.button("Process Document"):
        if not uploaded_file:
            st.error("Please upload a PDF document.")
        else:
            # Check if file already exists in our docs directory
            file_path = os.path.join("./docs", uploaded_file.name)
            if os.path.exists(file_path):
                st.info("✅ This document has already been processed! You can select it from the dropdown above.")
            else:
                with st.spinner("Saving document to local disk..."):
                    file_path = save_uploaded_file(uploaded_file)
                
                with st.spinner(f"Processing and embedding {uploaded_file.name}..."):
                    success, message = process_document(file_path)
                    
                if success:
                    st.success("Document embedded! You can now select it from the dropdown to chat.")
                    st.rerun() # Refresh to update dropdown
                else:
                    st.error(f"Failed to process document: {message}")

# Main chat interface
st.header("Chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the selected document..."):
    if not selected_doc or selected_doc == "-- Select a document --":
        st.error("Please select a document from the sidebar to chat with!")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner(f"Thinking (searching {selected_doc})..."):
                answer = get_answer(prompt, selected_doc)
            st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
