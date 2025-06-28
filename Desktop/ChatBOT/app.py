import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import torch

# Load environment variables
load_dotenv()

# Initialize summarizer with device placement
device = "cuda" if torch.cuda.is_available() else "cpu"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs, handling None returns."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text() or ""  # Handle None case
            text += extracted_text + "\n"
    return text.strip()

def summarize_text(text):
    """Summarize extracted PDF text while handling model constraints."""
    if len(text) > 500:
        summary = summarizer(text[:1024], max_length=200, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    return text 

def extract_topic(text):
    """Extract topic from the document using the first two sentences."""
    sentences = text.split(".")
    return ". ".join(sentences[:2]).strip() + "." if len(sentences) > 2 else text

def get_text_chunks(text):
    """Split text into smaller chunks for embeddings."""
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    """Generate and store vector embeddings using FAISS."""
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore, model_name):
    """Create a conversational retrieval chain using the selected LLM."""
    llm = ChatOllama(model=model_name)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

def handle_userinput(user_question):
    """Process user input and display chat history."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        role = "**You:**" if i % 2 == 0 else "**Chatbot:**"
        st.write(f"{role} {message.content}")

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")

    # Initialize session state
    for key in ["conversation", "chat_history", "pdf_topic", "pdf_summary"]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.title("📚 Chat with Your PDFs")
    st.write("Upload PDF documents, get summaries, and chat with your content.")

    # User input for questions
    user_question = st.text_input("🔍 Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("⚙️ Settings")
        model_name = st.selectbox("Choose an AI Model:", ["gemma:2b", "llama2", "mistral"])

        st.subheader("📂 Upload Documents")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)

        if st.button("🚀 Process"):
            if pdf_docs:
                with st.spinner("Processing your documents... ⏳"):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)

                    # Create conversation chain with selected model
                    st.session_state.conversation = get_conversation_chain(vectorstore, model_name)
                    st.session_state.pdf_topic = extract_topic(raw_text)
                    st.session_state.pdf_summary = summarize_text(raw_text)

                st.success("Processing complete! 🎉 Start chatting now.")

    # Display extracted topic and summary
    if st.session_state.pdf_topic:
        st.subheader("📌 Document Topic")
        st.write(st.session_state.pdf_topic)

    if st.session_state.pdf_summary:
        st.subheader("📝 Summary")
        st.write(st.session_state.pdf_summary)

if __name__ == '__main__':
    main()
