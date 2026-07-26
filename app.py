import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ---------------------- Load API Keys ----------------------

load_dotenv()

groq_api_key = os.getenv("API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not groq_api_key:
    st.error("Groq API Key not found in .env")
    st.stop()

if not google_api_key:
    st.error("Google API Key not found in .env")
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

# ---------------------- Streamlit ----------------------

st.set_page_config(
    page_title="Document Question Answering System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering System")
st.write("Upload PDF files and ask questions based on their contents.")

# ---------------------- LLM ----------------------

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# ---------------------- Prompt ----------------------

prompt = ChatPromptTemplate.from_template(
"""
You are an intelligent assistant.

Answer ONLY from the given context.

<context>
{context}
</context>

Question:
{input}

If the answer is not present in the context, simply say:

"I couldn't find the answer in the uploaded document."
"""
)

# ---------------------- Upload PDFs ----------------------

uploaded_files = st.file_uploader(
    "📄 Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# ---------------------- Create Vector Store ----------------------
def create_vector_store(files):

    all_docs = []

    for file in files:

        st.write("File Name:", file.name)
        st.write("File Size:", file.size)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:

            tmp.write(file.getvalue())

            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)

        docs = loader.load()

        st.write("Pages Loaded:", len(docs))

        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(all_docs)

    st.write("Chunks Created:", len(split_docs))

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )

    st.session_state.vector = FAISS.from_documents(
        split_docs,
        embeddings
    )
# ---------------------- Process PDFs ----------------------

if st.button("🚀 Process PDFs"):

    if uploaded_files:

        try:

            with st.spinner("Creating Vector Database..."):

                create_vector_store(uploaded_files)

            st.success("Vector Database Created Successfully!")

        except Exception as e:

            st.error(e)

    else:

        st.warning("Please upload PDF files first.")

# ---------------------- Question ----------------------

question = st.text_input("💬 Ask your Question")

# ---------------------- Answer ----------------------

if question:

    if "vector" not in st.session_state:

        st.warning("Please upload and process PDFs first.")

    else:

        try:

            document_chain = create_stuff_documents_chain(
                llm,
                prompt
            )

            retriever = st.session_state.vector.as_retriever()

            retrieval_chain = create_retrieval_chain(
                retriever,
                document_chain
            )

            start = time.time()

            response = retrieval_chain.invoke(
                {"input": question}
            )

            end = time.time()

            st.markdown("## ✅ Answer")

            st.write(response["answer"])

            st.info(f"Response Time : {end-start:.2f} seconds")

        except Exception as e:

            st.error(e)