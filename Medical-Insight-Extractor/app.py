import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import tempfile
import google.generativeai as genai

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GOOGLE_API_KEY not found.")

# Constants
FAISS_INDEX_PATH = "faiss_index"

# ---------- Functions ----------
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def create_faiss_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

def get_qa_chain():
    prompt_template = """
You are a fessional medical assistant AI. Use only the context vided.

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def handle_user_query(user_question, selected_lang):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question)

        chain = get_qa_chain()
        query = f"Please answer in {selected_lang}:\n\n{user_question}"
        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        answer = response.get("output_text", "No response generated.")

        st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp_file:
            tmp_file.write(f"Q ({selected_lang}): {user_question}\n\nAnswer:\n{answer}")
            path = tmp_file.name

        with open(path, "rb") as file:
            st.download_button("Download Answer", file, f"summary_{selected_lang}.txt", mime="text/plain")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---------- App UI ----------
st.set_page_config(page_title="Smart Health Assistant", layout="centered")

# Inject pastel CSS
st.markdown("""
<style>
body {
    background-color: #f6f8fc;
}
h1, h2, h3 {
    color: #4a4a6a;
}
.sidebar .sidebar-content {
    background-color: #eef1f9;
}
.upload-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.response-box {
    background-color: #fff8f3;
    padding: 20px;
    margin-top: 20px;
    border-radius: 10px;
    font-size: 16px;
    color: #333;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
button {
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- SLIDE 1: Welcome ----------
if "stage" not in st.session_state:
    st.session_state.stage = "welcome"

if st.session_state.stage == "welcome":
    st.title("Smart Health Report Assistant")
    st.markdown("Simplify your medical documents with AI, in your language.")
    if st.button("🚀 Start Now"):
        st.session_state.stage = "upload"

# ---------- SLIDE 2: Upload & Language ----------
elif st.session_state.stage == "upload":
    st.header("Upload Your Reports")
    st.markdown("Upload your medical reports in PDF format for analysis.")

    uploaded_pdfs = st.file_uploader("Select PDF(s)", type="pdf", accept_multiple_files=True)

    st.markdown("### Choose Language")
    language = st.selectbox("", ["English", "Tamil", "Hindi", "Telugu", "Malayalam", "Japanese (日本語)"])
    st.session_state.language = language

    if st.button("Process My Reports"):
        if not uploaded_pdfs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Reading and indexing your reports..."):
                raw_text = extract_text_from_pdfs(uploaded_pdfs)
                chunks = split_text_into_chunks(raw_text)
                create_faiss_vector_store(chunks)
                st.success("Done! You can now ask questions.")
                st.session_state.stage = "ask"

# ---------- SLIDE 3: Ask Questions ----------
elif st.session_state.stage == "ask":
    st.header(" Ask Your Question")
    st.markdown("Ask something specific based on your uploaded reports.")

    user_question = st.text_input("e.g., What is the diagnosis?")

    if user_question:
        with st.spinner("Generating answer..."):
            handle_user_query(user_question, st.session_state.language)

    if st.button("Back to Upload"):
        st.session_state.stage = "upload"
