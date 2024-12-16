# ingest.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import pickle
import os

PDF_PATH = "/Users/chiel/Documents/pb4/UAV 2012.pdf"  # Path to your local PDF

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text
raw_text = get_pdf_text(PDF_PATH)

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
text_chunks = text_splitter.split_text(raw_text)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Save FAISS index to disk
vectorstore.save_local("faiss_index")

# Optionally, store metadata in a pickle file if needed
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(text_chunks, f)
