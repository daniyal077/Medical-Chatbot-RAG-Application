import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_files(data_path):
    """Loads all PDF documents from a directory."""
    try:
        loader = DirectoryLoader(
            data_path,
            glob='*.pdf',
            loader_cls=PyPDFLoader  
        )
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error loading PDFs: {e}")
        return []

def create_chunks(extracted_data, chunk_size=1000, chunk_overlap=200):
    """Splits documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def get_embedding_model():
    """Loads the HuggingFace sentence transformer model."""
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Define paths
data_dir = 'data/'
vectorDB_path = 'vectorstore/db_faiss'

# Load and process documents
docs = load_files(data_dir)
if not docs:
    raise ValueError("No documents found. Check your directory path.")

text_chunks = create_chunks(docs)

# Load embeddings
embedding_model = get_embedding_model()

# Create FAISS vector store
db = FAISS.from_documents(text_chunks, embedding_model)

# Save vector database
if not os.path.exists(vectorDB_path):
    os.makedirs(vectorDB_path)
db.save_local(vectorDB_path)
print(f"FAISS vector database saved at {vectorDB_path}")
