import os
import subprocess
from tqdm import tqdm
from doc2docx import convert
import img2pdf
import ocrmypdf
import pandas as pd
import sweetviz

import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import CSVLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader

from config import emb_chunk_size, emb_chunk_overlap

vectorstore_folder_path = "vectorstore"
document_ingest_path = "documents"
#embeddings_model = "all-MiniLM-L6-v2"
embeddings_model = "hkunlp/instructor-large"

CHROMA_SETTINGS = Settings(
        persist_directory=vectorstore_folder_path,
        anonymized_telemetry=False
)

# Maps extensions to doc loaders
ext2loader = {
    ".csv": (CSVLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

CHUNK_SIZE = emb_chunk_size
CHUNK_OVERLAP = emb_chunk_overlap


def analyze(path):
    
    try:
        df=pd.read_csv(path)
        my_report = sweetviz.analyze([df, "Train"])
        my_report.show_html('Report.html')

        return my_report
    
    except Exception as e:
        print("Error: ", e)

def image2pdf(file_path):
    
    fname=file_path.rsplit("/")[-1].split(".")[0]
    pdf_path=f"documents/{fname}.pdf"
    
    with open(pdf_path,"wb") as f:
        f.write(img2pdf.convert(file_path))
    
    return pdf_path


def ocr(file_path):

    fname=file_path.rsplit("/")[-1]
    ls=fname.rsplit(".")
    
    if ls[1] in ['jpg', 'jpeg', 'png']:
        file_path=image2pdf(file_path)
        
    pdf_path=f"documents/OCR-{ls[0]}.pdf"
    print(pdf_path)
    ocrmypdf.ocr(file_path, pdf_path, deskew=True)
    
    return pdf_path


def load_document(file_path, existing_files):
    
    extension = "." + file_path.rsplit(".")[1]
    
    if extension in ext2loader:
        if file_path not in existing_files:
            
            loader_type, loader_args = ext2loader[extension]
            loader = loader_type(file_path, **loader_args)
            load=loader.load()
            return load
        else:
            return None

    raise ValueError(f" '{extension}' file type not supported")
    
        
def split_document(file_path, existing_files=[]):
   
    document = load_document(file_path, existing_files)
    if not document:
        print("Vector Embeddings for this file already exists")
        return None
        
    print("Document Loaded")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_text = text_splitter.split_documents(document)
    
    print("Document Split")
    
    return split_text

def check4vectorstore(directory, embeddings):

    db = Chroma(persist_directory=directory, embedding_function=embeddings)
    
    if not db.get()['documents']:
        return False
    else:
        return True

#MAIN FUNCTION CALL
def embed(file_path, scanned=False):

    if scanned:
        file_path=ocr(file_path)
    
    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model)
    
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=vectorstore_folder_path)

    if check4vectorstore(vectorstore_folder_path, embeddings):
        
        print("Exisiting vectorDB found")
        
        db = Chroma(persist_directory=vectorstore_folder_path, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        
        existing_docs=[md['source'] for md in collection['metadatas']]
        
        text = split_document(file_path, existing_docs)
        
        if text:
            db.add_documents(text)
            
    else:
        
        text = split_document(file_path)
        
        print("Creating new vectorDB")
        db = Chroma.from_documents(text, embeddings, persist_directory=vectorstore_folder_path, client_settings=CHROMA_SETTINGS, client=chroma_client)
        
        
    db.persist()
    db = None
    
    print("Document Embedded!")
    return True
