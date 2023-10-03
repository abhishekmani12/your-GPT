from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp

import os
import time
import chromadb


vectorstore_folder_path = "vectorstore"
embeddings_model = "all-MiniLM-L6-v2"

model_type = "GPT4ALL"
model_path = "MODELS/ggml-gpt4all-j-v1.3-groovy.bin"
model_n_ctx = 1000
model_n_batch = 8
target_source_chunks = 4

CHROMA_SETTINGS = Settings(
        persist_directory=vectorstore_folder_path,
        anonymized_telemetry=False
)

def get_RQA():

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    callbacks =[StreamingStdOutCallbackHandler()]
    
            
    if model_type == "GPT4All":
        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            
    else:
        raise Exception(f"Model type {model_type} is invalid")

    RQA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    return RQA

RQA=get_RQA()

def get_answer(query):
    
    if query == "quit":
        break
            
    if query.strip() == "":
        continue

    start = time.time()
    
    res = RQA(query)
    answer, docs = res['result'], res['source_documents']
    
    end = time.time()
        
    time_taken=round(end-start, 2)
        
    document_content={}
    
    for document in docs:
        document_content[document.metadata["source"]] = document.page_content
        
    
    return query, answer, document_content, time_taken
       
 