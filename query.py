from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.llms import CTransformers
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

import os
import time

import chromadb
from chromadb.config import Settings

vectorstore_folder_path = "vectorstore"
embeddings_model = "all-MiniLM-L6-v2"

model_type = "GPT4ALL"
model_path = "MODELS/ggml-gpt4all-j-v1.3-groovy.bin"
model_n_ctx = 1000
model_n_batch = 8
target_source_chunks = 4

mistral_config = {'max_new_tokens': 100, 'temperature': 0}

template = """<s>[INST] You are a helpful, respectful and honest assistant. If you do not know the answer, do not make up false facts, just say I don't know.Answer exactly in few words from the context
Answer the question below from context below :
{context}
{question} [/INST] </s>
"""

mistral_prompt = PromptTemplate(template=template, input_variables=["question","context"])


CHROMA_SETTINGS = Settings(
        persist_directory=vectorstore_folder_path,
        anonymized_telemetry=False
)

def get_pipe(model_type):

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=vectorstore_folder_path)
    
    db = Chroma(persist_directory=vectorstore_folder_path, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    callbacks =[StreamingStdOutCallbackHandler()]
    
            
    if model_type == "GPT4All":
        llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        
    elif model_type =="mistral":
        llm = CTransformers(model='MODELS/mistral-7b-instruct-v0.1.Q4_K_M.gguf', model_type="mistral", config=mistral_config)
        llm_chain = LLMChain(prompt=mistral_prompt, llm=llm) 
        
        return llm_chain, db
        
    else:
        raise Exception(f"Model type {model_type} is invalid")

    RQA = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    
    return RQA


def get_answer(query, RQA, model_type, db=None):
       
    if query.strip() == "":
        return None
    
    document_content={}
    
    if model_type == "GPT4ALL":
        
        start = time.time()
        
        res = RQA(query)
        answer, docs = res['result'], res['source_documents']
        
        end = time.time()
        time_taken=round(end-start, 2)

       
        for document in docs:
            document_content[document.metadata["source"]] = document.page_content
        
    elif model_type == "mistral":
        
        q=f"""{query}"""
        start = time.time()
        
        documents = db.similarity_search(q)
        
        context=""""""
        
        for doc in document:
            document_content[doc.metadata["source"]] = doc.page_content
            context += doc.page_content
        
        answer = RQA.run({"question":q, "context":context})
        
        end = time.time()
        time_taken=round(end-start, 2)
        
    
    return query, answer, document_content, time_taken
       
 