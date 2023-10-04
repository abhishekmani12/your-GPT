from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from huggingface_hub import hf_hub_download
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.vectorstores import Chroma

from langchain.llms import GPT4All
from langchain.llms import LlamaCpp
from langchain.llms import CTransformers

from langchain import HuggingFacePipeline
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import os
import time

import chromadb
from chromadb.config import Settings

vectorstore_folder_path = "vectorstore"
#embeddings_model = "all-MiniLM-L6-v2"
embeddings_model = "hkunlp/instructor-large"


target_source_chunks = 4

preprompt="""You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

template = ("<s>[INST]" + preprompt + """ 

                                            Context: {context} 
                                            User: {question}""" + "[/INST]" )

mistral_prompt = PromptTemplate(template=template, input_variables=["question","context"])


CHROMA_SETTINGS = Settings(
        persist_directory=vectorstore_folder_path,
        anonymized_telemetry=False
)

def load_model(model_id, model_basename):

    model_path = hf_hub_download(
        
        repo_id=model_id, 
        filename=model_basename, 
        resume_download=True, 
        cache_dir="MODELS/", )
        
    kwargs = { 
        
        "model_path": model_path, 
        "n_ctx": 4096, 
        "max_tokens": 4096, 
        "n_batch": 512 }
        
    return LlamaCpp(**kwargs)


def get_pipe(model_id, model_basename):

    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model)
    
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=vectorstore_folder_path)
    
    db = Chroma(
        
        persist_directory=vectorstore_folder_path, 
        embedding_function=embeddings, 
        client_settings=CHROMA_SETTINGS, 
        client=chroma_client, )
    
    #retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    
    retriever = db.as_retriever()
    
    #memory = ConversationBufferMemory(input_key="question", memory_key="history")
    
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm=load_model(model_id, model_basename)
    
    RQA = RetrievalQA.from_chain_type( 
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True, 
        callbacks=callback_manager, 
        chain_type_kwargs={"prompt": mistral_prompt}
        )
    
    return RQA


def get_answer(query, RQA, model_type):
       
    if query.strip() == "":
        return None
    
    document_content={}
    
    if model_type == "mistral":
        
        start = time.time()
        
        res = RQA(query)
        answer, docs = res['result'], res['source_documents']
        
        end = time.time()
        time_taken=round(end-start, 2)

       
        for document in docs:
            document_content[document.metadata["source"]] = document.page_content
    
    return query, answer, document_content, time_taken    
       
 