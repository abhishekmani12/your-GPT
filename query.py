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
from palmquery import ask_palm

import chromadb
from chromadb.config import Settings

vectorstore_folder_path = "vectorstore"
#embeddings_model = "all-MiniLM-L6-v2"
embeddings_model = "hkunlp/instructor-large"


target_source_chunks = 4

preprompt="""You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not provide false facts. Provide a detailed answer to the question."""

m_template = ("<s>[INST]" + preprompt + """ 

                                            Context: {context} 
                                            User: {question}""" + "[/INST]" )

mistral_prompt = PromptTemplate(template=m_template, input_variables=["question","context"])



f_template =( "[INST]" + "<<SYS>>\n" + preprompt + "\n<</SYS>>\n\n" + """
                                                                    
                                                                    Context: {context}
                                                                    User: {question}""" + "[/INST]")

falcon_prompt = PromptTemplate(template=f_template, input_variables=["context", "question"])


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


def get_pipe(model_id, model_basename, model_type):

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

    if model_type == "mistral":
        prompt=mistral_prompt
    elif model_type == "falcon:
        prompt=falcon_prompt
    else:
        raise Exception(f"Model Type - {model_type} Invalid")
    
    RQA = RetrievalQA.from_chain_type( 
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True, 
        callbacks=callback_manager, 
        chain_type_kwargs={"prompt": prompt}
        )
    
    return RQA


def get_answer(query, RQA=None, model_type=None, internet=False):
       
    if query.strip() == "":
        return None
        
    if internet:

        docs = db.similarity_search(query)
        context=""""""

        start = time.time()
        
        for document in docs:
            context += document.page_content
        
        answer=ask_palm(context, query)

        end=time.time()

    else:
        
        document_content={}   
        start = time.time()
            
        res = RQA(query)
        answer, docs = res['result'], res['source_documents']
            
        end = time.time()
        
    for document in docs:
        document_content[document.metadata["source"]] = document.page_content
        
    time_taken=round(end-start, 2)
    
    return query, answer, document_content, time_taken  