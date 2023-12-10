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

from config import SOURCE_CHUNKS, TOP_K, TOP_P, TEMPERATURE, REPEAT_PENALTY, CTX, MAX_TOKENS, BATCH_SIZE, GPT4ALL_BATCH_SIZE


vectorstore_folder_path = "vectorstore"
#embeddings_model = "all-MiniLM-L6-v2"
embeddings_model = "hkunlp/instructor-large"

MISTRAL = [
    "TheBloke/Mistral-7B-Instruct-v0.1-GGUF", 
    "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
]

LLAMA = [
    "TheBloke/Llama-2-7b-Chat-GGUF", 
    "llama-2-7b-chat.Q4_K_M.gguf"
]

gpt4all_model = "MODELS/ggml-model-gpt4all-falcon-q4_0.bin" #convert to gguf - pending

target_source_chunks = SOURCE_CHUNKS

CHROMA_SETTINGS = Settings(
        persist_directory=vectorstore_folder_path,
        anonymized_telemetry=False
)
###############################################################################################################################################

preprompt="""You are a helpful assistant, you will use the provided context to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not provide false facts. Provide a detailed answer to the question."""

###############################################################################################################################################

m_template = ("<s>[INST]" + preprompt + """ 
Context: {context} 
User: {question}""" + "[/INST]" )

mistral_prompt = PromptTemplate(template=m_template, input_variables=["question","context"])

###############################################################################################################################################

l_template =( "[INST]" + "<<SYS>>\n" + preprompt + "\n<</SYS>>\n\n" + """
Context: {context}
User: {question}""" + "[/INST]")

llama_prompt = PromptTemplate(template=l_template, input_variables=["context", "question"])

###############################################################################################################################################
g_template =("""
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step.""")

gpt4all_prompt = PromptTemplate(template=g_template, input_variables=["context", "question"])

###############################################################################################################################################


def load_model(model_id, model_basename):

    model_path = hf_hub_download(
        
        repo_id=model_id, 
        filename=model_basename, 
        resume_download=True, 
        cache_dir="MODELS/", )
        
    kwargs = { 
        
        "model_path": model_path, 
        "top_p":TOP_P,
        "top_k":TOP_K,
        "temperature":TEMPERATURE,
        "repeat_penalty":REPEAT_PENALTY,
        "n_ctx":CTX, 
        "max_tokens": MAX_TOKENS, 
        "n_batch": BATCH_SIZE }
        
    return LlamaCpp(**kwargs)

def get_pipe(model_type):

    model_id=''
    model_basename=''

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
    

    if model_type == "mistral":
        prompt=mistral_prompt
        model_id=MISTRAL[0]
        model_basename=MISTRAL[1]
        
    elif model_type == "llama":
        prompt=llama_prompt
        model_id=LLAMA[0]
        model_basename=LLAMA[1]

    elif model_type == "gpt4all":

        gpt4all_llm = LlamaCpp(
                        model_path=gpt4all_model, 
                        top_p=TOP_P,
                        top_k=TOP_K,
                        temperature=TEMPERATURE,
                        repeat_penalty=REPEAT_PENALTY,
                        n_ctx=CTX, 
                        max_tokens=MAX_TOKENS, 
                        n_batch=GPT4ALL_BATCH_SIZE 
                        )
        
        return gpt4all_llm, db
        
    elif model_type == "bard":
        return None,db
        
    else:
        raise Exception(f"Model Type - {model_type} Invalid")

    llm=load_model(model_id, model_basename)
    
    RQA = RetrievalQA.from_chain_type( 
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True, 
        callbacks=callback_manager, 
        chain_type_kwargs={"prompt": prompt}
        )
    
    return RQA


def get_answer(query=None, text=None, RQA=None, gpt4all=False, palm=False):
    document_content={} 
    answer=None
    
    if query.strip() == "":
        return None
    if palm and gpt4all:
        return None
    
    if palm or gpt4all:
        if query and not text:
            
            db=RQA[1]
            docs = db.similarity_search(query, k=target_source_chunks)
            context=""""""
    
            start = time.time()
            
            for document in docs:
                context += document.page_content

        if palm:
            if text:
                answer = ask_palm(context=text, summary=True)
            else:
                answer=ask_palm(context, query)
        else:            
            llm_chain = LLMChain(prompt=gpt4all_prompt, llm=RQA[0])
            answer=llm_chain.run(query)
            
        end=time.time()

    else:
         
        start = time.time()
            
        res = RQA(query)
        answer, docs = res['result'], res['source_documents']
            
        end = time.time()

    if text and not query:
        return answer time_taken
        
    for document in docs:
        document_content[document.metadata["source"]] = document.page_content
        
    time_taken=round(end-start, 2)
    
    return query, answer, document_content, time_taken  