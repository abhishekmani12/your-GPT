import google.generativeai as palm
import re
import os
from dotenv import load_dotenv
from config import PALM_MODEL, PALM_TEMPERATURE, PALM_CANDIDATE_COUNT, PALM_TOP_K, PALM_TOP_P

load_dotenv()

API_KEY=os.getenv('PALM_API_KEY')

def ask_palm(context, query=None, summary=False):
    
    palm.configure(api_key=API_KEY)
    messages=[]
    defaults = {
        
      'model': PALM_MODEL,
      'temperature': PALM_TEMPERATURE,
      'candidate_count': PALM_CANDIDATE_COUNT,
      'top_k': PALM_TOP_K,
      'top_p': PALM_TOP_P,
    }


    summary_preprompt = """You are a helpful assistant, You will use the provided text content to summarize it as short as possible.
    Do not make up facts nor do not make any mistakes."""
    
    preprompt = """You are a helpful assistant, you will only use the provided CONTEXT to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on 
the provided context, inform the user. Do not provide false facts. Provide short answers limited to 5 lines for the question."""
    
    
    p=re.compile("\\n")
    
    if context:
        cleaned_context=p.sub("",context)

    if summary:
        messages = [ f"""TEXT CONTENT: {cleaned_context}
                    Summarize the above""" ]
        preprompt = summary_preprompt
    else:
        messages = [ f"""CONTEXT: {cleaned_context}
                    
                     Question: Based on the above context {query} """ ]
    
    #messages.append("NEXT REQUEST")
    
    response = palm.chat(
      **defaults,
      context=preprompt,
      messages=messages
    )
    return response.last