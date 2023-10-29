import google.generativeai as palm
import re
import os
from dotenv import load_dotenv
from config import PALM_MODEL, PALM_TEMPERATURE, PALM_CANDIDATE_COUNT, PALM_TOP_K, PALM_TOP_P

load_dotenv()

API_KEY=os.getenv('PALM_API_KEY')

def ask_palm(context, query):
    
    palm.configure(api_key=API_KEY)
    
    defaults = {
        
      'model': PALM_MODEL,
      'temperature': PALM_TEMPERATURE,
      'candidate_count': PALM_CANDIDATE_COUNT,
      'top_k': PALM_TOP_K,
      'top_p': PALM_TOP_P,
    }
    preprompt = """You are a helpful assistant, you will only use the provided CONTEXT to answer user questions.
    Read the given context before answering questions and think step by step. If you can not answer a user question based on 
    the provided context, inform the user. Do not provide false facts. Provide short answers limited to 5 lines for the question."""
    
    
    p=re.compile("\\n")
    cleaned_context=p.sub("",context)
    
    messages = [ f"""CONTEXT: {cleaned_context}
                    
                     Question: Based on the above context {query} """ ]
    
    #messages.append("NEXT REQUEST")
    
    response = palm.chat(
      **defaults,
      context=preprompt,
      messages=messages
    )
    return response.last