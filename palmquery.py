import google.generativeai as palm
import re
import os
from dotenv import load_dotenv
from config import PALM_MODEL, PALM_TEMPERATURE, PALM_CANDIDATE_COUNT, PALM_TOP_K, PALM_TOP_P

load_dotenv()
API_KEY=os.getenv('PALM_API_KEY')

summary_preprompt = """You are a helpful assistant, Summarize the below text content into 10 points. These 10 points should cover most of the content"""

preprompt = """You are a helpful assistant, you will only use the provided CONTEXT to answer user questions.
Read the given context before answering questions and think step by step. If you can not answer a user question based on the provided context, inform the user. 
Do not provide false facts. Provide short answers limited to 5 lines for the question."""

def ask_palm(context, query=None, summary=False):

    if summary:
        prompt = summary_preprompt
        TEMPERATURE = 0.7
    else:
        prompt = preprompt
        TEMPERATURE = PALM_TEMPERATURE
    
    palm.configure(api_key=API_KEY)
    messages=[]
    defaults = {
        
      'model': PALM_MODEL,
      'temperature': TEMPERATURE,
      'candidate_count': PALM_CANDIDATE_COUNT,
      'top_k': PALM_TOP_K,
      'top_p': PALM_TOP_P,
    }

    p=re.compile("\\n")
    
    if context:
        cleaned_context=p.sub("",context)

    if summary:
        messages = [ f"""TEXT CONTENT: {cleaned_context} """]
    else:
        messages = [ f"""CONTEXT: {cleaned_context}
                    
                     Question: Based on the above context {query} """ ]
    
    #messages.append("NEXT REQUEST")
    
    response = palm.chat(
      **defaults,
      context=prompt,
      messages=messages
    )

    if summary:
        pattern = re.compile(r'here are 10[\s\S]*', re.IGNORECASE) #Pattern is too rigid
        match = pattern.search(response.messages[len(response.messages)-1]['content'])
        
        if match:    
            extract = match.group(0)
            return extract
        else:
            return "Please Retry"
    else:
        return response.last
    