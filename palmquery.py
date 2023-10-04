import google.generativeai as palm
import re

API_KEY="AIzaSyBNgfp1r9TLO5StflsjKQ8ZHZQakJ4X2xQ"

def ask_palm(context, query):
    
    palm.configure(api_key="AIzaSyBNgfp1r9TLO5StflsjKQ8ZHZQakJ4X2xQ")
    
    defaults = {
        
      'model': 'models/chat-bison-001',
      'temperature': 0.2,
      'candidate_count': 1,
      'top_k': 40,
      'top_p': 0.95,
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