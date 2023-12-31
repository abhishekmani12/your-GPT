import google.generativeai as gai
import re
import os
from dotenv import load_dotenv
from config import GEM_MODEL, GEM_TEMPERATURE, GEM_MAX_TOKENS, GEM_TOP_K, GEM_TOP_P

load_dotenv()
API_KEY=os.getenv('GEM_API_KEY')

summary_preprompt = """You are a helpful assistant, Summarize the below text content into 10 short points.
These 10 points should cover most of the content but should be short and readable"""

preprompt = """You are a helpful assistant that tries to answer the QUERY based on the provided CONTEXT.
You can make inferences from the context and parse it and extract data inherently and can use your own knowledge base"""

def ask_gemini(context, query=None, summary=False):

    if summary:
        prompt = summary_preprompt
        TEMPERATURE = 0.7
    else:
        prompt = preprompt
        TEMPERATURE = GEM_TEMPERATURE

    gai.configure(api_key=API_KEY)
    messages=[]
    defaults = {

      'temperature': TEMPERATURE,
      'top_k': GEM_TOP_K,
      'top_p': GEM_TOP_P,
      'max_output_tokens': GEM_MAX_TOKENS,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
    ]

    model = gai.GenerativeModel(model_name=GEM_MODEL,
                              generation_config=defaults,
                              safety_settings=safety_settings)


    p=re.compile("\\n")

    if context:
        cleaned_context=p.sub("",context)

    if summary:
        messages = [ f"""{prompt}

                    TEXT CONTENT: {cleaned_context} 
                    
                    QUERY: Summarize the above text content into 10 short, readable points"""]
    else:
        messages = [ f"""{prompt}

                    CONTEXT: {cleaned_context}

                    QUERY: Based on the above context {query} """ ]

    #messages.append("NEXT REQUEST")

    response = model.generate_content(messages)

    if response is None:

        return "Please Retry"
    else:
        return response.text
