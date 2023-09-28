import spacy
import pandas
import requests

number_of_results = 1
headers = {
    'User-Agent': 'Local-Mini-GPT'
}

base_url = 'https://api.wikimedia.org/core/v1/wikipedia/'
endpoint = '/search/page'
url = base_url + 'en' + endpoint

def get_keywords(text, medical):
    
    if medical:
        model="en_core_sci_lg"
    else:
        model="en_core_web_sm"
    
    nlp = spacy.load(model)
    doc = nlp(text)
    keywords=list(doc.ents)
    
    return keywords

def wiki(keywords):
    
    export={}
    
    for key in keywords:
        
        parameters = {'q': key, 'limit': number_of_results}
        response = requests.get(url, headers=headers, params=parameters)
        
        export[key] = response

def get_details(text, medical=False):

    keywords=get_keywords(text, medical)
    
    details_dict=wiki(keywords, lines)
    
    return details_dict