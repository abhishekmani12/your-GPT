import spacy
import pandas
import requests
import re
import wikipedia
from bs4 import BeautifulSoup
from tqdm import tqdm

#!python -m spacy download en_core_web_sm #Execute on first run

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
    keywords_obj=doc.ents
    
    string=str(keywords_obj)
    keywords=string.replace("(","").replace(")","").split(",")
    
    keywords=list(filter(None, keywords))
    
    return set(keywords)

def wiki(keyword, limit):
    
    parameters = {'q': keyword, 'limit': number_of_results}
    response=requests.get(url, headers=headers, params=parameters)
    
    if len(response.json()['pages']) == 0:
        print(f"No content found for keyword - {keyword}")
        return ["No content found"]
    
    desc=response.json()['pages'][0]['description']
    key=response.json()['pages'][0]['key']
    
    index=0
    if desc == 'Topics referred to by the same term':
        index=1
        

    closest_keyword=wikipedia.search(key)[index]
    closest_keyword=closest_keyword.replace(" ","_")

    
    if index == 1:
        print(f"{keyword} not found. Redirecting to {closest_keyword}") #for manual redirection
    elif key != keyword:
        print(f"{keyword} not found. Redirecting to {key}") #for auto redirection by wikipedia
    else:
        print(f"{keyword} found.")


    raw_result=requests.get(f'https://en.wikipedia.org/wiki/{closest_keyword}')

    html_content=BeautifulSoup(raw_result.text, "html.parser")

    content=""""""
    curr=0
    
    p1 = re.compile('\[[0-9]*[a-z]*\]')
    p2 = re.compile('\\n[0-9]*')
    
    for para in html_content.select('p'):
        
        text=para.getText()
        
        clean_text=p1.sub('', text)
        clean_text=p2.sub('', clean_text)
        
        clean_text=" ".join(clean_text.split())
        content += clean_text
        curr+=1

        if curr == limit:
            break
            
    return content
    

def get_details(text, medical=False, limit=3):

    keywords=get_keywords(text, medical)
    
    if len(keywords) == 0:
        print("No keywords found")
        return None
    
    details_dict={}
    
    for key in tqdm(keywords):
        
        key=key.strip()
        print(key)
        details_dict[key]=wiki(key, limit)
    
    return details_dict, keywords 