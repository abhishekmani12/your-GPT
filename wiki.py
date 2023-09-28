import spacy
import pandas
import requests
import re
import wikipedia
from tqdm import tqdm

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

    return set(keywords)

def wiki(keyword, limit):
    
    parameters = {'q': keyword, 'limit': number_of_results}
    response=requests.get(url, headers=headers, params=parameters)

    desc=response.json()['pages'][0]['description']

    index=0
    if desc == 'Topics referred to by the same term':
        index=1

    closest_keyword=wikipedia.search(keyword)[index]
    closest_keyword=closest_keyword.replace(" ","_")


    raw_result=requests.get(f'https://en.wikipedia.org/wiki/{closest_keyword}')

    html_content=BeautifulSoup(raw_result.text, "html.parser")

    content=[]
    curr=0
    for para in html_content.select('p'):
        content.append(para.getText())
        curr+=1

        if curr == limit:
            break
            
    return content
    


def get_details(text, medical=False, limit=3):

    keywords=get_keywords(text, medical)
    
    details_dict={}
    
    for key in tqdm(keywords):
        
        print(key)
        details_dict[key]=wiki(key, limit)
    
    return details_dict, keywords 