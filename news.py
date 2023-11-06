from GoogleNews import GoogleNews
from newspaper import Article
import pandas as pd
import base64
import re


googlenews = GoogleNews()
googlenews.enableException(True)
googlenews = GoogleNews(lang='en')
googlenews = GoogleNews(period='1d')
googlenews = GoogleNews(encode='utf-8')


def get_news_links(news_title): 

    googlenews.get_news(news_title)
    encoded_links=googlenews.get_links()
    googlenews.clear()

    links=[]
    for enc_link in encoded_links:

        enc_str = re.search(r'/([^/]+)\?', enc_link).group(1)

        padding = 4 - (len(enc_str) % 4)
        pad_str = enc_str + ('a'*padding)

        try:
            raw_str = base64.b64decode(pad_str)
            links.append(re.findall(url_pattern, str(raw_str))[0])
        except:
            continue
        
    return links

def get_content(links):

    dates=[]
    urls=[]
    text=[]
    summaries=[]
    
    for link in links:

        try:
            article = Article(link)
            article.download()
            article.parse()

            dates.append(article.publish_date)
            text.append(article.text)

            article.nlp()
            summaries.append(article.summary)

            urls.append(link)
        except:
            continue

    return pd.DataFrame({'Date': dates, 'Url': urls, 'Text': text, 'Summary': summaries})