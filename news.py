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
    i=0
    for enc_link in encoded_links:
        i+=1
        enc_str = re.search(r'/([^/]+)\?', enc_link).group(1)

        padding = 4 - (len(enc_str) % 4)
        pad_str = enc_str + ('a'*padding)

        try:
            raw_str = base64.b64decode(pad_str)
            links.append(re.findall(r"https://[^\\]+", str(raw_str))[0])
            print(f"-- News Articles Collected: {i}", end="\r", flush=True)
        except:
            continue

    return links

def get_news(news_title, count = -1):

    links=get_news_links(news_title)

    dates=[]
    urls=[]
    text=[]
    summaries=[]

    i=0
    print("\n")
    for link in links:
        i+=1
        try:

            article = Article(link)
            article.download()
            article.parse()

            dates.append(article.publish_date)
            text.append(article.text)

            article.nlp()
            #summaries.append(article.summary)
            urls.append(link)
            print(f"-- Processing: {i}", end="\r", flush=True)

            if i == count:
                break
        except:
            continue

    print("\n\n-- Completed")
    return pd.DataFrame({'Date': dates, 'Text': text})