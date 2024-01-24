# your-GPT
An internet free locally deployable GPT with a suite of text and data analayis/extraction tools

- **LLMs:** Llama-7b, Mistral-7b, GPT4All, Palm, Gemini
- **Embedding Model:** hkunlp/instructor-large
- **Vector DB:** chromadb
- **Finetuning:** PEFT, BNB, QLORA

## Scripts:

- `load.py`: Loads textual data, splits data into chunks, creates vector embeddings of these chunks and stores them in the vector database.
- `qdb.py`: Test loading script which uses [qdrant](https://github.com/qdrant/qdrant-client) as a vector database
- `query.py`: Text generation based on query from user and context provided from the database via similarity search.
- `palmquery.py`: Extension script to call palm-API
- `wiki.py`: Wikipedia Context generation script based on input sentence. Know more from my other [repo](https://github.com/abhishekmani12/Wiki-Content-Retriever)
- `news.py`: News extraction script which extracts news from Google News
- `Your-GPT.py`: Landing page script which integrates scripts present in `pages` directory for a multi page Streamlit app.

## Features:

### Data Vizualization:
Feed a CSV data file to get a visualized analysis report with sweetviz via a html page

https://github.com/abhishekmani12/your-GPT/assets/76105443/9cf34fe4-03c1-4a83-a54e-c9ce1ea7de19

### Content generation from extracted keywords with NER and Wikipedia:
When supplied with a text corpus, content for each keyword extracted is fetched and parsed from the most similar result from Wikipedia. NER - Spacy(en_core_web_sm) & SBERT

https://github.com/abhishekmani12/your-GPT/assets/76105443/15292964-1672-49ae-aac2-0dc217dda1de

### Query with Documents (RAG):
With 5 models (3 local and 2 internet based), embedded documents can be queried upon. Supports PDF, txt and word. Also has OCR support for wordless PDFs and images.

https://github.com/abhishekmani12/your-GPT/assets/76105443/8859403d-5a94-484b-840f-8da0119651a6

### Summarization:
Summarizes any kind of input text into 10 readable points.

https://github.com/abhishekmani12/your-GPT/assets/76105443/f8e3c7e2-7898-4005-8ae6-bc70acdf6056

### News Feed and Summarization:
Gets news feed based on supplied keywords and has the option to summarize them into readable points. Feed supports upto 100 news articles and summarizes the first 5 articles for now.

https://github.com/abhishekmani12/your-GPT/assets/76105443/ee6c473b-5c79-4090-8b15-c76149930f50

## Ongoing:
- Implementation of NeMo guardrails for topical awareness.
- Prototyping with Taipy for a production environment




