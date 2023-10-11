# your-GPT
An internet free locally deployable GPT ( If your PC can handle it )

My bible for this project right now: [Langchain Docs](https://python.langchain.com/docs/expression_language/interface)

- Models: Llama-7b, Mistral-7b, GPT4All, palmapi
- Vector DB: chromadb
- Finetuning: PEFT, BNB, QLORA

## Scripts:

- `load.py`: Loads textual data, splits data into chunks, creates vector embeddings of these chunks and stores them in the vector database.
- `query.py`: Text generation based on query from user and context provided from the database via similarity search.
- `palmquery.py`: Extension script to call palm-API
- `wiki.py`: Wikipedia Context generation script based on input sentence. Know more from my other [repo](https://github.com/abhishekmani12/Wiki-Content-Retriever)

Base framework scripts complete. Addition of utility/tool scripts ongoing.
