from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from text_generation import Client

def tokenize(fname, window_size, step_size)
    
    #extract
    text = extract_text(fname)
    text = " ".join(text.split())
    text_tokens = text.split()

    sentences = []
    for i in range(0, len(text_tokens), step_size):
        window = text_tokens[i : i + window_size] #window size determines the common phrases between the sentences for context
        sentences.append(window)
        if len(window) < window_size:
            break
            
    return sentences
    
    

def embed(fname, window_size=128, step_size=100, top-k=32):
    
    sentences=tokenize(fname, windows_size, step_size)   

    paragraphs = [" ".join(s) for s in sentences]
    
    embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2") #performance:69.57
    embedding_model.max_seq_length = 512
    
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")#for scoring

    embeddings = embedding_model.encode(paragraphs, show_progress_bar=True, convert_to_tensor=True)
    
    return model, cross_encoder, embeddings, paragraphs


def search(query, model, cross_encoder, embeddings, paragraphs, top_k):
    
    query_embeddings = model.encode(query, convert_to_tensor=True)
    #query_embeddings = query_embeddings.cuda()
    
    hits = util.semantic_search(query_embeddings, embeddings, top_k=top_k)[0] #single query

    cross_input = [[query, paragraphs[hit["corpus_id"]]] for hit in hits]
    
    cross_scores = cross_encoder.predict(cross_input)
    
    #adding score to hits
    for idx in range(len(cross_scores)):
        hits[idx]["cross_score"] = cross_scores[idx]

    results = []
    hits = sorted(hits, key=lambda x: x["cross_score"], reverse=True)
    
    #appending to results
    for hit in hits[:5]:
        results.append(paragraphs[hit["corpus_id"]].replace("\n", " "))
    return results

#Model Config
PREPROMPT = "Conversation between a human and AI, AI tries to be helpful but practical also, avoiding false answers"
PROMPT = """"Use the following pieces of context to answer the question at the end.
If you don't know the answer, don't try to
make up an answer. Don't make up new terms and facts which are not available in the context.
{context}"""

END_7B = "\n<|prompter|>{query}<|endoftext|><|assistant|>"

PARAMS = {
    "temperature": 0.9,
    "top_p": 0.95,
    "repetition_penalty": 1.2,
    "top_k": 50,
    "truncate": 1000,
    "max_new_tokens": 1024,
    "seed": 42,
    "stop_sequences": ["<|endoftext|>", "</s>"],
}

LLM_7B = Client("http://localhost:8080/")

#docker run -p 8080:80 -v  /home/abshk/Desktop/llm/data/DockerDesktop:/home/abshk/Desktop/llm/data/DockerDesktop -e LOG_LEVEL=info,text_generation_router=debug ghcr.io/huggingface/text-generation-inference:0.9.1 --model-id OpenAssistant/falcon-7b-sft-top1-696 --num-shard 1
#docker run --shm-size 1g -p 8080:80 -v /home/abshk/.docker/desktop/vms/0/data:/home/abshk/.docker/desktop/vms/0/data ghcr.io/huggingface/text-generation-inference:1.0.3 --model-id tiiuae/falcon-7b-instruct --disable-custom-kernels


def ask(model, cross_encoder, embeddings, paragraphs):
    print(embeddings.shape)
    while True:
        
        print("\n")
        query = input("Enter query: ")
        results = search(query, model, cross_encoder, embeddings, paragraphs, top_k=5)

        query_7b = PREPROMPT + PROMPT.format(context="\n".join(results))
        query_7b += END_7B.format(query=query))

        text = ""
        
        for output in LLM_7B.generate_stream(query_7b, **PARAMS):
            if not output.token.special:
                text += response.token.text

        print("\nMODEL")
        print(text)