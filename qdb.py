from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from qdrant_client.http.models import PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import CSVLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader
from InstructorEmbedding import INSTRUCTOR
from load import ocr
import numpy as np
import os

from config import QEMB_CHUNK_SIZE, QEMB_CHUNK_OVERLAP

#encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#encoder = SentenceTransformer("hkunlp/instructor-large")
#Requires pip install git+https://github.com/UKPLab/sentence-transformers.git
encoder = INSTRUCTOR('hkunlp/instructor-large')
instruction = ["Represent the text content:"]

CHUNK_SIZE=QEMB_CHUNK_SIZE
CHUNK_OVERLAP=QEMB_CHUNK_OVERLAP

ext2loader = {
    ".csv": (CSVLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

def load_document(file_path, existing_files):

    split_ls = file_path.rsplit(".")
    extension = "." + split_ls[1]
    fname = split_ls[0].split("/")[-1]

    if extension in ext2loader:
        if fname not in existing_files:

            loader_type, loader_args = ext2loader[extension]
            loader = loader_type(file_path, **loader_args)
            load=loader.load()
            return load, fname
        else:
            return None, fname

    raise ValueError(f" '{extension}' file type not supported")

def split_document(file_path, existing_files=[]):

    document, fname = load_document(file_path, existing_files)
    if not document:
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_text = text_splitter.split_documents(document)

    payload=[]
    content=[]
    for docs in split_text:
      payload.append({
          'source' : docs.metadata['source'],
          'content' : docs.page_content
          })
      content.append(docs.page_content)

    return payload, content

def get_collections():

  path = "QDB/collection"
  if os.path.exists(path):
    items = os.listdir(path)

    folders = [item for item in items if os.path.isdir(os.path.join(path, item))]

    return folders
  else:
     return None
  
def qdb_embed(db_path,fpath, cname=None, scanned=False):

  client = QdrantClient(path=db_path)
  collections = client.get_collections()
  path,ext = fpath.split('.')
  if cname:
     collection = cname
  else:
    collection = path.split('/')[-1]

  if collection not in [c.name for c in collections.collections]:
    print(f"Creating Collection: {fpath}")
    client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=Distance.COSINE),
    )
  else:
    print(f"Embeddings for Document: '{fpath}' already exists")
    client.close()
    return None

  if scanned or ext in ['jpeg','jpg','png']:
        fpath=ocr(fpath)
        print("OCR Extraction Completed")

  payload, content = split_document(fpath,[])
  ids=np.arange(1,len(content)+1)
  instruct_load=[[instruction[0], c] for c in content]
  info = client.upsert(
      collection_name = collection,
      points = Batch(

          ids=ids.tolist(),
          payloads = payload,
          vectors=encoder.encode(instruct_load).tolist()

      )
  )

  print(f"Document: '{fpath}' is embedded and stored in Vector DB")
  client.close()
  return True

def qdb_search(db_path, collection, query):

  client = QdrantClient(path=db_path)
  collections = client.get_collections()

  if collection not in [c.name for c in collections.collections]:
    print(f"Collection: '{collection}' does not exist")
    return None

  results = client.search(
      collection_name=collection,
      query_vector=encoder.encode([['Represent the text content:',query]]).tolist()[0],
      limit=3,
  )
  client.close()
  context = [text.payload['content'] for text in results]
  return results, context
