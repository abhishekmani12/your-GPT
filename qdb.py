from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from qdrant_client.http.models import PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import CSVLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader
import numpy as np

encoder = SentenceTransformer("hkunlp/instructor-large")
CHUNK_SIZE=256
CHUNK_OVERLAP=32

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


def qvdb_embed(db_path,fpath):

  client = QdrantClient(path=db_path)
  collections = client.get_collections()
  collection = fpath.split('.')[0].split('/')[-1]

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

  payload, content = split_document(fpath,[])
  ids=np.arange(1,len(content)+1)

  info = client.upsert(
      collection_name = collection,
      points = Batch(

          ids=ids.tolist(),
          payloads = payload,
          vectors=encoder.encode(content).tolist()

      )
  )

  print(f"Document: '{fpath}' is embedded and stored in Vector DB")
  client.close()
  return True

def qvdb_search(db_path, collection, query):

  client = QdrantClient(path=db_path)
  collections = client.get_collections()

  if collection not in [c.name for c in collections.collections]:
    print(f"Collection: '{collection}' does not exist")
    return None

  results = client.search(
      collection_name=collection,
      query_vector=encoder.encode(query).tolist(),
      limit=3,
  )
  client.close()
  return results

