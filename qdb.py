from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import CSVLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader

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
        print(f"Vector Embeddings for FILE:'{fname}' already exists")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_text = text_splitter.split_documents(document)

    text_string=""""""
    for doc in split_text:
      text_string += doc.page_content

    print("Document Split")
    return text_string, fname


def qvdb_embed(db_path, collection, fpath):

  client = QdrantClient(path=db_path)

  exist=True
  existing_docs=[]

  try:
    list_docs = client.search(
    collection_name=collection,
    query_vector=encoder.encode("").tolist())

  except ValueError:
    exist=False

  if exist:
    existing_docs=[]
    for doc in list_docs:
      existing_docs.append(doc.payload.get("source"))
  else:
    print(f"Creating Collection: {collection}")
    client.create_collection(
    collection_name=collection,
    vectors_config=VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=Distance.COSINE),
    )

  text, fname = split_document(fpath, existing_docs)

  if text:
    info = client.upsert(
    collection_name=collection,
    wait=True,
    points=[
        PointStruct(id=1,vector=encoder.encode(text).tolist(), payload={"source": fname})
      ]
    )

    print(info)
    print("Document Embedded and Stored in Vector DB")

  client.close()
