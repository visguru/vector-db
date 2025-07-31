# Install required packages (run these only once per environment)
!pip install pymupdf
!pip install --upgrade langchain langchain_community langchain_openai
!pip install pinecone-client==3.1.0 pinecone_notebooks==0.1.1

# Load and split the document
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

raw_documents = PyMuPDFLoader('/content/apple_10k.pdf').load()
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100, separator='\n')
chunks = text_splitter.split_documents(raw_documents)

# Set up OpenAI API key
import os
from google.colab import userdata
OPENAI_API_KEY = userdata.get('open_ai_key')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Generate embeddings for each chunk
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

docs = []
for idx, chunk in enumerate(chunks, 1):
    values = embeddings.embed_query(chunk.page_content)
    docs_json = {
        'id': str(idx),
        'values': values,
        'metadata': {
            'text': chunk.page_content,
            'source': chunk.metadata['source'],
            'author': 'vish',
            'createdon': '05-09-2025'
        }
    }
    docs.append(docs_json)

# Pinecone setup and index creation
import os
if not os.environ.get("pinecone_key"):
    from pinecone_notebooks.colab import Authenticate
    Authenticate()

from pinecone import Pinecone, ServerlessSpec

os.environ["PINECONE_API_KEY"] = userdata.get('pinecone_key')
api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

cloud = os.environ.get("PINECONE_CLOUD") or 'aws'
region = os.environ.get("PINECONE_REGION") or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = "semantic-search-apple"

if index_name not in pc.list_indexes():
    pc.create_index(index_name, dimension=len(docs[0]['values']), spec=spec, metric='cosine')

index = pc.Index(index_name)

# Upsert documents to Pinecone in batches
batch_size = 50
for i in range(0, len(docs), batch_size):
    batch = docs[i: i + batch_size]
    index.upsert(vectors=batch)

# Query the vector database
query = "can you tell me total number of share holders as of 2022?"
xq = embeddings.embed_query(query)
xc = index.query(vector=xq, top_k=5, include_metadata=True)

# Print query results in readable format
import json
print(json.dumps(xc.to_dict(), indent=2))
