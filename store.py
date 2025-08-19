from dotenv import load_dotenv
import os
from pinecone import  ServerlessSpec
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from src.helper import load_files,filter_to_docs,text_split,download_embeddings

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
API_KEY = os.getenv("API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["API_KEY"] = API_KEY

extracted_data = load_files(data="data/")
filter_data = filter_to_docs(extracted_data)
text_chunks = text_split(filter_data)

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "ai-medicalbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric= "cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1")     
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

