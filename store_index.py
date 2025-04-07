from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

## PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_KEY="pcsk_5jTuLc_Cdv36AvLzQWNxDofXGuaXbPvU4UBShhMjy4x9Cp9CxuJw1haFKMvDhBvKUEkEv6"

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data=load_pdf_file(data='Data/')
text_chunk=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)


# Ember each chunk and upsert the embeddings into your Pinecone index 
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunk,
    index_name=index_name,
    embedding=embeddings,
)




