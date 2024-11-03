from pinecone import Pinecone, ServerlessSpec
from pinecone.data import Index
from config import PINECONE_API_KEY
def get_pinecone_index(index_name: str,dim_size: int,metric: str = "cosine") -> Index:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dim_size,
            metric=metric,
            spec=  ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
    return pc.Index(index_name)