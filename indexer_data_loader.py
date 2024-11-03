# Import necessary modules
from llama_index.core.schema import RelatedNodeInfo, NodeRelationship
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from config import OPENAI_KEY
from embeddings.OpenAIEmbeddings import OpenAIEmbeddings
from indexes.pinecone import get_pinecone_index
from llms.OpenAI_LLM import OpenAI_LLM
from utils.dataloader import DataLoader
from utils.feature_extractor import LegalFeatureExtractor
from utils.logger import Logger
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core.schema import MetadataMode
from utils.utils import get_hash

# Initialize root logger
root_logger = Logger.get_root_logger("legal_bot_indexer")

def initialize_llm():
    """Initialize the OpenAI LLM model."""
    return OpenAI_LLM(
        api_key=OPENAI_KEY,  # Pass the OpenAI API key here
        model="gpt-4o-mini",  # Specify the model name
        options={  # Include any specific options for the model
            "temperature": 0,
        }
    )

def initialize_embedding():
    """Initialize the OpenAI embedding model."""
    return OpenAIEmbeddings(
        model_name="text-embedding-3-small"  # Specify the embedding model name
    )

def initialize_vector_store(embedding):
    """Initialize the Pinecone Vector Store."""
    pinecone_index = get_pinecone_index("risk_managment", embedding.size)
    return PineconeVectorStore(pinecone_index=pinecone_index)

def initialize_docstore():
    """Initialize the MongoDB Document Store."""
    return MongoDocumentStore.from_uri(
        uri="mongodb+srv://MedAziz:<db_password>@cluster0.0be9q.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    )

def process_risk_document(risk_document, feature_extractor, embedding, vector_store, docstore):
    """Process law documents: extract features, generate chunks, and store them."""
    for risk in risk_document:
        risk_feature = extract_risk_feature(risk, feature_extractor)
        chunks = generate_chunks(risk_feature, embedding)
        
        store_chunks(chunks, vector_store)
        store_documents([risk_feature, risk], docstore)

def extract_risk_feature(risk: Document, feature_extractor):
    """Extract features from the risk document."""
    features = feature_extractor.extract_context(risk.text)
    risk_feature = Document(
        text=features,
        metadata=risk.metadata,
        doc_id=get_hash(features)
    )
    # Establish relationships for the features document
    risk_feature.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=risk.doc_id)
    return risk_feature

def generate_chunks(risk_feature: Document, embedding):
    """Generate chunks from risk features using sentence splitting."""
    node_parser = SentenceSplitter.from_defaults(chunk_size=64, chunk_overlap=20)
    chunks = node_parser.get_nodes_from_documents([risk_feature])
    
    for chunk in chunks:
        chunk.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=risk_feature.doc_id)
        chunk.embedding = embedding.embed(chunk.get_content())
        chunk.metadata["isleaf"] = True
        chunk.excluded_embed_metadata_keys = ["isleaf"]
        chunk.excluded_llm_metadata_keys = ["isleaf"]
        chunk.id_ = get_hash(chunk.get_content(metadata_mode=MetadataMode.NONE))
        
    return chunks

def store_chunks(chunks, vector_store):
    """Store chunks in the vector store."""
    vector_store.add(chunks)

def store_documents(documents, docstore):
    """Store documents in the document store."""
    docstore.add_documents(documents)

def main():
    """Main function to initialize components and process risk documents."""
    llm = initialize_llm()
    embedding = initialize_embedding()
    feature_extractor = LegalFeatureExtractor(llm)
    vector_store = initialize_vector_store(embedding)
    docstore = initialize_docstore()
    
    dataloader = DataLoader("data/Icelandicrisks")
    
    for docs in dataloader.load_data():  # Ensure the correct method name is used
        process_risk_document(docs, feature_extractor, embedding, vector_store, docstore)

if __name__ == "__main__":
    main()
