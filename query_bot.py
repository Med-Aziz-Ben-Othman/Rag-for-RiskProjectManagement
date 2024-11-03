import sys 
import os 
from config import OPENAI_KEY, MONGO_URI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'embeddings','OpenAIEmbeddings')))
from embeddings.OpenAIEmbeddings import OpenAIEmbeddings
from indexes.pinecone import get_pinecone_index
from interfaces.cli import simple_cli_interaction_interface
from interfaces.web import streamlit_interaction_interface
from llms.LLM import LLM
from llms.OpenAI_LLM import OpenAI_LLM
from llms.llama_index.llm_wrapper import LamaIndexLLM
from llms.rerankers.OpenAIReranker import OpenAIReranker
from post_processor.ranker import RerankerPostProcessor
from query_engines.rag_fusion import RagFusionQueryEngine
from query_engines.simple_qe import SimpleQueryEngine
from retrievers.reranker_retriever import RerankRetrieverWrapper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'retrievers')))
from retrievers.retriever_with_metadata import RetrieverWithMetadata
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'retrievers','simple_retriever')))
from retrievers.simple_retriever import SimpleRetriever
from retrievers.retriever_with_processors import RetrieverWrapper
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import BaseRetriever
from utils.logger import Logger
root_logger = Logger.get_root_logger("legal_bot_query")


def initialize_embedding_model():
    """Initialize the OpenAI embedding model."""
    return OpenAIEmbeddings(
        model_name="text-embedding-3-small"  # Specify the embedding model name
    )


def initialize_llm():
    """Initialize the OpenAI LLM model."""
    return OpenAI_LLM(
        api_key=OPENAI_KEY,  # Pass the OpenAI API key here
        model="gpt-4o-mini",  # Specify the model name
        options={  # Include any specific options for the model
            "temperature": 0,
        }
    )

def init_reponse_synth(llm):
        system_prompt = """
            You are a legal Expert in icelandic laws
            Do Not Hallucinate or Applogies
            Given a query about legal information from icelandic 
            And A Set of Context From Icelandic Law That Might Be Relevant to the query Or Not Relevant To The Query
            The Context have also the sources : ( Law Number And Reference ( Article Number ))
            Answer the query with the most relevant information from the context 
            If An Answer is not available or is not correct or data is missing , Report the issue instead of hallucinating
            Finally Explain your answer with references from the context  
        """
        response_llm = LamaIndexLLM(llm, system_prompt)
        response_synthesizer = get_response_synthesizer(llm=response_llm, verbose=True)
        return response_synthesizer

def initialize_reranker() -> OpenAIReranker:
    """Initialize the OpenAI-compatible reranker using OpenAIReranker."""
    try:
        # Ensure you have set the OpenAI API key in the environment or configuration
        if not OPENAI_KEY:
            raise ValueError("The OpenAI API key is not set.")
        
        # Initialize the OpenAI reranker
        openai_reranker = OpenAIReranker(api_key=OPENAI_KEY)
        return openai_reranker
    except Exception as e:
        root_logger.error(f"Error initializing reranker: {e}")
        raise


def rewrite_query(llm: LLM, original_query: str) -> str:
    """
    Rewrite the user's query using a language model.
    
    Args:
        llm (LLM): The language model instance for rewriting the query.
        original_query (str): The original user query.
    
    Returns:
        str: The rewritten query.
    """
    try:
        prompt = f"""
        Given a query about Risks and Project Managment information from Projects
        rewrite the query to be more specific and concise.
        OriginalQuery: {original_query}
        Answer directly with the new query, without any other words.
        """
        rewritten_query = llm.chat(messages=[{"role": "user", "content": prompt}])
        return rewritten_query.content
    except Exception as e:
        root_logger.error(f"Error rewriting query: {e}")
        raise


def initialize_storage_context(embedding) -> StorageContext:
    """Initialize the vector store and document store."""
    try:
        pinecone_index = get_pinecone_index("risk_managment", embedding.size)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        docstore = MongoDocumentStore.from_uri(
            uri=MONGO_URI
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
        return storage_context
    except Exception as e:
        root_logger.error(f"Error initializing storage context: {e}")
        raise


def initialize_retriever(metdata_llm,embedding, storage_context, reranker):
    """Initialize the retriever with post-processors and reranker."""
    try:
        simple_retriever = RetrieverWithMetadata(
            vector_store=storage_context.vector_store,
            embed_model=embedding,
            metdata_llm=metdata_llm,
            query_mode="default",
            similarity_top_k=20
        )
        merging_retriever = AutoMergingRetriever(simple_retriever, storage_context, verbose=False)

        simple_filtered_retriever = RetrieverWrapper(
            merging_retriever,
            [SimilarityPostprocessor(similarity_cutoff=0.3, filter_empty=True, filter_duplicates=True)]
        )
        
        return RetrieverWrapper(    
            RerankRetrieverWrapper(simple_filtered_retriever, reranker),
            [SimilarityPostprocessor(similarity_cutoff=0.4)]
        )
    except Exception as e:
        root_logger.error(f"Error initializing retriever: {e}")
        raise


def initialize_query_engine(llm,response_synthesizer, retriever) -> BaseRetriever:
    """Initialize the query engine with retriever and LLM response synthesizer."""
    try:
        return RagFusionQueryEngine(
            llm = llm,
            retriever=retriever,
            response_synthesizer=response_synthesizer
        )
    except Exception as e:
        root_logger.error(f"Error initializing query engine: {e}")
        raise

# Static test interface
def static_test_interface(query_engine):
    """Static test for predefined queries."""
    original_query = "I have issues with telecommunications in Iceland, what are the laws regarding this?"
    # original_query = " what is article 2 or 3 about in telecommunication law ?"
    answer = query_engine.query(original_query)
    print("Legal Assistant:", answer)


# Main function to run the system with different interfaces
def main(interface: str):
    embedding = initialize_embedding_model()
    llm = initialize_llm()
    reranker = initialize_reranker()
    storage_context = initialize_storage_context(embedding)
    retriever = initialize_retriever(llm,embedding, storage_context, reranker)
    response_synth = init_reponse_synth(llm)
    query_engine = initialize_query_engine(llm,response_synth, retriever)

    if interface == "static_test":
        static_test_interface(query_engine)
    elif interface == "cli":
        simple_cli_interaction_interface(query_engine)
    elif interface == "web":
        streamlit_interaction_interface(query_engine)
    else:
        print(f"Unknown interface: {interface}")
        print("Supported interfaces: 'static_test', 'cli'.")

if __name__ == "__main__":
    # Choose the interface: 'static_test' or 'cli'
    selected_interface = input("Select interface ('static_test' or 'cli'): ").strip()
    main(selected_interface)
