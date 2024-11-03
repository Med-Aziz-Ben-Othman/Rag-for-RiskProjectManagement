import sys 
import os 
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import ExactMatchFilter
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.vector_stores.pinecone import PineconeVectorStore
from utils.logger import Logger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'llms','OpenAiCompatible')))
from llms.OpenAICompatible import OpenAiCompatible  # Adjusted import statement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'embeddings','OpenAIEmbeddings')))
from embeddings.OpenAIEmbeddings import OpenAIEmbeddings
import json

class RetrieverWithMetadata(BaseRetriever):
    """Retriever over a Pinecone vector store."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: OpenAIEmbeddings,
        openai_llm: OpenAiCompatible,  # Changed to OpenAiCompatible
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._llm = openai_llm  # Use OpenAiCompatible instance
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self.logger = Logger.get_logger(self)
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        meta_data = self.extract_metadata(query_bundle)
        metadata_filters = self.convert_to_filters(meta_data)
        
        if metadata_filters:
            self.logger.info(f"Applying metadata filters: {meta_data} for query: {query_bundle.query_str}")
        if not meta_data["contextual_query"]:
            raise Exception("No Contextual Query Found")
        query_bundle.query_str = meta_data["contextual_query"]
        if query_bundle.embedding is None:
            query_embedding = self._embed_model.embed(query_bundle.query_str)
        else:
            query_embedding = query_bundle.embedding
        
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
            filters=metadata_filters
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        print(f"Score Found for metadata {meta_data} : {list(map(lambda x: x.score, nodes_with_scores))}")
        return nodes_with_scores

    def extract_metadata(self, query_bundle: QueryBundle) -> dict:
        """Extract metadata from the query using the OpenAiCompatible LLM."""    
        system_prompt = ''' ... '''  # Keep your existing system prompt
        
        user_prompt = f"""
            Query : {query_bundle.query_str}
            JSON output:
        """

        # Use the OpenAiCompatible LLM to get the metadata in JSON format
        llm_response = self._llm.chat([  # Adjusted to use OpenAiCompatible
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # Parse the response as JSON
        answer = llm_response.split("{")[-1].split("}")[0]
        answer = "{" + answer + "}"
        try:
            metadata = json.loads(answer)
        except json.JSONDecodeError:
            self.logger.warning(f"Extracted Metadata but not in json format: + {llm_response} - {answer}")
            metadata = {}

        return metadata

    def convert_to_filters(self, metadata: dict) -> Optional[MetadataFilters]:
        """Convert the extracted metadata into exact match filters."""
        filters = []

        law_number = metadata.get('law_number')
        if law_number is not None:
            filters.append(ExactMatchFilter(key="law_number", value=law_number))

        article_number = metadata.get('article_number')
        if article_number is not None:
            filters.append(ExactMatchFilter(key="reference", value=article_number))

        if not filters:
            return None

        return MetadataFilters(filters=filters)
