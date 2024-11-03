import sys
import os
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.pinecone import PineconeVectorStore
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'embeddings','OpenAIEmbeddings')))
from embeddings.OpenAIEmbeddings import OpenAIEmbeddings
from llama_index.core.vector_stores.types import VectorStoreQuery

class SimpleRetriever(BaseRetriever):
    """Retriever over a Pinecone vector store using OpenAI embeddings."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embed_model: OpenAIEmbeddings,  # Use your OpenAI embedding model here
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        # Generate embeddings for the query using OpenAI
        if query_bundle.embedding is None:
            query_embedding = self._embed_model.embed(query_bundle.query_str)
        else:
            query_embedding = query_bundle.embedding

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
