from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from typing import List
from llama_index.core.postprocessor import LLMRerank

from post_processor.ranker import RerankerPostProcessor

class RerankRetrieverWrapper(BaseRetriever):
    """Wrapper retriever that applies reranking to the results of any retriever."""
    base_retriever: BaseRetriever = None
    reranker: RerankerPostProcessor = None
    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: RerankerPostProcessor
    ) -> None:
        """Initialize wrapper with a base retriever and optional reranking."""
        super().__init__()
        self.base_retriever = base_retriever
        self.reranker = reranker

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes from the base retriever and apply reranking if enabled."""
        # Retrieve the initial set of nodes using the base retriever
        nodes_with_scores = self.base_retriever._retrieve(query_bundle)
        nodes_with_scores = self.reranker.postprocess_nodes(
            nodes_with_scores, query_bundle
        )
        return nodes_with_scores

