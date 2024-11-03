from typing import List, Optional
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever

class RetrieverWrapper(BaseRetriever):
    """
    A retriever wrapper that takes another retriever and post-processes the results
    with a list of post-processors.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
    ) -> None:
        """
        Initialize the retriever wrapper.

        Args:
            retriever (BaseRetriever): The underlying retriever.
            node_postprocessors (Optional[List[BaseNodePostprocessor]]): A list of post-processors to apply on the retrieved nodes.
        """
        self.retriever = retriever
        self.node_postprocessors = node_postprocessors or []

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieves nodes using the underlying retriever and applies post-processors.

        Args:
            query_bundle (QueryBundle): The query bundle containing the query string and other metadata.

        Returns:
            List[NodeWithScore]: A list of post-processed nodes with their scores.
        """
        # Step 1: Retrieve nodes from the underlying retriever
        retrieved_nodes = self.retriever._retrieve(query_bundle)

        # Step 2: Apply post-processors sequentially
        for postprocessor in self.node_postprocessors:
            retrieved_nodes = postprocessor.postprocess_nodes(
                nodes=retrieved_nodes, query_bundle=query_bundle
            )

        # Return the post-processed nodes
        return retrieved_nodes
