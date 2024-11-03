from typing import Callable, List, Optional

from llama_index.core.bridge.pydantic import Field, PrivateAttr, SerializeAsAny
from llama_index.core.indices.utils import (
    default_format_node_batch_fn,
    default_parse_choice_select_answer_fn,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_CHOICE_SELECT_PROMPT
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.settings import Settings

from llms.rerankers.reranker_base import AbstractReranker

class RerankerPostProcessor(BaseNodePostprocessor):
    """LLM-based reranker."""

    reranker: AbstractReranker = None
    top_n: int | None = None
    def __init__(
        self,
        reranker: AbstractReranker,  # Take in the custom reranker
        top_n:int = None
    ) -> None:
        super().__init__()
        self.reranker = reranker
        self.top_n = top_n

    @classmethod
    def class_name(cls) -> str:
        return "RerankerProcessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        # Extract the text of the nodes
        node_texts = [node.node.get_content() for node in nodes]
        query_str = query_bundle.query_str

        # Use the custom reranker to rerank the nodes based on the query
        ranked_results = self.reranker.rerank(query_str, node_texts,top_n=self.top_n)

        # Convert the ranked results back into NodeWithScore objects
        reranked_nodes = []
        for (index, score) in ranked_results:
            # Find the corresponding node
            item = nodes[index]
            reranked_nodes.append(NodeWithScore(node=item.node, score=score))

        return sorted(reranked_nodes, key=lambda x: x.score or 0.0, reverse=True)
