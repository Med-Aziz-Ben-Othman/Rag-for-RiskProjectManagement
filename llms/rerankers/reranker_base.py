from abc import ABC, abstractmethod
from typing import List, Any

class AbstractReranker(ABC):
    """Abstract base class for a custom reranker."""

    @abstractmethod
    def rerank(self, query_prompt: str, document_prompt: List[str],top_n: int | None) -> List[Any]:
        """
        Abstract method to rerank a list of documents based on a query.-

        Args:
            query_prompt (str): The query to rank documents against.
            document_prompt (List[str]): A list of documents to be ranked.
            top_n (int | None): The number of top documents to return. If None, return all documents.

        Returns:
            List[Any]: A list of ranked documents or scores.
        """
        pass
