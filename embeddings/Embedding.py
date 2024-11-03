from abc import ABC, abstractmethod
from typing import Optional, Any
from utils.logger import Logger

class Embedding(ABC):
    def __init__(self,):
        """
        Initialize the Embedding with an optional cache store.

        Args:
            cache_store (Optional[CacheStore]): An instance of CacheStore for caching embeddings. Defaults to NoCacheStore.
        """
        self.logger = Logger.get_logger(self)

    def embed(self, query: str) -> Any:
        """
        Embed the content

        Args:
            query (str): The content to embed.

        Returns:
            Any: The embedding of the content.
        """
        self.logger.info('Embedding content')
        embedding = self.compute_embedding(query)
        return embedding

    @abstractmethod
    def compute_embedding(self, query: str) -> Any:
        """
        Compute the embedding for the given query.

        Args:
            query (str): The content to embed.

        Returns:
            Any: The embedding of the content.
        """
        pass
    @abstractmethod
    def similarity(self, query_1: str, query_2: str) -> float:
        """
        Compute the similarity between the embeddings of two queries.

        Args:
            query_1 (str): The first content to compare.
            query_2 (str): The second content to compare.

        Returns:
            float: The similarity score between the embeddings of the two queries.
        """
        pass
