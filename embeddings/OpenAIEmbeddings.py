import openai
from typing import Any, Optional
from embeddings.Embedding import Embedding
from utils.logger import Logger
import numpy as np

class OpenAIEmbeddings(Embedding):
    def __init__(self, model_name="text-embedding-3-small"):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name (str): The name of the model to use from OpenAI. Default is "text-embedding-3-small".
            cache_store (Optional[CacheStore]): An optional cache store for caching embeddings.
        """
        super().__init__()
        self.model_name = model_name
        self.logger = Logger.get_logger(self)

    def compute_embedding(self, content: str) -> Any:
        """
        Embed the content using the OpenAI model.

        Args:
            content (str): The content to embed.

        Returns:
            Any: The embedding for the content.
        """
        self.logger.debug('Embedding content using OpenAI model.')
        try:
            response = openai.Embedding.create(input=content, model=self.model_name)
            embedding = response['data'][0]['embedding']
            self.logger.debug('Computed embedding: %s', embedding)
            return np.array(embedding)  # Ensure the result is a numpy array
        except Exception as e:
            self.logger.error('Error computing embedding: %s', e, exc_info=True)
            raise

    def similarity(self, embedding_1: Any, embedding_2: Any) -> float:
        """
        Compute the similarity between two embeddings.

        Args:
            embedding_1 (Any): The first embedding.
            embedding_2 (Any): The second embedding.

        Returns:
            float: The similarity between the two embeddings.
        """
        self.logger.debug('Computing similarity between two embeddings.')
        try:
            # Compute cosine similarity between the embeddings
            similarity = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
            self.logger.debug('Computed similarity: %s', similarity)
            return similarity
        except Exception as e:
            self.logger.error('Error computing similarity: %s', e, exc_info=True)
            raise
