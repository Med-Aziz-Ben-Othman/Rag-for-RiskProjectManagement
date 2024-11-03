from time import sleep
from typing import Any, List
import openai
from llms.rerankers.reranker_base import AbstractReranker
from utils.logger import Logger

class OpenAIReranker(AbstractReranker):
    """
    A reranker class that uses OpenAI models for reranking.
    """

    def __init__(self, api_key: str):
        """
        Initializes the OpenAI reranker with the necessary configuration.

        Args:
            api_key (str): The API key for authenticating the OpenAI service.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

        # Initialize logger
        self.logger = Logger.get_logger(self)
        self.logger.info("Initialized OpenAIReranker.")

    def rerank(self, query_prompt: str, document_prompt: List[str], top_n: int = None, max_retries: int = 3, backoff_factor: float = 2.0) -> List[tuple]:
        """
        Reranks a list of documents based on the query using OpenAI's model.

        Args:
            query_prompt (str): The query to rank documents against.
            document_prompt (List[str]): A list of documents to be ranked.
            top_n (int, optional): The number of top documents to return. Defaults to returning all.
            max_retries (int, optional): Maximum number of retries in case of errors. Defaults to 3.
            backoff_factor (float, optional): Time multiplier for backoff between retries. Defaults to 2.0.

        Returns:
            List[tuple]: A list of tuples with the document and its relevance score.
        """
        # Log the input query and documents
        self.logger.info(f"Reranking documents for query: {query_prompt}")

        ranked_results = []

        for document in document_prompt:
            prompt = f"Rate the relevance of the following document to the query: '{query_prompt}'. Document: '{document}'. Give a score from 0 to 1."
            
            attempt = 0
            while attempt < max_retries:
                try:
                    # Call the OpenAI API
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )

                    # Parse the response
                    score = float(response['choices'][0]['message']['content'])
                    ranked_results.append((document, score))
                    self.logger.info(f"Document scored: {score}")

                    break  # Exit the retry loop on success

                except Exception as e:
                    # Handle any exception related to API errors
                    self.logger.error(f"Request failed on attempt {attempt + 1}: {e}")

                    attempt += 1
                    if attempt < max_retries:
                        # Apply exponential backoff before retrying
                        sleep_time = backoff_factor ** attempt
                        self.logger.info(f"Retrying in {sleep_time} seconds...")
                        sleep(sleep_time)
                    else:
                        raise Exception(f"Max retries ({max_retries}) reached. Failed to rerank")

        # Sort the documents based on their scores
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # Return only the top_n results if specified
        return ranked_results[:top_n] if top_n else ranked_results
