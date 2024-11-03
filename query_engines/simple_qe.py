from llama_index.core.query_engine import CustomQueryEngine
from llms.LLM import LLM
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever

from utils.logger import Logger

class SimpleQueryEngine(CustomQueryEngine):
    llm: LLM = None
    retriever: BaseRetriever = None
    response_synthesizer: BaseSynthesizer = None
    logger: Logger = None
    system_prompt: str = f"""
            You are an expert legal assistant
            Given a query about legal information from Icelandic laws,
            rewrite the query to be more specific and relevant to search for in legal documents.
            do not add any knowledge beyond what is mentioned in the query.
            Answer directly with the new query, without any other words.
            """
    query_history: list = []
    
    def __init__(self, llm: LLM, retriever: BaseRetriever, response_synthesizer: BaseSynthesizer):
        """
        Initialize the SimpleQueryEngine with necessary components.

        Args:
            llm (LLM): The language model instance for query rewriting.
            retriever: The retriever instance for querying.
            response_synthesizer: The response synthesizer instance for generating answers.
            system_prompt (str): System prompt for the LLM.
        """
        super().__init__(retriever=retriever)
        self.llm = llm
        self.response_synthesizer = response_synthesizer
        self.retriever = retriever
        self.logger = Logger.get_logger(self)
        self.query_history.append({
            "role": "system",
            "content": self.system_prompt
        })


    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite the user's query using the language model.

        Args:
            original_query (str): The original user query.

        Returns:
            str: The rewritten query.
        """
        try:
            prompt = f"""
            Query To Break Down: {original_query}
            """
            self.query_history.append({"role": "user", "content": prompt})
            rewritten_query = self.llm.chat(messages=self.query_history)
            return rewritten_query.content
        except Exception as e:
            self.logger.error(f"Error rewriting query: {e}")
            raise
    
    def custom_query(self, query_str: str):
        """
        Process a query by rewriting it, querying the retriever, and synthesizing the response.

        Args:
            query_str (str): The original query string.

        Returns:
            The synthesized response from the query engine.
        """
        try:
            # Rewrite the query using the LLM
            rewritten_query = self.rewrite_query(query_str)
            self.logger.info(f"Original query: {query_str}")
            self.logger.info(f"Rewritten query: {rewritten_query}")
            
            # Use the retriever to get the result
            retrieval_results = self.retriever.retrieve(rewritten_query)
            
            # Synthesize the response
            response = self.response_synthesizer.synthesize(
                rewritten_query,
                retrieval_results
            )
            self.query_history.append({
                "role": "assistant",
                "content": response.response
            })
            return response
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise
