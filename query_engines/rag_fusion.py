from llama_index.core.query_engine import CustomQueryEngine
from llms.LLM import LLM
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.retrievers import BaseRetriever
from utils.logger import Logger
import concurrent.futures

    # system_prompt : str = f"""
    # You are an expert legal assistant.
    # Given a complexe query about legal information from Icelandic laws,
    # generate several specific and relevant sub-queries to search for in legal documents.

    # - if a history of the conversation exists , make sure to not confuse data present in the history with extracted sub queries , only infer from it the context of the sub queries to be more specific
    # - If The sub query Requires to have the law number and articles there , please include them in the sub query only if they are **Explictly** mentioned in the original query , Never infer them on your own
    # - In Case you included a reference or a law or article number , Each Sub Query should contain At Most 1 law number and article refernces , if you need more , produce multiple queries with different refences , Only if **Explicit** refernces are available that is
    # - do not start the query with What is ? or how ? , write it in a way such that is mentioned just like in a legal document 
    # - the subqueries are each in a new line    
    # - the subqueries should follow a legal style and should be specific to the query 
       
    # Answer directly with the sub-queries, without any other words like apologies or anything.
    # """

class RagFusionQueryEngine(CustomQueryEngine):
    llm: LLM = None
    retriever: BaseRetriever = None
    response_synthesizer: BaseSynthesizer = None
    logger: Logger = None
    n_keep: int = None
    system_prompt : str = f"""
    You are an expert legal assistant.
    Given a complexe query about legal information from Icelandic laws,
    generate several specific and relevant sub-queries to search for in legal documents.

    - If The sub query Requires to have the law number and articles there , please include them in the sub query only if they are **Explictly** mentioned in the original query , Never infer them on your own
    - In Case you included a reference or a law or article number , Each Sub Query should contain At Most 1 law number and article refernces , if you need more , produce multiple queries with different refences , Only if **Explicit** refernces are available that is
    - do not start the query with What is ? or how ? , write it in a way such that is mentioned just like in a legal document 
    - the subqueries are each in a new line     
       
    Answer directly with the sub-queries, without any other words like apologies or anything.
    """
    
    query_history : list = []

    
    def __init__(self, llm: LLM, retriever: BaseRetriever, response_synthesizer: BaseSynthesizer,n_keep=20):
        """
        Initialize the RagFusionQueryEngine with necessary components.

        Args:
            llm (LLM): The language model instance for generating sub-queries.
            retriever: The retriever instance for querying.
            response_synthesizer: The response synthesizer instance for generating answers.
        """
        super().__init__(retriever=retriever)
        self.llm = llm
        self.response_synthesizer = response_synthesizer
        self.retriever = retriever
        self.n_keep = n_keep
        self.logger = Logger.get_logger(self)
        self.query_history.append({
            "role": "system",
            "content": self.system_prompt
        })
    def generate_queries(self, original_query: str) -> list:
        """
        Generate multiple sub-queries from the original query using the LLM.

        Args:
            original_query (str): The original query string.

        Returns:
            list: A list of generated sub-queries.
        """
        try:
            prompt = f"""
            Query To Break Down: {original_query}
            Sub-queries , each on a a new line :
            """
            self.query_history.append({"role": "user", "content": prompt})
            response = self.llm.chat(messages=self.query_history)
            sub_queries = response.content.split('\n')
            return [query.strip() for query in sub_queries if query.strip()]
        except Exception as e:
            self.logger.error(f"Error generating sub-queries: {e}")
            raise

    def custom_query(self, query_str: str):
        """
        Process a query by generating sub-queries, retrieving documents for each sub-query,
        and merging the results.

        Args:
            query_str (str): The original query string.

        Returns:
            The synthesized response from the query engine.
        """
        try:
            # Generate sub-queries
            sub_queries = self.generate_queries(query_str)
            self.logger.info(f"Original query: {query_str}")
            self.logger.info(f"Generated sub-queries: {sub_queries}")

            # Retrieve documents for each sub-query
            all_retrieval_results = self.retrieve_parallel(sub_queries)
                
            # sort by score and only keep the top k
            all_retrieval_results = sorted(all_retrieval_results, key=lambda x: x.score, reverse=True)[:self.n_keep]
            
            # Synthesize the response from merged results
            response = self.response_synthesizer.synthesize(query_str, all_retrieval_results)
            
            self.query_history.append({
                "role": "assistant",
                "content": response.response
            })
            
            return response
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            raise


    # Retrieve documents for each sub-query in parallel
    def retrieve_parallel(self, sub_queries):
        all_retrieval_results = []

        # Define the function to retrieve results for a single sub-query
        def retrieve_sub_query(sub_query):
            return self.retriever.retrieve(sub_query)

        # Use ThreadPoolExecutor to run tasks in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all sub-queries for parallel execution
            future_to_query = {executor.submit(retrieve_sub_query, sub_query): sub_query for sub_query in sub_queries}

            # Wait for all tasks to complete and collect results
            for future in concurrent.futures.as_completed(future_to_query):
                retrieval_results = future.result()
                all_retrieval_results.extend(retrieval_results)

        return all_retrieval_results
