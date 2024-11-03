from llms.LLM import LLM
from openai import OpenAI
from typing import List, Dict, Any, Optional
from utils.logger import Logger

class OpenAiCompatible(LLM):
    def __init__(self, api_key: str, model: str, url: str = "http://localhost:1234/v1", options: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAiCompatible with the given API key, model, and optional parameters.

        Args:
            api_key (str): The API key for authentication with the OpenAI service.
            model (str): The model to use for generating responses.
            url (str): The base URL of the OpenAI service. Defaults to "http://localhost:1234/v1".
            options (Optional[Dict[str, Any]]): Optional parameters for the chat request.
        """
        self.client = OpenAI(api_key=api_key, base_url=url)
        self.model = model
        self.options = options if options is not None else {}
        self.logger = Logger.get_logger(self)
        self.logger.debug('OpenAiCompatible initialized with model: %s and URL: %s', model, url)

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """
        Send a chat request to the OpenAI API and return the response.

        Args:
            messages (List[Dict[str, Any]]): A list of message objects, where each message is a dictionary 
                                             with keys such as 'role' and 'content'.

        Returns:
            str: The content of the response from the OpenAI API.

        Raises:
            RuntimeError: If an error occurs while communicating with the OpenAI API.
        """
        self.logger.debug('Sending chat request to OpenAI API with model: %s', self.model)
        self.logger.debug('Request messages: %s', messages)
        self.logger.debug('Request options: %s', self.options)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.options
            )
            response_content = response.choices[0].message
            self.logger.debug('Received response from OpenAI API: %s', response_content)
            return response_content
        except Exception as e:
            self.logger.error('Error occurred while communicating with the OpenAI API: %s', e, exc_info=True)
            raise e
