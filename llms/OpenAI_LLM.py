from typing import List, Dict, Any
import openai
from llms.LLM import LLM
from utils.logger import Logger
import time

class OpenAI_LLM(LLM):
    def __init__(self, api_key: str, model: str, options: Dict[str, Any] = None):
        """
        Initialize the OpenAI LLM with the OpenAI client.

        Args:
            api_key (str): The API key for authentication.
            model (str): The model name to use for completion.
            options (Dict[str, Any]): Optional parameters for the API request.
        """
        super().__init__(model)
        openai.api_key = api_key  # Set the API key for OpenAI
        self.options = options if options is not None else {}
        self.logger = Logger.get_logger(self)

    def chat(self, messages: List[Dict[str, Any]], max_retries: int = 3, initial_delay: float = 30.0, backoff_factor: float = 2.0) -> str:
        """
        Get a chat completion from the OpenAI service with a retry mechanism.

        Args:
            messages (List[Dict[str, Any]]): The messages to send to the model.
            max_retries (int): The maximum number of retries. Default is 3.
            initial_delay (float): The initial delay between retries in seconds. Default is 30.0.
            backoff_factor (float): The factor by which the delay increases after each retry. Default is 2.0.

        Returns:
            str: The model's response message.
        """
        self.logger.debug('Sending chat request with messages: %s', messages)
        attempt = 0
        delay = initial_delay

        while attempt < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=messages,
                    **self.options
                )
                self.logger.debug('Received response: %s', response)
                return response.choices[0].message['content']  # Access the content of the message
            except Exception as e:
                attempt += 1
                self.logger.error('Error during chat completion: %s', e, exc_info=True)
                if attempt >= max_retries:
                    raise
                else:
                    self.logger.info('Retrying chat completion in %s seconds...', delay)
                    time.sleep(delay)
                    delay *= backoff_factor
