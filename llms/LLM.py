from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLM(ABC):
    def __init__(self,model_name):
        self.model_name = model_name
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], options: Optional[Dict[str, Any]] = None) -> Any:
        """
        Perform a chat operation with the language model.

        Args:
            messages (List[Dict[str, Any]]): A list of message objects where each message is a dictionary 
                                             with keys such as 'role' and 'content'.
            options (Optional[Dict[str, Any]]): Optional parameters for the chat request.

        Returns:
            Any: The response from the language model.
        """
        pass
