from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

from llms.LLM import LLM

class LamaIndexLLM(CustomLLM):
    num_output: int = None
    context_window: int = None
    llm: LLM = None
    system_prompt: str = None
    chat_options: dict = None
    history: list = []
    def __init__(self,llm: LLM,system_prompt,max_context_window: int = 64000, num_output: int = 4096,chat_options=None):
        super().__init__()
        self.context_window = max_context_window
        self.num_output = num_output
        self.llm = llm
        self.system_prompt = system_prompt
        self.chat_options = chat_options
        if chat_options == None:
            self.chat_options = {}
        self.history = [
            {"role": "system", "content": self.system_prompt},
        ]
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.llm.model_name
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        self.history.append({"role": "user", "content": prompt})
        response = self.llm.chat(self.history,**self.chat_options)
        self.history.append({"role": "assistant", "content": response.content})
        return CompletionResponse(text=response.content)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        self.history.append({"role": "user", "content": prompt})
        response = self.llm.chat(self.history,**self.chat_options)
        self.history.append({"role": "assistant", "content": response.content})
        for token in response.content:
            response_text += token
            yield CompletionResponse(text=response_text, delta=token)
