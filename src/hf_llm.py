"""
Custom LangChain chat model backed by huggingface_hub.InferenceClient.
HuggingFaceEndpoint hangs on init in langchain-huggingface 1.2+ due to
provider-routing API changes; InferenceClient with provider='auto' works.
"""
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import PrivateAttr
from huggingface_hub import InferenceClient


class HFInferenceChat(BaseChatModel):
    model_id: str
    hf_token: str
    max_tokens: int = 512
    temperature: float = 0.1

    _client: Any = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = InferenceClient(token=self.hf_token, provider="auto")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        hf_msgs = []
        for m in messages:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "assistant"
            hf_msgs.append({"role": role, "content": m.content})

        result = self._client.chat_completion(
            messages=hf_msgs,
            model=self.model_id,
            max_tokens=self.max_tokens,
        )
        content = result.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "hf_inference_chat"
