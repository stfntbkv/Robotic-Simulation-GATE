from __future__ import annotations

from autogpt_p.helpers.singleton import Singleton
from autogpt_p.llm.chat_gpt_interface import ChatGPTInterface, GPT_4, GPT_3
from autogpt_p.llm.llm_interface import LLMInterface

GPT = "GPT"


class LLMFactory(Singleton):

    _instance = None

    @classmethod
    def get_instance(cls) -> LLMFactory:
        return cls._instance

    def __init__(self, llm_type: str, version=""):
        self.llm_type = llm_type
        self.version = version

    def produce_llm(self) -> LLMInterface:
        if self.llm_type == GPT:
            return ChatGPTInterface(GPT_4 if self.version == "4" else GPT_3)
