"""
modelFactory.py — Factory Pattern for model selection.

Encapsulates the threshold logic and model names so callers
just say factory.create(turns) and get the right model back.
Swapping models or thresholds only requires changing loadEnv / .env.
"""
from langchain_ollama import ChatOllama
from loadEnv import BASIC_MODEL, ADVANCED_MODEL, COMPLEXITY_THRESHOLD

class ModelFactory:
    def __init__(self, threshold: int = COMPLEXITY_THRESHOLD):
        self._threshold = threshold

    def create(self, human_turns: int, has_tools: bool = False) -> ChatOllama:
        """Return a fresh ChatOllama instance chosen by turn count.
        has_tools=True always picks ADVANCED_MODEL — small models lack reliable tool-calling.
        """
        name = ADVANCED_MODEL if (has_tools or human_turns > self._threshold) else BASIC_MODEL
        return ChatOllama(model=name)
