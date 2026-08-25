
import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node exists in the project path
from vishustra_core.nodes.base_node import BaseNode

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node that simulates language translation of text data.

    This node takes a string as input and simulates its translation
    to a specified target language. The target language can be set
    during initialization or overridden for a specific processing call
    via the 'context' dictionary.
    """

    def __init__(self, target_language: str = "en", source_language: Optional[str] = None):
        """
        Initializes the LanguageTranslatorNode.

        Args:
            target_language: The default language to translate the text into
                             (e.g., "es", "fr", "de"). Defaults to "en".
                             Must be a non-empty string.
            source_language: The expected source language. If not provided,
                             it's assumed to be auto-detected or irrelevant
                             for this simulation. Must be a non-empty string if provided.
        
        Raises:
            ValueError: If target_language is not a valid non-empty string,
                        or if source_language is provided but not a valid non-empty string.
        """
        if not isinstance(target_language, str) or not target_language:
            raise ValueError("target_language must be a non-empty string.")

        if source_language is not None and (not isinstance(source_language, str) or not source_language):
            raise ValueError("source_language, if provided, must be a non-empty string.")

        self._default_target_language = target_language.lower()
        self._default_source_language = source_language.lower() if source_language else None
        self._logger = logging.getLogger(self.__class__.__name__)
        
        self._logger.debug(
            f"Initialized LanguageTranslatorNode with default target_language='{self._default_target_language}' "
            f"and source_language='{self._default_source_language or 'auto-detect'}'"
        )

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating language translation.

        The method expects 'data' to be a string. The target language can be
        specified via the 'target_language' key in the context dictionary;
        otherwise, the default target language set during initialization is used.
        Similarly, 'source_language' can be overridden via context.

        Args:
            data: The input text data to be translated (expected to be a string).
            context: A dictionary containing additional information.
                     Can include 'target_language' and 'source_language' to
                     override the defaults for this specific invocation.

        Returns:
            A string representing the simulated translated text.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the resolved target language is invalid (e.g., not a string).
        """
        current_target_language = context.get("target_language", self._default_target_language)
        current_source_language = context.get("source_language", self._default_source_language)

        if not isinstance(current_target_language, str) or not current_target_language:
            self._logger.error(
                f"Invalid target_language '{current_target_language}' resolved from context or default. "
                "Must be a non-empty string."
            )
            raise ValueError("Resolved target_language must be a non-empty string.")

        if not isinstance(data, str):
            self._logger.error(f"Input data for LanguageTranslatorNode must be a string, but received {type(data).__name__}.")
            raise TypeError("LanguageTranslatorNode expects string data for translation.")

        self._logger.info(
            f"Attempting to translate text from '{current_source_language or 'auto-detect'}' to "
            f"'{current_target_language}' using node '{self.node_name}'."
        )

        # Simulate translation by prepending a descriptive tag.
        # In a real-world scenario, this would involve an actual call
        # to a translation API or an LLM service.
        simulated_translation = f"[Translated to {current_target_language.upper()}]: {data}"

        self._logger.debug(
            f"Successfully simulated translation of '{data[:50]}...' "
            f"to '{current_target_language}'. Result: '{simulated_translation[:50]}...'"
        )

        return simulated_translation
