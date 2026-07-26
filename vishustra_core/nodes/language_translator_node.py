import logging
from typing import Any, Dict, Optional

# Assuming the base_node is located at vishustra_core.nodes.base_node as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to simulate language translation of text data.

    This node expects a string as input and simulates its translation
    to a specified target language. The target language can be configured
    during the node's initialization, or dynamically overridden by providing
    a 'target_language' key in the process context.

    Raises:
        ValueError: If the input data is not a string or if no target language
                    is specified for translation.
    """

    def __init__(self, default_target_language: str = "en"):
        """
        Initializes the LanguageTranslatorNode with a default target language.

        Args:
            default_target_language: The default language code (e.g., 'en', 'es', 'fr')
                                     to translate to if no 'target_language' is found
                                     in the process context. Defaults to 'en'.
        """
        if not isinstance(default_target_language, str) or not default_target_language:
            raise ValueError("Default target language must be a non-empty string.")
        
        self._default_target_language = default_target_language.lower()
        logger.debug(
            f"[{self.node_name}] Initialized with default target language: "
            f"'{self._default_target_language}'"
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating its translation to a target language.

        The target language is determined first by the 'target_language' key in
        the `context` dictionary. If not present, it falls back to the
        `default_target_language` set during initialization.

        Args:
            data: The input data to be translated. Expected to be a string.
            context: A dictionary containing contextual information. This can
                     include a 'target_language' key (str) to specify the
                     translation target for this specific process invocation.

        Returns:
            A string representing the simulated translated text.

        Raises:
            ValueError: If `data` is not a string, or if no target language
                        (neither in context nor default) is available.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type for translation. "
                f"Expected 'str', received '{type(data).__name__}'."
            )
            raise ValueError(
                f"[{self.node_name}] Input data must be a string for translation. "
                f"Received type: {type(data).__name__}"
            )

        # Determine the target language for this specific process call
        target_language = context.get("target_language", self._default_target_language)

        if not target_language:
            logger.error(
                f"[{self.node_name}] No target language specified. "
                "Translation cannot proceed without a target language."
            )
            raise ValueError(
                f"[{self.node_name}] Target language must be specified either "
                "during initialization or in the process context."
            )
        
        # Simulate the translation by prepending a tag.
        # In a real scenario, this would involve calling an external translation service.
        translated_text = f"[Translated to {target_language.upper()}]: {data}"
        
        logger.info(
            f"[{self.node_name}] Successfully simulated translation of text "
            f"(first 30 chars: '{data[:30]}...') to '{target_language}'."
        )
        
        return translated_text

