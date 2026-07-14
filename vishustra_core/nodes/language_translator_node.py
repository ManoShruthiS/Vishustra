import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node designed to perform language translation of text data.

    This node takes an input string and simulates its translation into a specified
    target language. In a production environment, this would integrate with a
    third-party translation service or a local translation model.
    """

    def __init__(self, target_language: str):
        """
        Initializes the LanguageTranslatorNode with the desired target language.

        Args:
            target_language (str): The IETF language tag (e.g., 'es', 'fr', 'de')
                                   or common name (e.g., 'Spanish', 'French') for
                                   the language to translate into. This parameter
                                   is mandatory and must be a non-empty string.

        Raises:
            ValueError: If `target_language` is not a non-empty string.
        """
        if not isinstance(target_language, str) or not target_language.strip():
            logger.error(f"Initialization failed: target_language must be a non-empty string. Got: {target_language!r}")
            raise ValueError("target_language must be a non-empty string.")

        self._target_language = target_language.strip()
        logger.info(f"LanguageTranslatorNode initialized with target language: '{self._target_language}'")

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating a translation operation.

        If the input `data` is a string, it appends a translation indicator.
        Non-string data types are logged as warnings and returned unchanged,
        as translation is typically a string-specific operation.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        for successful 'translation'.
            context (Dict[str, Any]): A dictionary containing additional runtime context.
                                      This might include metadata like 'source_language'
                                      for more advanced translation services.

        Returns:
            Any: The 'translated' string or the original data if it was not a string
                 or an empty string.
        """
        logger.debug(f"Node '{self.node_name}' starting processing for data type: '{type(data).__name__}'.")

        if not isinstance(data, str):
            logger.warning(
                f"Node '{self.node_name}' received non-string data of type '{type(data).__name__}'. "
                "Translation is primarily applicable to strings. Returning original data without modification."
            )
            return data

        if not data.strip():
            logger.debug(f"Node '{self.node_name}' received an empty or whitespace-only string. Returning as-is.")
            return data

        # Simulate the translation process. In a real scenario, this would involve
        # an API call to a translation service (e.g., Google Translate, DeepL)
        # or an inference call to a local model.
        source_language_hint = context.get("source_language", "auto-detected")
        
        # A simple placeholder for actual translation logic
        translated_data = (
            f"{data} [translated to {self._target_language} from {source_language_hint}]"
        )

        logger.info(
            f"Node '{self.node_name}' successfully 'translated' text "
            f"from '{source_language_hint}' to '{self._target_language}'. "
            f"Original (first 50 chars): '{data[:50]}...', "
            f"Result (first 50 chars): '{translated_data[:50]}...'"
        )
        return translated_data
