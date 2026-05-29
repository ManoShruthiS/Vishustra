import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to simulate language translation of text data.

    This node expects the input `data` to be a string containing the text
    to be translated. The `context` dictionary must provide a 'target_language'
    key, specifying the language code for the desired translation (e.g., 'es' for Spanish,
    'fr' for French, 'de' for German).

    The `process` method simulates the translation by appending a language tag
    to the input text. In a real-world scenario, this would involve integration
    with an external translation service API.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a specified target language.

        Args:
            data (Any): The input data, expected to be a string that needs translation.
            context (Dict[str, Any]): A dictionary providing additional processing context.
                                      Must include 'target_language' (str) with the
                                      ISO 639-1 code of the desired target language.

        Returns:
            Any: The simulated translated string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing from `context`, is not a string,
                        or is an empty string.
            RuntimeError: For any unexpected errors occurring during the simulated
                          translation process.
        """
        logger.debug(f"[{self.node_name}] Initiating process for data type: {type(data).__name__}")

        if not isinstance(data, str):
            error_msg = (f"[{self.node_name}] Invalid input data type. Expected 'str', "
                         f"but received '{type(data).__name__}'.")
            logger.error(error_msg)
            raise TypeError(error_msg)

        target_language = context.get("target_language")

        if not isinstance(target_language, str) or not target_language.strip():
            error_msg = (f"[{self.node_name}] 'target_language' is missing, not a string, or empty in context. "
                         f"Received: '{target_language}'.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        normalized_target_language = target_language.strip().upper()

        try:
            # Simulate the translation by appending a tag.
            # In a production system, this would be replaced with actual API calls
            # to services like Google Translate, DeepL, or a self-hosted solution.
            translated_text = f"{data} [Translated to {normalized_target_language}]"
            logger.info(
                f"[{self.node_name}] Successfully simulated translation to "
                f"'{normalized_target_language}'. Original length: {len(data)}, "
                f"Translated length: {len(translated_text)}."
            )
            return translated_text
        except Exception as e:
            # Catching general exceptions here to encapsulate any unexpected issues
            # from the simulation logic, though for simple string ops, this is mostly
            # a safeguard for future complexity.
            error_msg = (f"[{self.node_name}] An unexpected error occurred during "
                         f"translation simulation: {e}")
            logger.exception(error_msg) # Logs the exception with traceback
            raise RuntimeError(error_msg) from e
