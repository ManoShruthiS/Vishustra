import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation.

    This node takes text data and simulates its translation into a specified
    target language. It requires 'target_language' in the context dictionary.
    Optionally, 'source_language' can also be provided in the context to
    indicate the original language of the data.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text data based on parameters in the context.

        Args:
            data: The text data to be translated (expected to be a string).
            context: A dictionary containing operational parameters for translation.
                     It *must* contain 'target_language' (str) indicating the desired
                     output language. It *may* contain 'source_language' (str)
                     indicating the input language, defaulting to 'auto-detected'
                     if not provided or invalid.

        Returns:
            A string representing the simulated translated text. If the input data
            is empty or whitespace, an empty string will be returned after
            logging the operation.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing from the `context` or is not a string.
            RuntimeError: For unexpected issues encountered during the simulated translation process.
        """
        logger.debug(f"[{self.node_name}] Starting process for data (first 50 chars): '{str(data)[:50]}'")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"but received {type(data).__name__}."
            )
            raise TypeError(
                f"Input data for {self.node_name} must be a string, "
                f"but received {type(data).__name__}."
            )

        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] 'target_language' not found or is invalid in context. "
                f"Context received: {context}."
            )
            raise ValueError(
                f"Context for {self.node_name} must contain a valid 'target_language' (string). "
                f"Received: '{target_language}'"
            )

        source_language = context.get("source_language", "auto-detected")
        if not isinstance(source_language, str):
            logger.warning(
                f"[{self.node_name}] 'source_language' provided in context is not a string "
                f"('{type(source_language).__name__}'). Defaulting to 'auto-detected'."
            )
            source_language = "auto-detected"

        translated_text = ""
        try:
            if not data.strip():
                logger.info(
                    f"[{self.node_name}] Input data is empty or consists only of whitespace. "
                    f"Returning an empty string for the 'translation'."
                )
                translated_text = "" # Explicitly returning empty for empty input
            else:
                # Simulate translation by constructing a descriptive string.
                # In a real scenario, this would involve calling a translation API.
                translated_text = (
                    f"TRANSLATED from {source_language.upper()} "
                    f"to {target_language.upper()}: '{data}'"
                )
                logger.info(
                    f"[{self.node_name}] Successfully simulated translation of "
                    f"'{data[:50]}...' from '{source_language}' to '{target_language}'."
                )
        except Exception as e:
            logger.exception(
                f"[{self.node_name}] An unexpected error occurred during the "
                f"translation simulation for data: '{data[:50]}...'."
            )
            raise RuntimeError(
                f"Failed to simulate translation in {self.node_name} due to an internal error."
            ) from e

        logger.debug(
            f"[{self.node_name}] Processed data successfully. Returning "
            f"translated text (first 50 chars): '{translated_text[:50]}'."
        )
        return translated_text