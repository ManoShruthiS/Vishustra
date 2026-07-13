import logging
from typing import Any, Dict

# Assuming this path from the problem description
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation.

    This node takes a string as input and "translates" it to a specified
    target language. The target language can be set during node instantiation
    or overridden via the processing context.
    """

    def __init__(self, default_target_language: str = "en"):
        """
        Initializes the LanguageTranslatorNode.

        Args:
            default_target_language (str): The default language to "translate" to
                                           if not specified in the context.
                                           Expected to be a valid language code (e.g., "en", "es", "fr").
        """
        if not isinstance(default_target_language, str) or not default_target_language.strip():
            logger.error(f"Invalid default_target_language '{default_target_language}' provided during initialization.")
            raise ValueError("default_target_language must be a non-empty string.")

        self._default_target_language = default_target_language.lower()
        logger.debug(f"LanguageTranslatorNode initialized with default target language: '{self._default_target_language}'")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "language_translator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating translation to a target language.

        The target language is determined in the following order:
        1. 'target_language' key in the provided `context` dictionary.
        2. The `default_target_language` provided during node initialization.

        Args:
            data (Any): The input data to be translated. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing additional
                                     information for processing.
                                     Can include 'target_language' (str) to
                                     override the default target language for this specific
                                     processing run.

        Returns:
            Any: The "translated" string.

        Raises:
            ValueError: If the input data is not a string, or if no valid
                        target language can be determined.
            RuntimeError: For unexpected issues during the simulated translation.
        """
        if not isinstance(data, str):
            logger.error(f"Node '{self.node_name}' received non-string data (type: {type(data)}). Expected string for translation.")
            raise ValueError(f"LanguageTranslatorNode requires string input for translation, but received {type(data)}.")

        if not data.strip():
            logger.warning(f"Node '{self.node_name}' received an empty or whitespace-only string for translation. Returning original data.")
            return data

        # Determine the target language, prioritizing context over instance default
        target_language_from_context = context.get("target_language")
        current_target_language = (
            str(target_language_from_context).lower()
            if isinstance(target_language_from_context, str) and target_language_from_context.strip()
            else self._default_target_language
        )

        if not current_target_language:
            logger.error(f"Node '{self.node_name}' could not determine a valid target language from context or default settings.")
            raise ValueError("Target language must be specified either in node configuration or context.")

        try:
            # Simulate translation by prepending a tag
            translated_text = f"[Translated to {current_target_language}]: {data}"
            logger.info(f"Node '{self.node_name}' successfully 'translated' data to '{current_target_language}'. "
                        f"Original content snippet: '{data[:50]}'")
            return translated_text
        except Exception as e:
            logger.exception(f"Node '{self.node_name}' encountered an unexpected error during 'translation' simulation.")
            raise RuntimeError(f"Translation simulation failed in '{self.node_name}': {e}") from e