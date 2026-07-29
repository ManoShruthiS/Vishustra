import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text.

    This node takes a string as input data and returns a translated version.
    Translation parameters can be configured during initialization or overridden
    via the runtime context for specific process calls.
    """

    def __init__(self, default_source_lang: str = "en", default_target_lang: str = "es"):
        """
        Initializes the LanguageTranslatorNode with default language settings.

        Args:
            default_source_lang: The default source language code (e.g., 'en', 'fr').
                                 This can be overridden by the context during process.
            default_target_lang: The default target language code (e.g., 'es', 'de').
                                 This can be overridden by the context during process.

        Raises:
            ValueError: If default_source_lang or default_target_lang are not valid strings.
        """
        if not isinstance(default_source_lang, str) or not default_source_lang:
            raise ValueError("default_source_lang must be a non-empty string.")
        if not isinstance(default_target_lang, str) or not default_target_lang:
            raise ValueError("default_target_lang must be a non-empty string.")
        if default_source_lang == default_target_lang:
            logger.warning(
                f"LanguageTranslatorNode initialized with identical default_source_lang ('{default_source_lang}') "
                f"and default_target_lang ('{default_target_lang}'). Translation will not occur unless overridden."
            )

        self._default_source_lang = default_source_lang
        self._default_target_lang = default_target_lang
        logger.debug(
            f"LanguageTranslatorNode initialized with default_source_lang='{default_source_lang}' "
            f"and default_target_lang='{default_target_lang}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text.

        The node expects 'data' to be a string containing the text to be translated.
        It uses the initialized default source and target languages, which can be
        dynamically overridden by 'source_lang' and 'target_lang' keys found
        within the 'context' dictionary for the current execution.

        Args:
            data: The text (str) to be translated.
            context: A dictionary containing runtime information. Can include
                     'source_lang' and 'target_lang' to override node defaults.

        Returns:
            The simulated translated text (str).

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If the effective source_lang or target_lang (from init or context)
                        are not valid non-empty strings, or if they are identical.
        """
        if not isinstance(data, str):
            logger.error(f"Invalid input data type for '{self.node_name}'. Expected str, got {type(data).__name__}.")
            raise TypeError(
                f"'{self.node_name}' expects string input for translation, "
                f"but received {type(data).__name__}."
            )

        # Determine effective source and target languages for this specific process call
        current_source_lang = context.get('source_lang', self._default_source_lang)
        current_target_lang = context.get('target_lang', self._default_target_lang)

        if not isinstance(current_source_lang, str) or not current_source_lang:
            raise ValueError("Effective source_lang must be a non-empty string.")
        if not isinstance(current_target_lang, str) or not current_target_lang:
            raise ValueError("Effective target_lang must be a non-empty string.")

        if current_source_lang == current_target_lang:
            logger.info(
                f"Node '{self.node_name}' received a translation request from "
                f"'{current_source_lang}' to '{current_target_lang}'. "
                f"Languages are identical, returning original text."
            )
            return data # No translation needed if languages are the same

        # In a real-world scenario, this would involve an actual call to a translation service API
        # For simulation, we append a descriptive tag to the text.
        translated_text = (
            f"{data} [TRANSLATED from {current_source_lang.upper()} to {current_target_lang.upper()}]"
        )

        logger.info(
            f"Node '{self.node_name}' successfully translated text from "
            f"'{current_source_lang}' to '{current_target_lang}'. "
            f"Original length: {len(data)}, Translated length: {len(translated_text)}."
        )
        logger.debug(
            f"Original text snippet: '{data[:50].replace('\n', ' ')}...' -> "
            f"Translated snippet: '{translated_text[:50].replace('\n', ' ')}...'"
        )

        return translated_text