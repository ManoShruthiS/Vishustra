import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core is a package and nodes is a subpackage,
# and base_node.py contains the BaseNode class.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class TranslationError(Exception):
    """Custom exception raised when translation fails."""
    pass

class LanguageTranslator(BaseNode):
    """
    A Vishustra processing node designed to translate textual data
    from a source language to a specified target language.

    This node simulates interaction with an external translation service
    and includes robust error handling for various failure scenarios.
    """

    def __init__(self, target_lang: str, source_lang: Optional[str] = None):
        """
        Initializes the LanguageTranslator node with default language settings.

        Args:
            target_lang: The ISO 639-1 code for the language to translate to (e.g., 'en', 'fr', 'es').
                         This is a mandatory parameter.
            source_lang: The ISO 639-1 code for the language of the input text (e.g., 'en').
                         If None, the translation service is expected to auto-detect the source language.

        Raises:
            ValueError: If `target_lang` is not a valid non-empty string,
                        or if `source_lang` is provided but invalid.
        """
        if not isinstance(target_lang, str) or not target_lang.strip():
            raise ValueError("Target language must be a non-empty string.")
        if source_lang is not None and (not isinstance(source_lang, str) or not source_lang.strip()):
            raise ValueError("Source language must be a non-empty string if provided, or None for auto-detection.")

        self._target_lang: str = target_lang.strip().lower()
        self._source_lang: Optional[str] = source_lang.strip().lower() if source_lang else None

        logger.info(
            f"LanguageTranslator node initialized. "
            f"Default target language: '{self._target_lang}'. "
            f"Default source language: '{self._source_lang if self._source_lang else 'auto-detect'}'."
        )

        # In a production environment, this would typically involve initializing
        # a client for an actual translation service (e.g., Google Cloud Translation, DeepL).
        # For this implementation, we will simulate the translation logic.
        self._translation_service_client = None # Placeholder for a real client instance

    @property
    def node_name(self) -> str:
        """
        Returns the unique and descriptive name of this processing node.
        """
        return "LanguageTranslator"

    def _simulate_translation_service(self, text: str, source_lang: Optional[str], target_lang: str) -> str:
        """
        Simulates an external translation service call.

        This method is a placeholder for actual API integration. In a real application,
        it would make network requests, handle authentication, retries, and parse responses
        from a service like Google Translate, DeepL, etc.

        Args:
            text: The text string to translate.
            source_lang: The detected or specified source language code.
            target_lang: The target language code.

        Returns:
            The simulated translated text.

        Raises:
            TranslationError: If the simulation fails to provide a translation
                              for the given input.
        """
        logger.debug(
            f"Simulating translation for text (first 50 chars: '{text[:50]}...') "
            f"from {source_lang if source_lang else 'auto-detect'} to {target_lang}."
        )

        # --- Simplistic hardcoded simulation ---
        # This part would be replaced by actual API calls in a real scenario.
        # Adding a few specific examples for different language pairs.

        # Example 1: French to English
        if source_lang == 'fr' and target_lang == 'en':
            if text.lower() == "bonjour le monde":
                return "Hello world"
            elif text.lower() == "comment allez-vous ?":
                return "How are you?"
        
        # Example 2: English to Spanish
        elif source_lang == 'en' and target_lang == 'es':
            if text.lower() == "hello world":
                return "Hola mundo"
            elif text.lower() == "how are you?":
                return "¿Cómo estás?"

        # Example 3: Auto-detect (assume French) to English
        elif source_lang is None and target_lang == 'en':
            if text.lower() == "merci beaucoup":
                return "Thank you very much"
            elif text.lower() == "au revoir":
                return "Goodbye"

        # If no specific simulation matches, raise an error to mimic service limitations
        logger.warning(
            f"Simulated translation failed: No specific rule for text '{text[:50]}...' "
            f"from {source_lang if source_lang else 'auto-detect'} to {target_lang}. "
            "Raising TranslationError."
        )
        raise TranslationError(f"Simulated translation could not process: '{text[:50]}...' "
                               f"({source_lang if source_lang else 'auto-detect'} -> {target_lang})")

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by translating the text.

        The `data` input is expected to be a string containing the text to be translated.
        The `context` dictionary can override the node's default `target_lang`
        and `source_lang` for a specific processing call.

        Args:
            data: The input text (expected to be `str`) for translation.
            context: A dictionary containing runtime context variables.
                     Optional keys:
                       - 'target_lang' (str): Overrides the node's default target language.
                       - 'source_lang' (str): Overrides the node's default source language.

        Returns:
            The translated text as a `str`.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the input `data` is an empty string, or if an
                        effective language (from config or context) is invalid.
            TranslationError: If the underlying translation service (simulated here)
                              fails to translate the text.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for '{self.node_name}'. "
                f"Expected 'str', but received '{type(data).__name__}'."
            )
            raise TypeError(f"'{self.node_name}' requires string input, but received {type(data).__name__}.")

        if not data.strip():
            logger.warning(f"Received empty or whitespace-only string for translation in '{self.node_name}'.")
            raise ValueError("Input text for translation cannot be empty or solely whitespace.")

        # Determine the effective target and source languages for this specific process call.
        # Context overrides node defaults.
        effective_target_lang = context.get('target_lang', self._target_lang)
        effective_source_lang = context.get('source_lang', self._source_lang)
        
        # Validate effective languages
        if not isinstance(effective_target_lang, str) or not effective_target_lang.strip():
            logger.error(
                f"Effective target language for '{self.node_name}' is invalid: "
                f"'{effective_target_lang}' (type: {type(effective_target_lang).__name__})."
            )
            raise ValueError("Effective target language derived from context or node configuration is invalid.")
        effective_target_lang = effective_target_lang.strip().lower()

        if effective_source_lang is not None and (not isinstance(effective_source_lang, str) or not effective_source_lang.strip()):
            logger.error(
                f"Effective source language for '{self.node_name}' is invalid: "
                f"'{effective_source_lang}' (type: {type(effective_source_lang).__name__})."
            )
            raise ValueError("Effective source language derived from context or node configuration is invalid.")
        effective_source_lang = effective_source_lang.strip().lower() if effective_source_lang else None

        logger.info(
            f"Attempting to translate data (length={len(data)}) "
            f"from {effective_source_lang if effective_source_lang else 'auto-detect'} "
            f"to {effective_target_lang} using '{self.node_name}'."
        )

        try:
            # In a real scenario, this would be:
            # translated_text = self._translation_service_client.translate(
            #     text=data,
            #     source_language=effective_source_lang,
            #     target_language=effective_target_lang
            # )
            translated_text = self._simulate_translation_service(
                text=data,
                source_lang=effective_source_lang,
                target_lang=effective_target_lang
            )
            logger.info(
                f"Successfully translated text using '{self.node_name}' "
                f"(original length={len(data)}, translated length={len(translated_text)})."
            )
            return translated_text
        except TranslationError as e:
            logger.error(
                f"Translation failed for data (len={len(data)}) using '{self.node_name}': {e}"
            )
            raise # Re-raise the specific translation error
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during translation in '{self.node_name}' "
                f"for data (len={len(data)})."
            )
            raise TranslationError(
                f"An unforeseen issue prevented translation for data (len={len(data)}): {e}"
            ) from e
