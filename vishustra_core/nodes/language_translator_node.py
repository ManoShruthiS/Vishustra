import logging
from typing import Any, Dict

# Assuming BaseNode is available in the specified path within Vishustra project structure
from vishustra_core.nodes.base_node import BaseNode

# Initialize logger for this module
logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node designed to translate text from a source language
    to a specified target language.

    This node encapsulates the logic for interacting with a (simulated) translation
    service, providing robust input validation and error handling for various scenarios.
    """

    def __init__(self, target_language: str, default_source_language: str = "en"):
        """
        Initializes the LanguageTranslatorNode with the desired target language
        and an optional default source language.

        Args:
            target_language (str): The language code (e.g., 'es', 'fr', 'de')
                                   to which the input data should be translated. This
                                   parameter is mandatory and must be a non-empty string.
            default_source_language (str): The default language code of the
                                           input data if not explicitly specified in the
                                           processing context. Defaults to 'en'.
        Raises:
            ValueError: If `target_language` or `default_source_language` is
                        not a non-empty string.
        """
        if not isinstance(target_language, str) or not target_language.strip():
            raise ValueError("target_language must be a non-empty string.")
        if not isinstance(default_source_language, str) or not default_source_language.strip():
            raise ValueError("default_source_language must be a non-empty string.")

        self._target_language = target_language.lower()
        self._default_source_language = default_source_language.lower()
        logger.debug(
            f"Initialized LanguageTranslatorNode with target_language='{self._target_language}' "
            f"and default_source_language='{self._default_source_language}'."
        )

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by attempting to translate it to the
        configured target language.

        The source language for translation can be explicitly provided in the `context`
        dictionary under the key 'source_language'. If 'source_language' is not present
        or is empty, the `default_source_language` configured during initialization
        will be used.

        Args:
            data (Any): The input data to be translated. This node expects a string.
            context (Dict[str, Any]): A dictionary containing contextual information
                                       relevant to the processing. May include 'source_language'.

        Returns:
            Any: The translated string. The return type is `Any` to conform to `BaseNode`
                 signature, but typically a `str` is expected.

        Raises:
            TypeError: If the input `data` is not a string, as this node is designed
                       to handle text translation.
            RuntimeError: If a simulated translation service error occurs, indicating
                          a critical issue during the translation attempt.
        """
        if not isinstance(data, str):
            logger.error(
                f"Invalid input data type for LanguageTranslatorNode. "
                f"Expected str, got {type(data).__name__}."
            )
            raise TypeError(
                f"LanguageTranslatorNode expects string data, but received {type(data).__name__}."
            )

        # Determine the source language, prioritizing context over default
        source_language = context.get("source_language", self._default_source_language).lower()
        if not source_language:
            source_language = self._default_source_language
            logger.debug(
                f"No 'source_language' provided in context or it was empty. "
                f"Falling back to default: '{source_language}'."
            )
        
        logger.info(
            f"Attempting to translate data from '{source_language}' to '{self._target_language}'."
        )
        # Log a snippet of the data for debugging, to avoid logging very large texts
        logger.debug(f"Input data snippet: '{data[:100]}{'...' if len(data) > 100 else ''}'")

        # --- Simulate an external translation service call ---
        # In a production Vishustra environment, this section would typically involve:
        # 1. Instantiating or retrieving a client for a specific translation API
        #    (e.g., Google Cloud Translate, DeepL, Azure Cognitive Services).
        # 2. Making an authenticated API call with the input `data`, `source_language`,
        #    and `target_language`.
        # 3. Handling potential network errors, API rate limits, or service-specific
        #    translation failures.
        #
        # For this demonstration, we use a simple hardcoded dictionary to simulate
        # translation outcomes.
        
        simulated_translations = {
            # Format: (input_text, source_lang, target_lang): translated_text
            ("hello", "en", "es"): "hola",
            ("world", "en", "es"): "mundo",
            ("hello world", "en", "es"): "hola mundo",
            ("goodbye", "en", "fr"): "au revoir",
            ("merci", "fr", "en"): "thank you",
            ("thank you", "en", "de"): "danke schön",
            ("good morning", "en", "ja"): "ohayō gozaimasu",
            ("error case", "en", "es"): "[TRANSLATION_SERVICE_ERROR]", # Simulating a service-side error
            ("", "en", "es"): "", # Graceful handling of empty string
        }

        # Normalize input for simulation lookup key (case-insensitive and trimmed)
        normalized_data_key = data.lower().strip()
        
        # Attempt to find a direct simulated translation
        translated_text = simulated_translations.get(
            (normalized_data_key, source_language, self._target_language),
            None
        )

        # Handle explicit simulated service errors
        if translated_text == "[TRANSLATION_SERVICE_ERROR]":
            logger.error(
                f"Simulated translation service failure for input: '{data[:50]}...'. "
                "This indicates an unrecoverable error during the translation attempt."
            )
            # In a robust system, a more specific custom exception (e.g., TranslationServiceError)
            # would be raised here, possibly wrapping the original service error.
            raise RuntimeError(f"Translation service reported an unrecoverable error for data: '{data}'")
        
        # If no specific simulated translation is found, generate a placeholder
        if translated_text is None:
            translated_text = (
                f"[SIMULATED TRANSLATION from {source_language.upper()} "
                f"to {self._target_language.upper()}: {data}]"
            )
            logger.warning(
                f"No specific simulated translation found for '{data[:50]}...'. "
                f"Returning a placeholder translation. Integration with a real "
                f"translation service would provide actual results here."
            )
        else:
            logger.debug(
                f"Successfully simulated translation for '{data[:50]}...'. "
                f"Result: '{translated_text[:50]}...'"
            )

        # Augment the context with details about the translation for downstream nodes
        # or for audit/logging purposes in the orchestration layer.
        context["vishustra_translation_details"] = {
            "source_language": source_language,
            "target_language": self._target_language,
            "was_simulated_fallback": (translated_text.startswith("[SIMULATED TRANSLATION")),
            "original_data_length": len(data),
            "translated_data_length": len(translated_text),
        }
        
        logger.info(
            f"Language translation completed. "
            f"Result snippet: '{translated_text[:100]}{'...' if len(translated_text) > 100 else ''}'."
        )
        return translated_text