import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core.nodes.base_node exists as specified
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text.

    This node expects a string `data` to be translated and
    a `target_language` in the `context` dictionary.
    An optional `source_language` can also be provided in the context,
    defaulting to 'en' (English) if not specified.

    For demonstration purposes, translation is simulated using an internal
    mapping of common phrases. In a production system, this would integrate
    with a robust external translation API (e.g., Google Translate, DeepL).
    """

    # A simple, static translation map for simulation purposes.
    # Keys are (source_language, target_language, original_phrase_lowercase).
    # Values are the translated phrase.
    _TRANSLATION_MAP = {
        ("en", "es", "hello"): "hola",
        ("en", "fr", "hello"): "bonjour",
        ("en", "de", "hello"): "hallo",
        ("en", "es", "world"): "mundo",
        ("en", "fr", "world"): "monde",
        ("en", "de", "world"): "welt",
        ("en", "es", "how are you?"): "¿cómo estás?",
        ("en", "fr", "how are you?"): "comment allez-vous?",
        ("en", "de", "how are you?"): "wie geht es dir?",
        ("en", "es", "thank you"): "gracias",
        ("en", "fr", "thank you"): "merci",
        ("en", "de", "thank you"): "danke schön",
        ("en", "es", "goodbye"): "adiós",
        ("en", "fr", "goodbye"): "au revoir",
        ("en", "de", "goodbye"): "auf wiedersehen",
        # Example for Spanish to English
        ("es", "en", "hola"): "hello",
        ("es", "en", "gracias"): "thank you",
        # More complex phrases for robustness test
        ("en", "es", "this is a test sentence"): "esta es una frase de prueba",
        ("en", "fr", "this is a test sentence"): "c'est une phrase de test",
    }

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Translates the input text `data` to the `target_language` specified in `context`.

        Args:
            data: The text data to be translated. Expected to be a string.
            context: A dictionary containing operational parameters.
                     Must include 'target_language' (str, e.g., 'es', 'fr', 'de').
                     Can optionally include 'source_language' (str, defaults to 'en').

        Returns:
            The translated text as a string. If a translation is not found in the
            internal map, a warning is logged, and the original text is returned
            prefixed with a placeholder indicating untranslated status.

        Raises:
            TypeError: If `data` is not a string.
            ValueError: If `target_language` is missing from `context` or is not a string.
            ValueError: If `source_language` is provided but is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'."
            )
            raise TypeError(
                f"Input 'data' for {self.node_name} must be a string, "
                f"but got '{type(data).__name__}'."
            )

        target_language = context.get("target_language")
        if not isinstance(target_language, str):
            logger.error(
                f"[{self.node_name}] 'target_language' is missing or not a string in the context. "
                f"Context received: {context}"
            )
            raise ValueError(
                f"'target_language' must be provided as a string in the context for {self.node_name}."
            )

        source_language = context.get("source_language", "en") # Default to English if not specified
        if not isinstance(source_language, str):
            logger.error(
                f"[{self.node_name}] 'source_language' must be a string if provided in the context. "
                f"Context received: {context}"
            )
            raise ValueError(
                f"'source_language' must be a string if provided in the context for {self.node_name}."
            )

        logger.debug(
            f"[{self.node_name}] Attempting to translate text "
            f"from '{source_language}' to '{target_language}'. "
            f"Data snippet: '{data[:70]}{'...' if len(data) > 70 else ''}'"
        )

        # Normalize languages and data to lowercase for consistent map lookups.
        # This simplifies the mock but a real service would handle case sensitivity.
        source_language_lower = source_language.lower()
        target_language_lower = target_language.lower()
        data_lower = data.lower()

        # Attempt to retrieve a translation from the internal map
        translated_text = self._TRANSLATION_MAP.get(
            (source_language_lower, target_language_lower, data_lower)
        )

        if translated_text:
            logger.info(
                f"[{self.node_name}] Successfully translated from '{source_language}' "
                f"to '{target_language}'. Original: '{data}', Translated: '{translated_text}'"
            )
            return translated_text
        else:
            warning_msg = (
                f"[{self.node_name}] No direct translation found in the internal mock map "
                f"for the phrase ('{source_language}', '{target_language}', '{data}'). "
                "Returning original text with a placeholder indicating untranslated status. "
                "Consider expanding the mock map or integrating a real translation service."
            )
            logger.warning(warning_msg)
            # In a real-world scenario, this might trigger a fallback,
            # send to a more general translation API, or raise a specific exception.
            # For this simulation, we return a modified string.
            return f"[UNTRANSLATED:{source_language.upper()}->{target_language.upper()}]: {data}"