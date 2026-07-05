import logging
from typing import Any, Dict, Optional

# Assuming BaseNode is located here as per project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node designed to simulate language translation of text data.

    This node expects the input `data` to be a string. The `context` dictionary
    is required to contain a 'target_language' key with the desired language code
    (e.g., 'es' for Spanish, 'fr' for French). An optional 'source_language'
    key can also be provided; otherwise, English ('en') is assumed as the source.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text data to a specified target language based on context.

        Args:
            data: The input text string intended for translation.
            context: A dictionary containing operational parameters for the node.
                     Expected keys:
                     - 'target_language' (str): The ISO 639-1 code for the target language.
                     - 'source_language' (str, optional): The ISO 639-1 code for the source language.
                                                          Defaults to 'en' if not provided.

        Returns:
            The translated text string if successful. If translation fails for
            non-critical reasons (e.g., unsupported language pair in simulation),
            the original data is returned after logging a warning.

        Raises:
            ValueError: If `data` is not a string, or if 'target_language' is
                        missing or invalid in the `context`.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected string, but received %s. Data: %s",
                self.node_name,
                type(data).__name__,
                data,
            )
            raise ValueError(f"Input data for {self.node_name} must be a string.")

        target_language: Optional[str] = context.get("target_language")
        source_language: str = context.get("source_language", "en") # Default to English for simulation

        if not target_language or not isinstance(target_language, str):
            logger.error(
                "[%s] 'target_language' is missing or not a valid string in the context. Context: %s",
                self.node_name,
                context,
            )
            raise ValueError(
                f"Context for {self.node_name} must contain a valid 'target_language' string."
            )

        # --- Simulate Translation Logic ---
        # In a production system, this section would typically involve an
        # asynchronous call to an external Language Model (LLM) service,
        # a dedicated translation API (e.g., DeepL, Google Cloud Translate),
        # or an integrated in-process translation model.
        # For demonstration purposes within Vishustra, we use a simple
        # hardcoded dictionary mapping.

        simulated_translations = {
            ("Hello world", "en", "es"): "Hola mundo",
            ("Hello world", "en", "fr"): "Bonjour le monde",
            ("How are you?", "en", "es"): "¿Cómo estás?",
            ("How are you?", "en", "fr"): "Comment allez-vous?",
            ("Goodbye", "en", "es"): "Adiós",
            ("Goodbye", "en", "fr"): "Au revoir",
            # Add more simulated translations as needed for testing
        }

        # Create a lookup key based on the normalized input data, source, and target languages.
        # Normalization (e.g., lowercasing, stripping whitespace) can be important
        # for robust lookup and API calls.
        lookup_key = (data.strip(), source_language.lower(), target_language.lower())

        translated_text = simulated_translations.get(lookup_key)

        if translated_text is None:
            logger.warning(
                "[%s] No simulated translation found for text '%s' from '%s' to '%s'. "
                "Returning original data.",
                self.node_name,
                data,
                source_language,
                target_language,
            )
            # In a real scenario, depending on framework policy, this might:
            # - Trigger a fallback translation service.
            # - Raise a specific TranslationError.
            # - Return the original text (as done here for robustness).
            return data
        else:
            logger.info(
                "[%s] Successfully translated text from '%s' to '%s'. Original: '%s', Translated: '%s'",
                self.node_name,
                source_language,
                target_language,
                data,
                translated_text,
            )
            return translated_text