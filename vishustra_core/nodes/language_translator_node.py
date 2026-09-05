import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra node designed to simulate language translation of text.
    It expects a string as input `data` and a 'target_language' key within the `context`
    dictionary to determine the translation destination.

    This node demonstrates a data transformation step where textual content
    is processed to fulfill localization requirements.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Translates the input text based on the 'target_language' specified in the context.
        This implementation provides a simulated translation for demonstration purposes.

        Args:
            data: The text data to be translated. Expected to be a string.
            context: A dictionary containing additional runtime context, which *must*
                     include 'target_language' (e.g., 'es' for Spanish, 'fr' for French,
                     'de' for German) to specify the translation target.

        Returns:
            The translated text as a string, or the original data if translation
            is not possible due to invalid input or context, after logging an error.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is missing from the context or is invalid.
        """
        logger.debug("LanguageTranslatorNode: Starting text translation process.")

        # --- Input Validation ---
        if not isinstance(data, str):
            logger.error(
                f"LanguageTranslatorNode: Invalid input data type. Expected 'str', got '{type(data).__name__}'. "
                "Translation aborted. Raising TypeError."
            )
            raise TypeError(f"Input data for LanguageTranslatorNode must be a string, but got {type(data).__name__}.")

        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                "LanguageTranslatorNode: 'target_language' not found or is invalid in context. "
                "Context received: %s. Translation aborted. Raising ValueError.", context
            )
            raise ValueError("Missing or invalid 'target_language' in context. Expected a string language code.")

        logger.info(
            f"LanguageTranslatorNode: Attempting to translate text to '{target_language}'. "
            f"Original text snippet: '{data[:75]}...'" if len(data) > 75 else f"Original text: '{data}'"
        )

        # --- Simulated Translation Logic ---
        # In a production scenario, this section would integrate with an external
        # translation API (e.g., Google Cloud Translate, DeepL, AWS Translate, etc.).
        # For this simulation, we perform a basic word-for-word replacement
        # for a small set of common phrases, primarily for demonstration.
        translated_text = data
        
        # Highly simplified word-level translation map for demonstration
        # Note: This does not handle grammar, sentence structure, context, or morphology.
        translation_map = {
            "es": {  # Spanish
                "hello": "hola", "world": "mundo", "how are you": "¿cómo estás?",
                "thank you": "gracias", "goodbye": "adiós", "please": "por favor",
                "yes": "sí", "no": "no", "this is": "esto es"
            },
            "fr": {  # French
                "hello": "bonjour", "world": "monde", "how are you": "comment allez-vous?",
                "thank you": "merci", "goodbye": "au revoir", "please": "s'il vous plaît",
                "yes": "oui", "no": "non", "this is": "c'est"
            },
            "de": {  # German
                "hello": "hallo", "world": "welt", "how are you": "wie geht es dir?",
                "thank you": "danke", "goodbye": "auf wiedersehen", "please": "bitte",
                "yes": "ja", "no": "nein", "this is": "das ist"
            }
        }

        # Apply simulated translation if a map exists for the target language
        if target_language in translation_map:
            current_lang_map = translation_map[target_language]
            
            # Iterate through phrases from longest to shortest to handle multi-word phrases first
            # This helps prevent partial replacements (e.g., "how are you" before "how")
            sorted_phrases = sorted(current_lang_map.keys(), key=len, reverse=True)

            for original_phrase in sorted_phrases:
                translated_phrase = current_lang_map[original_phrase]
                
                # Attempt to replace while trying to preserve casing of the original phrase
                # This is a very simplistic heuristic for simulation.
                if original_phrase.capitalize() in translated_text:
                    translated_text = translated_text.replace(original_phrase.capitalize(), translated_phrase.capitalize())
                elif original_phrase.upper() in translated_text:
                    translated_text = translated_text.replace(original_phrase.upper(), translated_phrase.upper())
                elif original_phrase.lower() in translated_text:
                    translated_text = translated_text.replace(original_phrase.lower(), translated_phrase.lower())
        else:
            logger.warning(
                f"LanguageTranslatorNode: No specific translation map available for target language "
                f"'{target_language}'. The text will be returned largely untranslated, with a language tag."
            )
        
        # Append a language tag to the translated text to signify processing and target language.
        final_translated_text = f"{translated_text.strip()} [{target_language.upper()}]"

        logger.info(
            f"LanguageTranslatorNode: Successfully processed text for '{target_language}'. "
            f"Translated text snippet: '{final_translated_text[:75]}...'" if len(final_translated_text) > 75 else f"Translated text: '{final_translated_text}'"
        )
        return final_translated_text