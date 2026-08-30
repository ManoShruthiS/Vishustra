import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node is designed to act as an abstraction layer for integrating with
    external language translation services. It takes a string as input data
    and, based on the `target_language` specified in the processing context,
    returns a simulated translated string.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text data to a specified target language.

        The `data` parameter is expected to be a string representing the text
        content to be translated.
        The `context` dictionary must contain a 'target_language' key, which
        should be a string specifying the language code (e.g., 'es' for Spanish,
        'fr' for French, 'de' for German) for the translation.

        Args:
            data: The text content (string) that needs to be translated.
            context: A dictionary containing operational parameters for the node.
                     Must include 'target_language': 'str'.

        Returns:
            A string representing the simulated translated text.

        Raises:
            ValueError: If the `data` parameter is not a string, indicating
                        an invalid input type for translation.
        """
        if not isinstance(data, str):
            logger.error(
                "LanguageTranslatorNode received non-string data. Expected 'str', got '%s'. Data: %s",
                type(data).__name__,
                data
            )
            raise ValueError(
                f"LanguageTranslatorNode expects string data for translation, "
                f"but received {type(data).__name__}."
            )

        target_language = context.get("target_language")
        translated_prefix = ""

        if not target_language or not isinstance(target_language, str):
            logger.warning(
                "LanguageTranslatorNode context is missing or has an invalid "
                "'target_language'. Defaulting to a generic translation simulation. "
                "Context: %s",
                context
            )
            translated_prefix = "[Translated (unknown target)]:"
        else:
            # Simulate different language translations for demonstration
            # In a real scenario, this would involve calling a translation API
            if target_language.lower() == "es":
                translated_prefix = "[Translated to Spanish]:"
            elif target_language.lower() == "fr":
                translated_prefix = "[Translated to French]:"
            elif target_language.lower() == "de":
                translated_prefix = "[Translated to German]:"
            elif target_language.lower() == "zh":
                translated_prefix = "[Translated to Chinese]:"
            else:
                logger.info(
                    "LanguageTranslatorNode received unsupported or unknown "
                    "target language '%s'. Applying a generic translation simulation.",
                    target_language
                )
                translated_prefix = f"[Translated to {target_language.capitalize()}]:"
        
        simulated_translation = f"{translated_prefix} {data}"
        
        logger.debug(
            "LanguageTranslatorNode processed data. Original: '%s', Target Language: '%s', Simulated Translated: '%s'",
            data,
            target_language,
            simulated_translation
        )
        
        return simulated_translation