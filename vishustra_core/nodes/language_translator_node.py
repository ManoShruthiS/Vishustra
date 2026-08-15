import logging
from typing import Any, Dict, List, Union

# Assuming vishustra_core.nodes.base_node exists in the project structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node designed to translate text content.

    This node operates on various data structures:
    - If the input `data` is a string, it translates the string directly.
    - If `data` is a dictionary, it identifies common text-bearing keys
      ('text', 'message', 'content') and translates their string values.
      Nested dictionaries and lists within these keys are also handled.
    - If `data` is a list, it iterates through its elements, attempting to
      translate each string or dictionary item found.
    - For other data types, a debug message is logged, and the original
      data is returned without modification.

    Translation requires a 'target_language' (e.g., 'es', 'fr', 'de') to
    be present as a string in the `context` dictionary. The actual translation
    logic is simulated within this node for demonstration purposes, and would
    typically interface with an external translation service in a production environment.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of this processing node."""
        return "LanguageTranslator"

    def _simulate_translation(self, text_content: Union[str, Any], target_language: str) -> Any:
        """
        Simulates text translation. In a real-world scenario, this method would
        dispatch a call to an external translation API (e.g., Google Translate, DeepL).
        For this simulation, it appends a language tag to the text.

        Args:
            text_content: The content to be translated. Expected to be a string.
            target_language: The ISO code of the target language (e.g., 'es', 'fr').

        Returns:
            The simulated translated string, or the original content if it's not a string
            and cannot be converted.
        """
        if not isinstance(text_content, str):
            logger.debug(
                f"Attempted to translate non-string content of type {type(text_content).__name__}. "
                f"Attempting conversion to string for content: {text_content!r}"
            )
            try:
                text_content = str(text_content)
            except Exception as e:
                logger.warning(
                    f"Failed to convert non-string content {text_content!r} to string for translation. "
                    f"Error: {e}. Returning original content."
                )
                return text_content # Return original if conversion fails

        # Simple simulation: just appends a language tag.
        # This is where an actual API call to a translation service would be made.
        return f"{text_content} [TRANSLATED_TO_{target_language.upper()}]"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by translating text content based on the
        'target_language' specified in the context.

        Args:
            data: The input data to be processed. Can be a string, dict, or list.
            context: A dictionary containing contextual information, expected to
                     include 'target_language' (e.g., 'es', 'fr', 'de').

        Returns:
            The translated data, maintaining the original structure where possible.

        Raises:
            ValueError: If 'target_language' is missing from the context or is not a string.
            Exception: Propagates any unexpected errors encountered during processing.
        """
        target_language = context.get("target_language")

        if not isinstance(target_language, str) or not target_language:
            logger.error(
                "LanguageTranslatorNode requires 'target_language' (a non-empty string) "
                f"in context. Received: {target_language!r}. Data will not be translated."
            )
            raise ValueError(
                "Missing or invalid 'target_language' in context for LanguageTranslatorNode. "
                "Expected a non-empty string."
            )

        def _recursive_transform(item: Any) -> Any:
            """Recursively translates text within nested structures."""
            if isinstance(item, str):
                return self._simulate_translation(item, target_language)
            elif isinstance(item, dict):
                translated_dict = item.copy()
                # Common keys that might hold translatable text
                text_keys = ["text", "message", "content", "description", "title"]
                for key, value in item.items():
                    if key in text_keys and isinstance(value, str):
                        translated_dict[key] = self._simulate_translation(value, target_language)
                    elif isinstance(value, (dict, list)):
                        # Recurse into nested dictionaries or lists
                        translated_dict[key] = _recursive_transform(value)
                return translated_dict
            elif isinstance(item, list):
                # Recurse into list elements
                return [_recursive_transform(element) for element in item]
            else:
                logger.debug(
                    f"LanguageTranslatorNode encountered non-translatable data type "
                    f"{type(item).__name__} during recursion. Returning item as is: {item!r}."
                )
                return item

        try:
            translated_data = _recursive_transform(data)
            logger.info(
                f"Successfully processed data for translation to '{target_language}'."
            )
            return translated_data
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred during translation in LanguageTranslatorNode "
                f"for input data: {data!r}. Error: {e}"
            )
            # Re-raise the exception to indicate a processing failure,
            # allowing the orchestration framework to handle it.
            raise
