import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of text data.

    This node expects the input `data` to be a string and requires
    'target_language' to be specified in the `context` dictionary.
    Optionally, 'source_language' can also be provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "LanguageTranslatorNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates the translation of input text data.

        This method expects `data` to be a string that requires translation.
        The `context` dictionary must contain a 'target_language' string
        indicating the desired output language. An optional 'source_language'
        string can also be provided in the `context`.

        In a production system, this method would interface with a dedicated
        language translation service. For demonstration, it prefixes the
        input text with a simulated translation marker.

        Args:
            data (Any): The input data to be translated, expected as a string.
            context (Dict[str, Any]): A dictionary containing parameters for
                                      translation:
                                      - 'target_language' (str): The language
                                        to translate the text into (e.g., "en", "fr").
                                      - 'source_language' (Optional[str]): The
                                        original language of the text (e.g., "es", "de").
                                        Defaults to 'auto-detect' if not provided.

        Returns:
            Any: A string representing the simulated translated text.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_language' is not provided in the context,
                        or if it is not a string.
        """
        if not isinstance(data, str):
            logger.error(
                "[%s] Invalid input data type. Expected string for translation, received %s.",
                self.node_name,
                type(data).__name__,
            )
            raise TypeError(
                f"[{self.node_name}] Data for translation must be a string, "
                f"but received {type(data).__name__}."
            )

        target_language = context.get("target_language")
        if not target_language or not isinstance(target_language, str):
            logger.error(
                "[%s] Missing or invalid 'target_language' in context. Expected a string.",
                self.node_name,
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' must be provided as a string "
                f"in the context for translation. Received: {target_language!r}"
            )

        source_language = context.get("source_language", "auto-detect")
        if not isinstance(source_language, str):
            logger.warning(
                "[%s] 'source_language' in context is not a string. Using default 'auto-detect'. "
                "Context: %s",
                self.node_name,
                context,
            )
            source_language = "auto-detect"  # Ensure it's a string for consistent logging

        # Simulate translation - in a real-world scenario, this would involve
        # an API call to a translation service (e.g., Google Translate, DeepL).
        translated_text = (
            f"[Simulated Translation from {source_language.upper()} "
            f"to {target_language.upper()}]: {data}"
        )

        logger.info(
            "[%s] Successfully simulated translation from '%s' to '%s' for text of length %d.",
            self.node_name,
            source_language,
            target_language,
            len(data),
        )
        return translated_text