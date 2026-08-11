import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node is available in the environment or path
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node that simulates language translation of input text.

    This node expects the input 'data' to be a string and the 'context' dictionary
    to contain a 'target_language' key with a string value indicating the
    language to translate to.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the language translator node."""
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Simulates language translation of the input data.

        This simulation appends a suffix to the original text indicating the
        target language. In a production environment, this would interface
        with an actual translation service or model.

        Args:
            data (Any): The text string to be translated.
            context (Dict[str, Any]): A dictionary containing processing context.
                                      Expected to have 'target_language' (str).

        Returns:
            Any: The simulated translated text string.

        Raises:
            ValueError: If 'data' is not a string, or 'target_language' is missing
                        or not a valid string in the context.
            RuntimeError: For any other unexpected errors during processing.
        """
        # Generate a unique identifier for this node instance for detailed logging
        node_instance_id = id(self)
        log_prefix = f"[{self.node_name}:{node_instance_id}]"
        logger.debug(f"{log_prefix} Starting process with data type: {type(data)}, context keys: {context.keys()}")

        if not isinstance(data, str):
            logger.error(f"{log_prefix} Invalid input data type. Expected 'str', got {type(data)}.")
            raise ValueError(
                f"{log_prefix} LanguageTranslatorNode expects 'data' to be a string, "
                f"but received {type(data)}."
            )

        if 'target_language' not in context:
            logger.error(f"{log_prefix} 'target_language' key missing in context for translation.")
            raise ValueError(
                f"{log_prefix} Context must contain a 'target_language' key for LanguageTranslatorNode."
            )

        target_language = context['target_language']
        if not isinstance(target_language, str) or not target_language.strip():
            logger.error(
                f"{log_prefix} Invalid 'target_language' in context. Expected non-empty string, "
                f"got type {type(target_language)} with value: '{target_language}'."
            )
            raise ValueError(
                f"{log_prefix} Context 'target_language' must be a non-empty string for "
                f"LanguageTranslatorNode."
            )

        try:
            # Simulate translation: In a real scenario, this would involve calling an external API
            # or a local translation model. For this simulation, we append a suffix.
            translated_text = f"{data} [TRANSLATED_TO_{target_language.strip().upper()}]"
            
            logger.info(
                f"{log_prefix} Successfully simulated translation. "
                f"Original (first 50 chars): '{data[:50]}...', "
                f"Target Language: '{target_language}', "
                f"Translated (first 50 chars): '{translated_text[:50]}...'"
            )
            return translated_text
        except Exception as e:
            logger.exception(
                f"{log_prefix} An unexpected error occurred during translation simulation."
            )
            raise RuntimeError(
                f"{log_prefix} Failed to process data in LanguageTranslatorNode due to an "
                f"unexpected error: {e}"
            ) from e