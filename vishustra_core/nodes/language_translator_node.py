import logging
from typing import Any, Dict

# Assuming BaseNode is located within vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A Vishustra processing node responsible for simulating language translation.

    This node takes a string as input data and requires a 'target_language'
    key in the context dictionary to perform its simulated translation.
    In a production environment, this would integrate with a real translation service.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the unique name of this node, which is 'LanguageTranslator'.
        """
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating a translation into the specified
        target language.

        This method expects `data` to be a string that needs translation.
        The `context` dictionary must contain a 'target_language' key,
        specifying the language code (e.g., 'es', 'fr', 'de') for translation.

        Args:
            data (Any): The input data to be translated. Expected to be a string.
            context (Dict[str, Any]): A dictionary providing contextual information.
                                       Must contain 'target_language': str.

        Returns:
            Any: A string representing the simulated translated text.

        Raises:
            ValueError: If `data` is not a string, or if 'target_language'
                        is missing from the `context`.
            TypeError: If the 'target_language' value in `context` is not a string.
        """
        if not isinstance(data, str):
            error_message = (
                f"[{self.node_name}] Invalid input data type. "
                f"Expected string, but received {type(data).__name__}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        target_language = context.get("target_language")

        if target_language is None:
            error_message = (
                f"[{self.node_name}] 'target_language' key is missing from the context. "
                "Unable to perform translation without a target language."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        
        if not isinstance(target_language, str):
            error_message = (
                f"[{self.node_name}] Invalid type for 'target_language' in context. "
                f"Expected string, but received {type(target_language).__name__}."
            )
            logger.error(error_message)
            raise TypeError(error_message)

        # In a real-world scenario, this would involve calling a sophisticated
        # translation service or an internal language model.
        # For this simulation, we append a tag indicating translation.
        translated_text = f"{data} [translated to {target_language}]"

        logger.debug(
            f"[{self.node_name}] Successfully simulated translation. "
            f"Original (first 50 chars): '{data[:50]}...', "
            f"Target Language: '{target_language}', "
            f"Translated (first 50 chars): '{translated_text[:50]}...'"
        )

        return translated_text