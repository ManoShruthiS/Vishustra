import logging
from typing import Any, Dict

# Assuming BaseNode is available at this path as per Vishustra's architecture
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class LanguageTranslatorNode(BaseNode):
    """
    A processing node designed for simulating language translation of text data.

    This node accepts a string as input and, guided by parameters in the context,
    produces a simulated translated string. It serves as a foundational component
    for workflows that require text localization, enabling integration with
    external translation services or basic rule-based transformations.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this processing node.
        """
        return "LanguageTranslator"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating a language translation operation.

        The `context` dictionary is crucial for defining the translation behavior
        and is expected to contain the following keys:
        - 'target_language' (str): The ISO language code (e.g., 'es', 'fr')
                                   to which the text should be translated. This is
                                   a mandatory parameter.
        - 'source_language' (str, optional): The ISO language code of the input
                                             text. Defaults to 'en' (English) if
                                             not specified.
        - 'translation_map' (Dict[str, str], optional): An optional dictionary
                                                        mapping source words to
                                                        target words. If provided,
                                                        this map is used for a simple
                                                        word-by-word simulation.
                                                        If absent, a generic
                                                        translation suffix is applied.

        Args:
            data (Any): The input data to be translated. Expected to be a string.
            context (Dict[str, Any]): A dictionary containing parameters required
                                      for the translation process.

        Returns:
            Any: The simulated translated string.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If the 'target_language' key is missing from the `context`.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Unable to translate."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for translation. "
                f"Received type '{type(data).__name__}'."
            )

        target_language = context.get("target_language")
        if not target_language:
            logger.error(
                f"[{self.node_name}] 'target_language' key is missing from the "
                f"context. Translation cannot proceed without a specified target language."
            )
            raise ValueError(
                f"[{self.node_name}] 'target_language' is a mandatory key in the "
                f"context for {self.node_name}."
            )

        source_language = context.get("source_language", "en")
        translation_map = context.get("translation_map")

        logger.info(
            f"[{self.node_name}] Initiating simulated translation from "
            f"'{source_language}' to '{target_language}'. Input data length: "
            f"{len(data)} characters."
        )

        simulated_translation = ""
        if translation_map and isinstance(translation_map, dict):
            # Simulate word-by-word translation using the provided map
            words = data.split()
            translated_words = []
            for word in words:
                # Use a case-insensitive lookup for simplicity, but append the found value or original word
                translated_word = translation_map.get(word.lower(), word)
                if translated_word == word:
                    logger.debug(
                        f"[{self.node_name}] Word '{word}' not found in the "
                        f"provided translation map. Keeping original word."
                    )
                translated_words.append(translated_word)
            simulated_translation = " ".join(translated_words)
            logger.debug(f"[{self.node_name}] Applied map-based translation strategy.")
        else:
            # Apply a generic transformation if no specific translation map is provided
            simulated_translation = f"{data} [translated_to_{target_language}]"
            logger.debug(
                f"[{self.node_name}] No 'translation_map' provided. "
                f"Applied generic suffix-based translation."
            )

        logger.info(
            f"[{self.node_name}] Successfully simulated translation to "
            f"'{target_language}'. Output length: {len(simulated_translation)} characters."
        )
        return simulated_translation