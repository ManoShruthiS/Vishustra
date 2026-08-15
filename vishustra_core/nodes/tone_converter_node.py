import logging
from typing import Any, Dict

# Assuming BaseNode is located in vishustra_core.nodes.base_node
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to convert the tone of input text.
    It expects a string as input data and a 'target_tone' in the context
    to simulate the desired transformation.
    """

    def __init__(self):
        """
        Initializes the ToneConverterNode. No specific parameters are required
        for instantiation beyond the base node's capabilities.
        """
        super().__init__()
        logger.debug("ToneConverterNode initialized.")

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by converting its tone based on the
        'target_tone' specified in the context.

        Args:
            data (Any): The input data, expected to be a string representing the text.
            context (Dict[str, Any]): A dictionary containing contextual information,
                                      including the required 'target_tone' string.

        Returns:
            Any: The transformed string with the new tone.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_tone' is missing from `context`, is not a string,
                        or if the specified 'target_tone' is not supported.
        """
        if not isinstance(data, str):
            error_msg = (
                f"ToneConverterNode requires string data for processing, "
                f"but received type '{type(data).__name__}'."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)

        original_text: str = data
        logger.debug(f"ToneConverterNode received text (first 50 chars): '{original_text[:50]}...'")

        target_tone_raw = context.get("target_tone")
        if not target_tone_raw or not isinstance(target_tone_raw, str):
            error_msg = (
                "Context dictionary for ToneConverterNode must contain a "
                "valid 'target_tone' string."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        target_tone: str = target_tone_raw.lower()
        transformed_text: str = original_text

        # Simulate tone conversion based on the target_tone
        if target_tone == "formal":
            transformed_text = f"Respected reader, {original_text}"
            logger.debug("Text tone converted to 'formal'.")
        elif target_tone == "informal":
            transformed_text = f"Hey there! {original_text}"
            logger.debug("Text tone converted to 'informal'.")
        elif target_tone == "sarcastic":
            transformed_text = f"{original_text} (Oh, how truly fascinating!)"
            logger.debug("Text tone converted to 'sarcastic'.")
        elif target_tone == "joyful":
            transformed_text = f"Yay! {original_text}!!!"
            logger.debug("Text tone converted to 'joyful'.")
        else:
            error_msg = (
                f"Unsupported target tone '{target_tone_raw}' provided for ToneConverterNode. "
                "Supported tones are 'formal', 'informal', 'sarcastic', 'joyful'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Successfully converted text to '{target_tone}' tone.")
        return transformed_text