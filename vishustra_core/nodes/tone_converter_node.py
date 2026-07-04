import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ToneConverterNode(BaseNode):
    """
    A processing node designed to convert the tone of input text.

    This node expects a 'target_tone' key in the context dictionary,
    which specifies the desired output tone (e.g., 'formal', 'informal', 'neutral', 'sarcastic').
    The node simulates the tone conversion process.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> str:
        """
        Processes the input data, typically a string, to simulate a tone conversion
        based on the 'target_tone' specified in the context.

        Args:
            data: The input data, expected to be a string representing text.
            context: A dictionary containing operational parameters for the node.
                     Must include 'target_tone' (str) specifying the desired tone.

        Returns:
            A string representing the input text with its tone simulated to be converted.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_tone' is missing from `context`, is not a string,
                        is empty, or specifies an unsupported tone.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode: Invalid input data type. Expected 'str', but received '%s'.",
                type(data).__name__,
            )
            raise TypeError(
                f"ToneConverterNode expects 'data' to be a string, but received {type(data).__name__}"
            )

        target_tone_raw = context.get("target_tone")

        if not isinstance(target_tone_raw, str) or not target_tone_raw:
            logger.error(
                "ToneConverterNode: 'target_tone' is missing or invalid in context. "
                "Expected a non-empty string for 'target_tone'."
            )
            raise ValueError(
                "ToneConverterNode requires a non-empty string value for 'target_tone' in the context dictionary."
            )

        target_tone = target_tone_raw.lower()
        converted_text: str

        # Simulate tone conversion based on the specified target_tone
        if target_tone == "formal":
            converted_text = f"Regarding the matter at hand, it is observed that: '{data}' (tone adjusted to formal)."
            logger.info("ToneConverterNode: Successfully converted text to formal tone.")
        elif target_tone == "informal":
            converted_text = f"Hey, just wanted to let you know: '{data}' (pretty informal, right?)."
            logger.info("ToneConverterNode: Successfully converted text to informal tone.")
        elif target_tone == "neutral":
            converted_text = f"The information provided is: '{data}' (presented in a neutral tone)."
            logger.info("ToneConverterNode: Successfully converted text to neutral tone.")
        elif target_tone == "sarcastic":
            converted_text = f"Oh, how absolutely *fascinating*! '{data}' (infused with a delightful layer of sarcasm)."
            logger.info("ToneConverterNode: Successfully converted text to sarcastic tone.")
        else:
            logger.error(
                "ToneConverterNode: Unsupported 'target_tone' specified: '%s'.",
                target_tone_raw,
            )
            raise ValueError(
                f"Unsupported 'target_tone': '{target_tone_raw}'. "
                "Supported tones are 'formal', 'informal', 'neutral', 'sarcastic'."
            )

        logger.debug(
            "ToneConverterNode: Processed data (original: '%s'...) with target_tone '%s'. Output: '%s'...",
            data[:50], target_tone_raw, converted_text[:50]
        )
        return converted_text