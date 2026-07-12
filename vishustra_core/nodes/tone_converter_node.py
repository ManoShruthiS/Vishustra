import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ToneConverterNode(BaseNode):
    """
    A Vishustra processing node designed to simulate converting the tone of an input text.
    The desired tone for conversion is specified via the 'target_tone' key within the
    'context' dictionary provided during processing.
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, simulating a tone conversion based on the
        'target_tone' specified in the context.

        Args:
            data (Any): The input data to be processed. Expected to be a string
                        containing the text whose tone needs to be converted.
            context (Dict[str, Any]): A dictionary containing additional processing
                                       parameters. It must contain a 'target_tone' key
                                       whose value is a string representing the desired tone.

        Returns:
            Any: The simulated tone-converted text (string). If the input data is not
                 a string, or if 'target_tone' is missing/invalid in context,
                 the original data or a less-transformed version may be returned
                 after logging appropriate warnings/errors.

        Raises:
            ValueError: If the input 'data' is not of type string, as this node
                        is designed to operate on textual content.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"but received {type(data)}. Cannot perform tone conversion."
            )
            raise ValueError(
                f"[{self.node_name}] Input data must be a string for tone conversion. "
                f"Received type: {type(data)}."
            )

        original_text = data
        target_tone = context.get("target_tone")

        if not target_tone or not isinstance(target_tone, str):
            logger.warning(
                f"[{self.node_name}] 'target_tone' not found or is not a string in "
                f"context. Cannot convert tone. Returning original text."
            )
            return original_text

        logger.info(
            f"[{self.node_name}] Attempting to convert tone for text (first 80 chars): "
            f"'{original_text[:80].replace('\'', '\"')}...' to target tone: '{target_tone}'."
        )

        converted_text = original_text  # Default to original if no specific conversion

        # Simple simulation of tone conversion based on common target tones
        # In a real-world scenario, this would involve NLP models or sophisticated templating.
        lower_target_tone = target_tone.lower()
        if lower_target_tone == "formal":
            converted_text = (
                f"Esteemed recipient, kindly be advised that, following due consideration, "
                f"it has been determined: {original_text}. Your prompt attention to this matter "
                f"is greatly appreciated. Respectfully."
            )
        elif lower_target_tone == "informal":
            converted_text = (
                f"Hey there! Just wanted to quickly fill you in: {original_text}. "
                f"Catch ya later!"
            )
        elif lower_target_tone == "sarcastic":
            converted_text = (
                f"Oh, how absolutely *thrilling*! Prepare to be astonished by this "
                f"earth-shattering revelation: {original_text}. What a truly unique insight!"
            )
        elif lower_target_tone == "joyful":
            converted_text = (
                f"Wonderful news! I'm absolutely delighted to share: {original_text}! "
                f"Isn't that just fantastic?!"
            )
        elif lower_target_tone == "serious":
            converted_text = (
                f"Please take a moment to absorb this critical information: {original_text}. "
                f"The gravity of this situation cannot be overstated."
            )
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported 'target_tone': '{target_tone}'. "
                f"Returning original text without conversion."
            )

        logger.info(
            f"[{self.node_name}] Tone conversion simulated successfully. Result (first 80 chars): "
            f"'{converted_text[:80].replace('\'', '\"')}...'."
        )
        return converted_text