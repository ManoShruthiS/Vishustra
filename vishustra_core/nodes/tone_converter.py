import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to simulate the conversion of input text to a
    specified target tone.
    
    This node expects the input `data` to be a string representing the text
    to be converted. The `context` dictionary must contain a 'target_tone'
    key, specifying the desired tone (e.g., 'professional', 'casual',
    'sarcastic').
    
    The actual conversion logic is simulated for demonstration purposes,
    appending a descriptive tag to the original text.
    """

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of this node.
        """
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating a tone conversion based on
        the 'target_tone' provided in the context.

        Args:
            data: The input data, expected to be a string.
            context: A dictionary containing operational parameters.
                     Must include 'target_tone' (str).

        Returns:
            A string representing the input data converted to the target tone.

        Raises:
            TypeError: If the input `data` is not a string.
            ValueError: If 'target_tone' is missing from the `context`.
        """
        logger.info("ToneConverterNode received data for processing.")

        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode expects input 'data' to be a string, "
                "but received type: %s", type(data).__name__
            )
            raise TypeError("ToneConverterNode requires 'data' to be a string.")

        if 'target_tone' not in context:
            logger.error(
                "ToneConverterNode requires 'target_tone' in context, "
                "but it was not found."
            )
            raise ValueError("Context must contain a 'target_tone' key.")

        target_tone = context['target_tone']
        if not isinstance(target_tone, str) or not target_tone:
            logger.error(
                "ToneConverterNode requires 'target_tone' to be a non-empty string, "
                "but received: %s (type: %s)", target_tone, type(target_tone).__name__
            )
            raise ValueError("'target_tone' in context must be a non-empty string.")

        logger.debug(
            "Attempting to convert text to '%s' tone. Original text length: %d",
            target_tone, len(data)
        )

        # Simulate tone conversion. In a real scenario, this would involve
        # an LLM call or a sophisticated NLP library.
        # For demonstration, we simply append a descriptive tag.
        converted_text = f"{data} [Converted to {target_tone} tone]"

        logger.info(
            "Successfully simulated tone conversion to '%s' tone. "
            "Result length: %d", target_tone, len(converted_text)
        )
        return converted_text
