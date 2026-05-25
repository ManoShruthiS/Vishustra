import logging
from typing import Any, Dict

# Assuming vishustra_core is installed and available in the environment
# The BaseNode class from the project context is provided and assumed to be at this path.
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node that simulates converting the tone of an input text.

    This node expects the 'data' to be a string and the 'context' dictionary
    to contain a 'target_tone' key specifying the desired tone.

    Supported tones include: "formal", "casual", "enthusiastic", "concise", "sarcastic".
    """

    # A mapping of supported tones to illustrative phrases for simulation purposes.
    _TONE_PHRASES = {
        "formal": "This message has been formally presented.",
        "casual": "Just letting you know, right?",
        "enthusiastic": "Wow! Isn't this just fantastic?!",
        "concise": "To be brief.",
        "sarcastic": "...Oh, how truly fascinating.",
    }

    @property
    def node_name(self) -> str:
        """
        Returns the descriptive name of the node.
        """
        return "ToneConverterNode"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data to simulate a tone conversion based on the
        'target_tone' specified in the context.

        Args:
            data (Any): The input data, expected to be a string for tone conversion.
            context (Dict[str, Any]): A dictionary containing additional information
                                      needed for processing. Must include 'target_tone'.

        Returns:
            Any: The transformed data (string with tone-specific suffix), or
                 raises an error if input or context is invalid.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing from context or is not supported.
            RuntimeError: For any other unexpected errors during processing.
        """
        if not isinstance(data, str):
            logger.error(
                "ToneConverterNode received non-string data. Type: %s, Data: %s",
                type(data).__name__,
                data
            )
            raise TypeError(
                f"ToneConverterNode requires string data, but received {type(data).__name__}"
            )

        target_tone = context.get("target_tone")

        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                "ToneConverterNode: 'target_tone' is missing or not a string in the context. Context: %s",
                context
            )
            raise ValueError("ToneConverterNode requires a string 'target_tone' in the context.")

        if target_tone not in self._TONE_PHRASES:
            supported_tones_list = ', '.join(self._TONE_PHRASES.keys())
            logger.error(
                "ToneConverterNode: Unsupported 'target_tone' '%s'. Supported tones: %s",
                target_tone,
                supported_tones_list
            )
            raise ValueError(
                f"Unsupported target tone '{target_tone}'. "
                f"Supported tones are: {supported_tones_list}"
            )

        try:
            # Simulate tone conversion by appending a tone-specific phrase
            tone_phrase = self._TONE_PHRASES[target_tone]
            transformed_data = f"{data} {tone_phrase}"
            
            logger.debug(
                "ToneConverterNode: Converted data to '%s' tone. Original: '%s', Transformed: '%s'",
                target_tone,
                data,
                transformed_data
            )
            return transformed_data
        except Exception as e:
            logger.exception(
                "ToneConverterNode: An unexpected error occurred during tone conversion for data: '%s'",
                data
            )
            raise RuntimeError(
                f"Failed to process data in ToneConverterNode due to an internal error: {e}"
            ) from e