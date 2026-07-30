import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node that simulates converting the tone of an input text.

    This node expects the input `data` to be a string (the text to convert)
    and the `context` to contain a 'target_tone' key, specifying the desired
    output tone (e.g., 'formal', 'casual', 'sarcastic').
    """

    @property
    def node_name(self) -> str:
        """Returns the name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Converts the tone of the input text based on the 'target_tone' specified
        in the context.

        Args:
            data: The input text, expected to be a string.
            context: A dictionary containing operational context,
                     must include 'target_tone' (str).

        Returns:
            The text with its tone ostensibly converted.

        Raises:
            ValueError: If `data` is not a string, or if 'target_tone' is
                        missing or not a string in the context.
        """
        if not isinstance(data, str):
            logger.error(f"ToneConverterNode expects string data, but received {type(data)}. Node: {self.node_name}")
            raise ValueError(f"ToneConverterNode: Invalid input data type. Expected string, got {type(data)}.")

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone:
            logger.error(
                f"ToneConverterNode requires 'target_tone' (str) in context. Received: {target_tone}. Node: {self.node_name}"
            )
            raise ValueError(
                "ToneConverterNode: 'target_tone' (string) is required in the context."
            )

        # Simulate tone conversion. In a real scenario, this would involve
        # an external service call, a complex NLP model, or an LLM prompt.
        # For demonstration, we append a descriptive suffix.
        processed_data: str = ""
        sanitized_tone = target_tone.lower().strip()

        if sanitized_tone == "formal":
            processed_data = f"{data} (converted to a formal tone)"
        elif sanitized_tone == "casual":
            processed_data = f"{data} (converted to a casual tone)"
        elif sanitized_tone == "sarcastic":
            processed_data = f"{data} (said with a hint of sarcasm)"
        elif sanitized_tone == "neutral":
            processed_data = f"{data} (tone set to neutral)"
        else:
            # For unsupported tones, we can either default or raise an error.
            # Here, we default to a general conversion and log a warning.
            logger.warning(
                f"ToneConverterNode received unsupported target_tone '{target_tone}'. Applying generic conversion. Node: {self.node_name}"
            )
            processed_data = f"{data} (converted to {target_tone} tone)"

        logger.debug(
            f"ToneConverterNode '{self.node_name}' processed text for tone '{target_tone}'."
        )
        return processed_data