import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node exists as per the project context
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra node that simulates converting the tone of text data
    based on a specified target tone provided in the context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data (expected to be a string) and converts its
        simulated tone based on the 'target_tone' key in the context dictionary.

        Supported tones for simulation: 'formal', 'casual', 'professional', 'empathetic'.

        Args:
            data: The input text data to be converted. Must be a string.
            context: A dictionary containing operational parameters.
                     Requires a 'target_tone' key with a string value.

        Returns:
            The text data with its simulated tone adjusted. Returns original data
            if target_tone is unsupported, logging a warning.

        Raises:
            TypeError: If 'data' is not a string.
            ValueError: If 'target_tone' is missing from context or is not a non-empty string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid data type received. Expected str, got {type(data).__name__}."
            )
            raise TypeError(f"[{self.node_name}] 'data' must be a string.")

        target_tone = context.get("target_tone")
        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error(
                f"[{self.node_name}] Required context key 'target_tone' missing or invalid. "
                f"Expected a non-empty string, got: {target_tone!r}"
            )
            raise ValueError(f"[{self.node_name}] Context must contain a non-empty string for 'target_tone'.")

        processed_data = data
        lower_tone = target_tone.strip().lower()

        # Simulate tone conversion. In a real scenario, this would involve an LLM call.
        if lower_tone == "formal":
            processed_data = f"In a formal context, the preceding statement can be expressed as: {data}"
            logger.info(f"[{self.node_name}] Converted text to formal tone (first 50 chars: '{data[:50]}...')")
        elif lower_tone == "casual":
            processed_data = f"Hey there, just wanted to share: {data.lower()}... Pretty chill, right?"
            logger.info(f"[{self.node_name}] Converted text to casual tone (first 50 chars: '{data[:50]}...')")
        elif lower_tone == "professional":
            processed_data = f"From a professional standpoint, we can state: {data}"
            logger.info(f"[{self.node_name}] Converted text to professional tone (first 50 chars: '{data[:50]}...')")
        elif lower_tone == "empathetic":
            processed_data = f"I hear what you're saying, and I understand. Here's a thought: {data}"
            logger.info(f"[{self.node_name}] Converted text to empathetic tone (first 50 chars: '{data[:50]}...')")
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported target tone '{target_tone}'. "
                "Returning original data. Supported tones: formal, casual, professional, empathetic."
            )
            # For unsupported tones, we return the original data without modification, logging a warning.

        return processed_data
