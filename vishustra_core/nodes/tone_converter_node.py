import logging
from typing import Any, Dict

# Assuming vishustra_core.nodes.base_node provides the BaseNode class
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)


class ToneConverter(BaseNode):
    """
    A Vishustra processing node responsible for simulating the conversion
    of text tone. It expects text data and a 'target_tone' specified
    within the execution context.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data by simulating a tone conversion.

        This method expects the `data` to be a string (the text to convert)
        and the `context` dictionary to contain a 'target_tone' key
        with a string value indicating the desired tone (e.g., "formal", "casual", "sarcastic").

        Args:
            data (Any): The input data, expected to be a string containing the text.
            context (Dict[str, Any]): A dictionary containing contextual information
                                     required for processing. Must include 'target_tone' (str).

        Returns:
            Any: The processed data, which is a string representing the text
                 with a simulated tone adjustment.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing from the context or is not a
                        non-empty string.
        """
        logger.info(f"[{self.node_name}] Initiating text tone conversion process.")

        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected string, "
                f"received {type(data).__name__}."
            )
            raise TypeError(
                f"[{self.node_name}] Input 'data' must be a string for tone conversion."
            )

        target_tone = context.get("target_tone")

        if not isinstance(target_tone, str) or not target_tone.strip():
            logger.error(
                f"[{self.node_name}] 'target_tone' is missing or invalid in context. "
                f"Expected a non-empty string, received: {target_tone}."
            )
            raise ValueError(
                f"[{self.node_name}] Context must provide a valid 'target_tone' "
                f"(non-empty string) for tone conversion."
            )

        # Simulate the tone conversion. In a real-world scenario with an LLM,
        # this would involve an API call to transform the text based on the
        # target_tone prompt. For this node, we append a descriptive suffix.
        processed_text = f"{data.strip()} (rephrased for a {target_tone.strip()} tone)"

        logger.info(
            f"[{self.node_name}] Successfully simulated conversion to '{target_tone}' tone."
        )
        logger.debug(
            f"[{self.node_name}] Original text (first 100 chars): '{data[:100]}...'"
        )
        logger.debug(
            f"[{self.node_name}] Converted text (first 100 chars): '{processed_text[:100]}...'"
        )

        return processed_text