import logging
from typing import Any, Dict

# Assuming the vishustra_core package structure
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverterNode(BaseNode):
    """
    A processing node designed to simulate converting the tone of text data.
    It expects a 'target_tone' key within the context dictionary to guide
    the simulated tone transformation. This node is useful for demonstrating
    how text attributes can be modified within an orchestration flow.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this processing node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input data, simulating a tone conversion based on
        the 'target_tone' specified in the context dictionary.

        This method performs the following steps:
        1. Validates that `data` is a string and `context` is a dictionary.
        2. Extracts and validates the 'target_tone' from the `context`.
           If 'target_tone' is missing, it defaults to 'neutral' with a warning.
        3. Applies a simulated tone transformation based on predefined tones.
        4. Returns the modified string.

        Args:
            data: The input data, which is expected to be a string
                  for tone conversion.
            context: A dictionary containing operational context. Must include
                     a 'target_tone' key (e.g., 'formal', 'informal', 'enthusiastic', 'neutral').

        Returns:
            Any: The processed data, typically a string with the simulated tone applied.

        Raises:
            TypeError: If `data` is not a string, or `context` is not a dictionary.
            ValueError: If 'target_tone' is present but not a string, or is invalid.
        """
        logger.debug(f"[{self.node_name}] Initiating process for data type: {type(data)}")

        # Validate input data type
        if not isinstance(data, str):
            logger.error(f"[{self.node_name}] Invalid input data type. Expected 'str', received '{type(data).__name__}'.")
            raise TypeError(f"[{self.node_name}] Input 'data' must be a string, but received {type(data).__name__}.")

        # Validate context type
        if not isinstance(context, dict):
            logger.error(f"[{self.node_name}] Invalid context type. Expected 'dict', received '{type(context).__name__}'.")
            raise TypeError(f"[{self.node_name}] Input 'context' must be a dictionary, but received {type(context).__name__}.")

        # Extract and validate 'target_tone' from context
        target_tone = context.get("target_tone")
        if target_tone is None:
            logger.warning(f"[{self.node_name}] 'target_tone' key not found in context. Defaulting to 'neutral' tone.")
            target_tone = "neutral"
        elif not isinstance(target_tone, str):
            logger.error(f"[{self.node_name}] Invalid 'target_tone' type. Expected 'str', received '{type(target_tone).__name__}'.")
            raise ValueError(f"[{self.node_name}] 'target_tone' in context must be a string, but received {type(target_tone).__name__}.")

        processed_data = data
        lower_tone = target_tone.lower()

        # Simulate tone conversion based on the specified target_tone
        if lower_tone == "formal":
            processed_data = f"Regarding the matter at hand, it is imperative to state: {data}. Please consider this formally."
            logger.info(f"[{self.node_name}] Data transformed to a formal tone.")
        elif lower_tone == "informal":
            processed_data = f"Hey there! Just wanted to share: {data}. Keep it chill!"
            logger.info(f"[{self.node_name}] Data transformed to an informal tone.")
        elif lower_tone == "enthusiastic":
            processed_data = f"Absolutely fantastic news! Get ready for this: {data}!!! We are thrilled!"
            logger.info(f"[{self.node_name}] Data transformed to an enthusiastic tone.")
        elif lower_tone == "neutral":
            processed_data = f"Statement for consideration: {data}. No specific emotional inflection intended."
            logger.info(f"[{self.node_name}] Data processed with a neutral tone.")
        else:
            logger.warning(f"[{self.node_name}] Unrecognized target tone '{target_tone}'. Returning original data without modification.")
            # If the tone is not recognized, return the original data without changes
            processed_data = data

        logger.debug(f"[{self.node_name}] Processing complete. Output sample: '{processed_data[:75]}...'")
        return processed_data
