import logging
from typing import Any, Dict

from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A Vishustra node designed to simulate the conversion of text tone.

    This node expects a 'target_tone' key in the context dictionary, specifying the
    desired tone (e.g., 'formal', 'informal', 'sarcastic', 'neutral').
    The 'data' input is expected to be a string.
    """

    @property
    def node_name(self) -> str:
        """Returns the programmatic name of the node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Processes the input text data, attempting to convert its tone based on the
        'target_tone' provided in the context dictionary.

        Args:
            data: The input text (string) to be transformed.
            context: A dictionary containing operational parameters for the node.
                     Must include 'target_tone' (str) to specify the desired output tone.

        Returns:
            A string representing the input text with the simulated converted tone.

        Raises:
            TypeError: If the input 'data' is not a string.
            ValueError: If 'target_tone' is missing from the context or is not a valid string.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"but received '{type(data).__name__}'. Aborting process."
            )
            raise TypeError(
                f"Input data for '{self.node_name}' must be a string, "
                f"but got '{type(data).__name__}'."
            )

        target_tone = context.get("target_tone")
        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                f"[{self.node_name}] 'target_tone' not found or is invalid in context. "
                "Expected a string value for 'target_tone'. Aborting process."
            )
            raise ValueError(
                f"Context for '{self.node_name}' must contain a valid 'target_tone' (str)."
            )

        logger.info(f"[{self.node_name}] Initiating tone conversion to '{target_tone}' for input data.")
        
        converted_text = data
        # For logging, truncate long input data
        original_data_preview = data[:70] + "..." if len(data) > 70 else data

        # Simulate tone conversion based on the specified target_tone
        lower_target_tone = target_tone.lower()
        if lower_target_tone == "formal":
            converted_text = f"Regarding the aforementioned subject, it is imperative to convey: '{data}'."
        elif lower_target_tone == "informal":
            converted_text = f"Hey, just wanted to let you know: '{data}'."
        elif lower_target_tone == "sarcastic":
            converted_text = f"Oh, how truly insightful; precisely what we needed to hear: '{data}'."
        elif lower_target_tone == "neutral":
            # For a 'neutral' tone, no transformative action is simulated.
            pass
        else:
            logger.warning(
                f"[{self.node_name}] Unsupported target tone '{target_tone}' requested. "
                "Returning original data without conversion. "
                "Consider defining this tone or handling it as an error."
            )
            # In a production system, one might raise a ValueError here for strictness
            # Or pass it to an actual LLM if available to handle "any" tone.

        # For logging, truncate long output data
        converted_data_preview = converted_text[:70] + "..." if len(converted_text) > 70 else converted_text
        
        logger.debug(
            f"[{self.node_name}] Successfully processed. Original: '{original_data_preview}', "
            f"Converted ('{target_tone}'): '{converted_data_preview}'."
        )

        return converted_text