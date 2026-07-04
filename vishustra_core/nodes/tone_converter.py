import logging
from typing import Any, Dict, Optional

# Assuming vishustra_core is the package root and BaseNode is accessible this way
from vishustra_core.nodes.base_node import BaseNode

logger = logging.getLogger(__name__)

class ToneConverter(BaseNode):
    """
    A processing node designed to convert the tone of input textual data.

    This node expects a string as input 'data' and uses the 'target_tone'
    parameter found in the 'context' dictionary to perform a simulated
    tone transformation.
    """

    @property
    def node_name(self) -> str:
        """Returns the descriptive name of this node."""
        return "ToneConverter"

    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """
        Converts the tone of the input text based on the specified 'target_tone'
        in the context.

        The conversion is a simulation; it applies predefined structural changes
        and stylistic elements to represent different tones.

        Args:
            data: The input data to be processed. Expected to be a string
                  representing the text whose tone needs to be converted.
            context: A dictionary containing operational parameters for the node.
                     It *must* contain a 'target_tone' key with a string value
                     (e.g., 'professional', 'casual', 'neutral').

        Returns:
            The text with its tone converted, or raises an error if the input
            or configuration is invalid.

        Raises:
            ValueError: If 'data' is not a string, 'target_tone' is missing
                        or invalid in the context, or the 'target_tone' is
                        unsupported by this converter.
        """
        if not isinstance(data, str):
            logger.error(
                f"[{self.node_name}] Invalid input data type. Expected 'str', "
                f"received '{type(data).__name__}'. Unable to proceed with tone conversion."
            )
            raise ValueError(
                f"ToneConverter expects string input for 'data', but received '{type(data).__name__}'."
            )

        target_tone: Optional[str] = context.get("target_tone")

        if not target_tone or not isinstance(target_tone, str):
            logger.error(
                f"[{self.node_name}] 'target_tone' parameter is missing or not a string in the context. "
                f"Context received: {context}. This parameter is critical for tone conversion."
            )
            raise ValueError(
                f"'target_tone' (str) is a required parameter in the context for {self.node_name}."
            )

        original_text: str = data.strip()
        converted_text: str = original_text
        lower_target_tone = target_tone.lower()

        # Simulate tone conversion based on predefined rules
        if lower_target_tone == "professional":
            converted_text = (
                f"Dear Sir/Madam,\n\n"
                f"Regarding the matter at hand, please be advised that:\n"
                f"'{original_text}'\n\n"
                f"Your prompt attention to this matter is appreciated.\n\n"
                f"Sincerely,\n"
                f"Vishustra System"
            )
            logger.debug(f"[{self.node_name}] Converted text to a professional tone.")
        elif lower_target_tone == "casual":
            converted_text = (
                f"Hey there!\n\n"
                f"Just wanted to let you know about this:\n"
                f"'{original_text}' :)\n\n"
                f"Catch you later,\n"
                f"Vishustra"
            )
            logger.debug(f"[{self.node_name}] Converted text to a casual tone.")
        elif lower_target_tone == "neutral":
            # For neutral, we simply return the original text, perhaps normalized
            converted_text = original_text
            logger.debug(f"[{self.node_name}] Retained neutral tone (no specific stylistic changes applied).")
        else:
            logger.error(
                f"[{self.node_name}] Unsupported 'target_tone': '{target_tone}'. "
                f"Supported tones are 'professional', 'casual', and 'neutral'."
            )
            raise ValueError(
                f"Unsupported 'target_tone' '{target_tone}'. "
                f"Supported values for {self.node_name} are 'professional', 'casual', and 'neutral'."
            )

        return converted_text

